"""Exercise real command ownership and durable bounded-run records."""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest

from scripts import run_bounded
from scripts.run_bounded import is_compiler_telemetry

RUNNER = Path(__file__).resolve().parents[1] / "scripts" / "run_bounded.py"
CHILD_PYTHON = getattr(sys, "_base_executable", sys.executable)


def arguments(output: Path, code: str, *, seconds: float | None = 20, rss: float = 1) -> list[str]:
    """Keep each test command within an explicit small resource bound."""
    return [
        "--output-dir",
        str(output),
        *(["--no-deadline"] if seconds is None else ["--max-seconds", str(seconds)]),
        "--max-rss-gib",
        str(rss),
        "--",
        CHILD_PYTHON,
        "-c",
        code,
    ]


def wait_file(path: Path, process: subprocess.Popen[str], timeout: float = 30) -> None:
    """Fail with command evidence instead of an indefinite wait."""
    deadline = time.monotonic() + timeout
    while not path.exists():
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            pytest.fail(f"The supervisor exits before {path}: {process.returncode}\n{stdout}\n{stderr}")
        if time.monotonic() >= deadline:
            pytest.fail(f"The test command does not create {path}")
        time.sleep(0.015)


def start(args: list[str], cwd: Path, prefix: list[str] | None = None) -> subprocess.Popen[str]:
    """Use real processes without a shell or a dependency setup step."""
    return subprocess.Popen(
        (prefix or [sys.executable, str(RUNNER)]) + args,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )


def finish(process: subprocess.Popen[str]) -> tuple[str, str]:
    """Bound the test supervisor lifetime."""
    try:
        return process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        for child in psutil.Process(process.pid).children(recursive=True)[::-1]:
            try:
                child.kill()
            except psutil.NoSuchProcess:
                pass
        process.kill()
        process.communicate(timeout=5)
        raise


def test_lifecycle_records_threads_and_output_verification(tmp_path: Path, subtests: pytest.Subtests) -> None:
    """Verify full lifecycle records, no-deadline mode, thread/disk budgets, missing output, and exit codes."""
    with subtests.test("complete-no-deadline-and-thread-budget"):
        output = tmp_path / "complete"
        code = (
            "import json, os, sys, time; from pathlib import Path; "
            "time.sleep(0.12); Path('made.txt').write_text('ok'); "
            "Path('env.json').write_text(json.dumps({k: os.environ.get(k) for k in "
            "['OMP_NUM_THREADS','RAYON_NUM_THREADS','POLARS_MAX_THREADS','LOKY_MAX_CPU_COUNT',"
            "'OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})); "
            "print('out'); print('err', file=sys.stderr)"
        )
        args = [
            "--threads",
            "2",
            "--min-free-gib",
            "0.001",
            "--expected-output",
            "made.txt",
            "--expected-output",
            "env.json",
            *arguments(output, code, seconds=None),
        ]
        args[args.index("--") + 1] = sys.executable
        process = start(args, tmp_path)
        finish(process)
        result = json.loads((output / "result.json").read_text(encoding="utf-8"))
        initial = json.loads((output / "run.json").read_text(encoding="utf-8"))
        assert process.returncode == 0
        assert result["status"] == "complete"
        assert result["command_exit_code"] == 0
        assert result["max_seconds"] is None and initial["max_seconds"] is None
        assert initial["command_pid"] == result["command_pid"] and initial["status"] == "running"
        assert result["max_rss_bytes"] == 1024**3
        assert result["rss_samples"] > 0
        assert result["cleanup"]["active_processes_after"] == 0
        assert result["expected_outputs"][0]["size_bytes"] == 2
        assert (output / "stdout.log").read_text().strip() == "out"
        assert (output / "stderr.log").read_text().strip() == "err"
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        assert events[0]["event"] == "start" and events[-1]["status"] == "complete"
        observed = json.loads((tmp_path / "env.json").read_text(encoding="utf-8"))
        for primary in ("OMP_NUM_THREADS", "RAYON_NUM_THREADS", "POLARS_MAX_THREADS", "LOKY_MAX_CPU_COUNT"):
            assert observed[primary] == "2"
        for secondary in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            assert observed[secondary] == "1"
        assert result["thread_budget"]["allocated_threads"] == 2
        assert result["cpu_capacity"]["logical_cpus"] >= 1
        assert result["disk_free_bytes_at_start"] > 0
        assert "git_head" in result

    with subtests.test("missing-output-and-nonzero-exit"):
        missing_out = tmp_path / "missing"
        args = [
            "--expected-output",
            "missing.txt",
            "--expected-output",
            "made2.txt",
            *arguments(missing_out, "from pathlib import Path; Path('made2.txt').write_text('ok')"),
        ]
        proc_missing = start(args, tmp_path)
        finish(proc_missing)
        res_missing = json.loads((missing_out / "result.json").read_text())
        assert proc_missing.returncode == 1
        assert res_missing["status"] == "missing_output"
        assert res_missing["command_exit_code"] == 0
        assert [row["is_file"] for row in res_missing["expected_outputs"]] == [False, True]

        fail_out = tmp_path / "exit7"
        proc_fail = start(arguments(fail_out, "raise SystemExit(7)"), tmp_path)
        finish(proc_fail)
        res_fail = json.loads((fail_out / "result.json").read_text())
        assert proc_fail.returncode == 1
        assert res_fail["status"] == "command_failed"
        assert res_fail["command_exit_code"] == 7


def test_prelaunch_validation_and_telemetry_matching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, subtests: pytest.Subtests
) -> None:
    """Reject invalid CLI limits, stale/reserved outputs, over-budget threads, and low disk without launching."""
    marker = tmp_path / "launched.txt"
    touch_code = f"from pathlib import Path; Path({str(marker)!r}).touch()"

    for limits in (
        ["--max-rss-gib", "1"],
        ["--max-seconds", "1", "--no-deadline", "--max-rss-gib", "1"],
        ["--no-deadline"],
        ["--no-deadline", "--max-rss-gib", "inf"],
        ["--max-seconds", "0", "--max-rss-gib", "1"],
    ):
        with subtests.test(limits=limits):
            output = tmp_path / "invalid"
            monkeypatch.setattr(
                sys, "argv", ["run_bounded", "--output-dir", str(output), *limits, "--", CHILD_PYTHON, "-c", touch_code]
            )
            with pytest.raises(SystemExit) as raised:
                run_bounded.main()
            assert raised.value.code == 2
            assert not marker.exists() and not output.exists()

    with subtests.test("preexisting-expected-output"):
        stale = tmp_path / "stale.txt"
        stale.write_text("old")
        output = tmp_path / "stale_out"
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_bounded", "--expected-output", str(stale), *arguments(output, touch_code)],
        )
        with pytest.raises(SystemExit) as raised:
            run_bounded.main()
        assert raised.value.code == 1
        assert stale.read_text() == "old"
        assert not marker.exists() and not output.exists()

    if os.name == "nt":
        with subtests.test("reserved-output-case-insensitive"):
            output = tmp_path / "reserved"
            monkeypatch.setattr(
                sys,
                "argv",
                ["run_bounded", "--expected-output", str(output / "RUN.JSON"), *arguments(output, touch_code)],
            )
            with pytest.raises(SystemExit) as raised:
                run_bounded.main()
            assert raised.value.code == 1
            assert not marker.exists() and not output.exists()

    with subtests.test("insufficient-disk-space"):
        output = tmp_path / "disk_fail"
        monkeypatch.setattr(
            sys,
            "argv",
            ["run_bounded", "--min-free-gib", "1000000000", *arguments(output, touch_code)],
        )
        with pytest.raises(SystemExit) as raised:
            run_bounded.main()
        assert raised.value.code == 1
        assert not marker.exists() and not output.exists()

    with subtests.test("thread-budget-and-telemetry-path-rules"):
        with pytest.raises(ValueError, match="exceeds available budget"):
            run_bounded.build_thread_environment(1000)
        env_override, _ = run_bounded.build_thread_environment(2, omp_threads=1, allow_over_budget=True)
        assert env_override["OMP_NUM_THREADS"] == "1" and env_override["RAYON_NUM_THREADS"] == "2"

        for image, expected in (
            (r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\bin\Hostx64\x64\vctip.exe", True),
            (r"C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\bin\HostX86\x64\VCTIP.EXE", True),
            (r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\mspdbsrv.exe", False),
            (r"C:\Program Files\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\bin\Hostx64\x64\link.exe", False),
            (r"C:\tools\vctip.exe", False),
            ("vctip.exe", False),
        ):
            assert is_compiler_telemetry(image) is expected, image


def test_real_resource_limits(tmp_path: Path, subtests: pytest.Subtests) -> None:
    cases = (
        ("deadline", 0.25, 1.0, "import time; time.sleep(60)"),
        (
            "memory_limit",
            None,
            0.08,
            "import subprocess,sys,time; allocation=bytearray(40*1024**2); "
            "subprocess.Popen([sys.executable,'-c','import time; allocation=bytearray(40*1024**2); time.sleep(60)']); "
            "time.sleep(60)",
        ),
    )
    for status, seconds, rss, code in cases:
        with subtests.test(status=status):
            output = tmp_path / status
            process = start(arguments(output, code, seconds=seconds, rss=rss), tmp_path)
            finish(process)
            result = json.loads((output / "result.json").read_text())
            assert process.returncode == 1
            assert result["status"] == status
            assert result["max_seconds"] == seconds
            assert result["command_exit_code"] is not None
            assert result["cleanup"]["forced"]
            assert result["cleanup"]["active_processes_after"] == 0
            assert not psutil.pid_exists(result["command_pid"])
            if status == "memory_limit":
                assert result["peak_tree_rss_bytes"] > result["max_rss_bytes"]


@contextmanager
def windows_read_lock(path: Path) -> Iterator[None]:
    """Deny writes and delete operations through a real Windows file handle."""
    from ctypes import WinDLL, WinError, get_last_error, wintypes

    kernel = WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    deadline = time.monotonic() + 3
    while True:
        handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0x80, None)
        if handle != ctypes.c_void_p(-1).value:
            break
        error = get_last_error()
        if error not in (32, 33) or time.monotonic() >= deadline:
            raise WinError(error)
        time.sleep(0.01)
    try:
        yield
    finally:
        kernel.CloseHandle(handle)


@pytest.mark.skipif(os.name != "nt", reason="The regression uses real Windows share locks")
def test_reader_lock_and_telemetry_error_do_not_abandon_worker(tmp_path: Path) -> None:
    output = tmp_path / "locked"
    marker = tmp_path / "completed.txt"
    gate = tmp_path / "release_child.txt"
    code = (
        f"import time; from pathlib import Path\n"
        f"while not Path({str(gate)!r}).exists(): time.sleep(0.01)\n"
        f"Path({str(marker)!r}).write_text('done')"
    )
    inject = (
        f"import sys; sys.path.insert(0,{str(RUNNER.parents[1])!r}); from scripts import run_bounded as runner\n"
        "runner.POLL_SECONDS = 0.02\n"
        "runner.PROGRESS_INTERVAL_SECONDS = 0.04\n"
        "raise SystemExit(runner.main())\n"
    )
    args = ["--expected-output", str(marker), *arguments(output, code)]
    process = start(args, tmp_path, [sys.executable, "-c", inject])
    try:
        wait_file(output / "run.json", process)
        wait_file(output / "events.jsonl", process)
        original = (output / "run.json").read_bytes()
        with windows_read_lock(output / "run.json"):
            with windows_read_lock(output / "events.jsonl"):
                time.sleep(0.18)
                gate.write_text("go")
            wait_file(output / "result.json", process)
            assert (output / "run.json").read_bytes() == original
        _, stderr = finish(process)
        result = json.loads((output / "result.json").read_text())
        assert process.returncode == 0
        assert result["status"] == "complete"
        assert result["command_exit_code"] == 0
        assert result["telemetry_errors"]
        assert stderr.count("Warning: telemetry I/O error:") == 1
        assert marker.read_text() == "done"
    finally:
        gate.write_text("go")
        if process.poll() is None:
            process.kill()
            finish(process)


def family_code(marker: Path, *, parent_exits: bool = False) -> str:
    """Create a real child and grandchild before the failure trigger."""
    grandchild = (
        "import json,os,sys,time; from pathlib import Path; "
        f"Path({str(marker) + '.tmp'!r}).write_text("
        "json.dumps({'child':int(sys.argv[1]),'grandchild':os.getpid()})); "
        f"Path({str(marker) + '.tmp'!r}).rename({str(marker)!r}); "
        "time.sleep(60)"
    )
    code = (
        "import os,subprocess,sys,time; from pathlib import Path; "
        f"subprocess.Popen([sys.executable,'-c',{grandchild!r},str(os.getpid())])\n"
    )
    return code + (f"while not Path({str(marker)!r}).exists(): time.sleep(0.01)" if parent_exits else "time.sleep(60)")


def assert_family_stops(marker: Path) -> None:
    """Check process exit rather than process disappearance from a parent list."""
    observed = json.loads(marker.read_text())
    pids = observed.get("members", list(observed.values()))
    deadline = time.monotonic() + 5
    while any(psutil.pid_exists(pid) for pid in pids):
        if time.monotonic() >= deadline:
            pytest.fail(f"Owned family remains alive: {pids}")
        time.sleep(0.02)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object family containment")
def test_windows_job_family_containment(tmp_path: Path, subtests: pytest.Subtests) -> None:
    with subtests.test("supervisor-exception"):
        output, marker = tmp_path / "exception", tmp_path / "family_exc.json"
        inject = (
            f"import sys,json; sys.path.insert(0,{str(RUNNER.parents[1])!r})\n"
            "from scripts import run_bounded as runner\n"
            "from pathlib import Path\n"
            "runner.POLL_SECONDS = 0.02\n"
            "original=runner.tree_rss\n"
            "def fail_after_grandchild(process,job):\n"
            "    members=job.pids()\n"
            "    for pid in members:\n"
            "        try: argv=runner.psutil.Process(pid).cmdline()\n"
            "        except (runner.psutil.NoSuchProcess,runner.psutil.AccessDenied): continue\n"
            "        if len(argv)>=4 and argv[-1].isdecimal() and int(argv[-1]) in members:\n"
            f"            Path({str(marker)!r}).write_text("
            "json.dumps({'members':members,'child':int(argv[-1]),'grandchild':pid}))\n"
            "            raise RuntimeError('injected supervisor failure')\n"
            "    return original(process,job)\n"
            "runner.tree_rss=fail_after_grandchild\n"
            "raise SystemExit(runner.main())\n"
        )
        process = start(
            arguments(output, family_code(tmp_path / "ready.json")), tmp_path, [sys.executable, "-c", inject]
        )
        finish(process)
        result = json.loads((output / "result.json").read_text())
        assert process.returncode == 1
        assert result["status"] == "supervisor_error"
        assert result["error"] == "RuntimeError: injected supervisor failure"
        assert result["cleanup"]["active_processes_before"] >= 2
        assert result["cleanup"]["active_processes_after"] == 0
        observed = json.loads(marker.read_text())
        assert observed["child"] in observed["members"] and observed["grandchild"] in observed["members"]
        assert_family_stops(marker)

    for delay_initial_record in (False, True):
        with subtests.test(hard_death_delayed_record=delay_initial_record):
            sub = tmp_path / f"hard_{int(delay_initial_record)}"
            sub.mkdir()
            output, marker = sub / "hard_death", sub / "family.json"
            prefix = None
            if delay_initial_record:
                inject = (
                    f"import sys,time; sys.path.insert(0,{str(RUNNER.parents[1])!r})\n"
                    "from scripts import run_bounded as runner\n"
                    "original=runner.publish\n"
                    "def delayed_publish(path,payload):\n"
                    "    if path.name == 'run.json': time.sleep(0.12)\n"
                    "    return original(path,payload)\n"
                    "runner.publish=delayed_publish\n"
                    "raise SystemExit(runner.main())\n"
                )
                prefix = [sys.executable, "-c", inject]
            process = start(arguments(output, family_code(marker)), sub, prefix)
            try:
                wait_file(marker, process)
                wait_file(output / "run.json", process)
                initial = json.loads((output / "run.json").read_text())
                assert initial["containment"] == "windows_kill_on_close_job"
                psutil.Process(initial["supervisor_pid"]).kill()
                finish(process)
                assert_family_stops(marker)
                assert not (output / "result.json").exists()
            finally:
                if process.poll() is None:
                    process.kill()
                    finish(process)

    with subtests.test("live-grandchild-holds-job-until-deadline"):
        output, marker = tmp_path / "remaining_family", tmp_path / "family_live.json"
        # Leave the parent time to exit before the deadline: the grandchild starts Python first, slowly under load.
        process = start(arguments(output, family_code(marker, parent_exits=True), seconds=3), tmp_path)
        finish(process)
        result = json.loads((output / "result.json").read_text())
        assert process.returncode == 1
        assert result["command_exit_code"] == 0
        assert result["status"] == "deadline"
        assert_family_stops(marker)


def test_fast_exit_and_record_alias_guards(tmp_path: Path, subtests: pytest.Subtests) -> None:
    for idx, (seconds, status) in enumerate(((20.0, "complete"), (0.05, "deadline"))):
        with subtests.test(fast_exit_status=status):
            output = tmp_path / f"fast_{idx}"
            inject = (
                f"import sys,time; sys.path.insert(0,{str(RUNNER.parents[1])!r})\n"
                "from scripts import run_bounded as runner\n"
                "original=runner.publish\n"
                "def slow_initial(path,payload):\n"
                "    if path.name == 'run.json':\n"
                "        try: runner.psutil.Process(payload['command_pid']).wait(timeout=10)\n"
                "        except runner.psutil.NoSuchProcess: pass\n"
                "        time.sleep(0.08)\n"
                "    return original(path,payload)\n"
                "runner.publish=slow_initial\n"
                "raise SystemExit(runner.main())\n"
            )
            command = [os.environ.get("COMSPEC", "cmd.exe"), "/d", "/c", "exit 0"] if os.name == "nt" else ["/bin/true"]
            args = arguments(output, "unused", seconds=seconds)
            args[args.index("--") + 1 :] = command
            process = start(args, tmp_path, [sys.executable, "-c", inject])
            finish(process)
            result = json.loads((output / "result.json").read_text())
            assert process.returncode == int(status != "complete")
            assert result["status"] == status
            assert result["command_exit_code"] == 0

    with subtests.test("external-hardlink-alias"):
        output, expected = tmp_path / "hard_link", tmp_path / "artifact.txt"
        record = output / "run.json"
        code = (
            f"import os,time; from pathlib import Path; source=Path({str(record)!r})\n"
            "while not source.exists(): time.sleep(0.01)\n"
            f"os.link(source,{str(expected)!r})"
        )
        args = ["--expected-output", str(expected), *arguments(output, code)]
        process = start(args, tmp_path)
        finish(process)
        result = json.loads((output / "result.json").read_text())
        assert os.path.samefile(expected, record)
        assert process.returncode == 1
        assert result["status"] == "invalid_output"
        assert result["expected_outputs"][0]["runner_record_alias"]

    if os.name == "nt":
        with subtests.test("windows-trailing-dot-space-alias"):
            output = tmp_path / "win_alias"
            expected = output / "run.json."
            args = ["--expected-output", str(expected), *arguments(output, "pass")]
            process = start(args, tmp_path)
            finish(process)
            result = json.loads((output / "result.json").read_text())
            assert os.path.samefile(expected, output / "run.json")
            assert process.returncode == 1
            assert result["status"] == "invalid_output"
            assert result["expected_outputs"][0]["runner_record_alias"]


def fake_uploader(directory: Path) -> Path:
    """Copy the interpreter and the DLLs it loads under the telemetry image name."""
    source = Path(CHILD_PYTHON)
    directory.mkdir(parents=True)
    for library in source.parent.glob("*.dll"):
        shutil.copy2(library, directory / library.name)
    image = directory / "vctip.exe"
    shutil.copy2(source, image)
    return image


@pytest.mark.skipif(os.name != "nt", reason="MSVC telemetry cleanup uses Windows Job Objects")
def test_lingering_compiler_telemetry(tmp_path: Path, subtests: pytest.Subtests) -> None:
    cases = (
        ("vs_uploader", Path("Microsoft Visual Studio", "2022", "BuildTools", "VC", "bin"), 20.0, "complete"),
        # The uploader must start and write its marker before this deadline, slowly under load.
        ("non_vs_tool", Path("tools"), 1.5, "deadline"),
    )
    for label, install, seconds, status in cases:
        with subtests.test(case=label):
            case_dir = tmp_path / label
            output, marker = case_dir / "telemetry", case_dir / "uploader.json"
            image = fake_uploader(case_dir / install)
            uploader = (
                "import json,os,time; from pathlib import Path; "
                f"Path({str(marker) + '.tmp'!r}).write_text(json.dumps({{'members':[os.getpid()]}})); "
                f"Path({str(marker) + '.tmp'!r}).rename({str(marker)!r}); time.sleep(60)"
            )
            code = (
                "import os,subprocess,time; from pathlib import Path\n"
                f"env = dict(os.environ, PYTHONHOME={sys.base_prefix!r})\n"
                f"subprocess.Popen([{str(image)!r}, '-c', {uploader!r}], env=env)\n"
                f"while not Path({str(marker)!r}).exists(): time.sleep(0.01)"
            )
            inject = (
                f"import sys; sys.path.insert(0,{str(RUNNER.parents[1])!r})\n"
                "from scripts import run_bounded as runner\n"
                "runner.POLL_SECONDS = 0.02\n"
                "runner.TELEMETRY_GRACE_SECONDS = 0.08\n"
                "raise SystemExit(runner.main())\n"
            )
            process = start(arguments(output, code, seconds=seconds), case_dir, [sys.executable, "-c", inject])
            finish(process)
            result = json.loads((output / "result.json").read_text())
            pid = json.loads(marker.read_text())["members"][0]
            assert process.returncode == int(status != "complete")
            assert result["status"] == status
            assert result["command_exit_code"] == 0
            if status == "complete":
                assert [row["pid"] for row in result["telemetry_cleanup"]] == [pid]
                assert os.path.samefile(result["telemetry_cleanup"][0]["image"], image)
                assert not result["cleanup"]["forced"]
            else:
                assert result["telemetry_cleanup"] == []
                assert result["cleanup"]["forced"]
            assert_family_stops(marker)
