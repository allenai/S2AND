"""Exercise real command ownership and durable bounded-run records."""

from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import psutil
import pytest

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
        time.sleep(0.025)


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


@pytest.mark.parametrize("missing", [False, True])
def test_output_verification(tmp_path: Path, missing: bool) -> None:
    output = tmp_path / "success"
    command = (
        "from pathlib import Path; import sys; Path('made.txt').write_text('ok'); "
        "print('out'); print('err',file=sys.stderr)"
    )
    args = arguments(output, command)
    if not missing:
        args[args.index("--") + 1] = sys.executable
    args[0:0] = ["--expected-output", "made.txt"]
    if missing:
        args[0:0] = ["--expected-output", "missing.txt"]
    process = start(args, tmp_path)
    finish(process)
    result = json.loads((output / "result.json").read_text())
    assert process.returncode == int(missing)
    assert result["status"] == ("missing_output" if missing else "complete")
    assert result["command_exit_code"] == 0
    assert [row["is_file"] for row in result["expected_outputs"]] == ([False, True] if missing else [True])
    assert result["expected_outputs"][-1]["size_bytes"] == 2
    assert (output / "stdout.log").read_text().strip() == "out"
    assert (output / "stderr.log").read_text().strip() == "err"
    initial = json.loads((output / "run.json").read_text())
    assert initial["command_pid"] == result["command_pid"]
    assert initial["status"] == "running"
    events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "start"
    assert events[-1]["status"] == result["status"]


def test_actual_command_exit_code(tmp_path: Path) -> None:
    output = tmp_path / "exit"
    process = start(arguments(output, "raise SystemExit(7)"), tmp_path)
    finish(process)
    result = json.loads((output / "result.json").read_text())
    assert process.returncode == 1
    assert result["status"] == "command_failed"
    assert result["command_exit_code"] == 7


def test_no_deadline_preserves_records_and_output_checks(tmp_path: Path) -> None:
    output = tmp_path / "no_deadline"
    code = "import time; from pathlib import Path; time.sleep(0.25); Path('made.txt').write_text('ok')"
    args = arguments(output, code, seconds=None)
    args[0:0] = ["--expected-output", "made.txt"]
    process = start(args, tmp_path)
    finish(process)
    result = json.loads((output / "result.json").read_text())
    initial = json.loads((output / "run.json").read_text())
    assert process.returncode == 0
    assert result["status"] == "complete"
    assert result["max_seconds"] is None
    assert initial["max_seconds"] is None
    assert result["max_rss_bytes"] == 1024**3
    assert result["rss_samples"] > 0
    assert result["cleanup"]["active_processes_after"] == 0
    assert result["command_exit_code"] == 0
    assert result["expected_outputs"][0]["size_bytes"] == 2


@pytest.mark.parametrize(
    "limits",
    [
        ["--max-rss-gib", "1"],
        ["--max-seconds", "1", "--no-deadline", "--max-rss-gib", "1"],
        ["--no-deadline"],
        ["--no-deadline", "--max-rss-gib", "0"],
        ["--no-deadline", "--max-rss-gib", "nan"],
        ["--no-deadline", "--max-rss-gib", "inf"],
        ["--max-seconds", "0", "--max-rss-gib", "1"],
        ["--max-seconds", "nan", "--max-rss-gib", "1"],
        ["--max-seconds", "inf", "--max-rss-gib", "1"],
    ],
)
def test_invalid_resource_choices_do_not_launch(tmp_path: Path, limits: list[str]) -> None:
    output, marker = tmp_path / "invalid", tmp_path / "launched.txt"
    code = f"from pathlib import Path; Path({str(marker)!r}).touch()"
    args = ["--output-dir", str(output), *limits, "--", CHILD_PYTHON, "-c", code]
    process = start(args, tmp_path)
    finish(process)
    assert process.returncode == 2
    assert not marker.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    ("status", "seconds", "rss", "code"),
    [
        ("deadline", 0.4, 1, "import time; time.sleep(60)"),
        (
            "memory_limit",
            20,
            0.08,
            "import subprocess,sys,time; allocation=bytearray(40*1024**2); "
            "subprocess.Popen([sys.executable,'-c','import time; allocation=bytearray(40*1024**2); time.sleep(60)']); "
            "time.sleep(60)",
        ),
        (
            "memory_limit",
            None,
            0.08,
            "import subprocess,sys,time; allocation=bytearray(40*1024**2); "
            "subprocess.Popen([sys.executable,'-c','import time; allocation=bytearray(40*1024**2); time.sleep(60)']); "
            "time.sleep(60)",
        ),
    ],
)
def test_real_resource_limits(tmp_path: Path, status: str, seconds: float | None, rss: float, code: str) -> None:
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
    assert result["elapsed_seconds"] < 30
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
    code = f"import time; from pathlib import Path; time.sleep(3); Path({str(marker)!r}).write_text('done')"
    args = arguments(output, code)
    args[0:0] = ["--expected-output", str(marker)]
    process = start(args, tmp_path)
    try:
        wait_file(output / "run.json", process)
        wait_file(output / "events.jsonl", process)
        original = (output / "run.json").read_bytes()
        with windows_read_lock(output / "run.json"):
            with windows_read_lock(output / "events.jsonl"):
                locked_at = time.monotonic()
                time.sleep(2.2)
                assert time.monotonic() - locked_at > 1.575
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
        time.sleep(0.025)


@pytest.mark.skipif(os.name != "nt", reason="The ownership proof uses Windows Job Objects")
def test_supervisor_exception_stops_real_family(tmp_path: Path) -> None:
    output, marker = tmp_path / "exception", tmp_path / "family.json"
    inject = (
        f"import sys,json; sys.path.insert(0,{str(RUNNER.parents[1])!r}); from scripts import run_bounded as runner; "
        "from pathlib import Path\n"
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
    process = start(arguments(output, family_code(tmp_path / "ready.json")), tmp_path, [sys.executable, "-c", inject])
    finish(process)
    result = json.loads((output / "result.json").read_text())
    assert process.returncode == 1
    assert result["status"] == "supervisor_error"
    assert result["error"] == "RuntimeError: injected supervisor failure"
    assert result["command_exit_code"] is not None
    assert result["cleanup"]["active_processes_before"] >= 2
    assert result["cleanup"]["active_processes_after"] == 0
    observed = json.loads(marker.read_text())
    assert observed["child"] in observed["members"]
    assert observed["grandchild"] in observed["members"]
    assert_family_stops(marker)


@pytest.mark.skipif(os.name != "nt", reason="Hard supervisor death requires Windows Job Object containment")
def test_hard_supervisor_death_stops_real_family(tmp_path: Path) -> None:
    output, marker = tmp_path / "hard_death", tmp_path / "family.json"
    process = start(arguments(output, family_code(marker)), tmp_path)
    try:
        wait_file(marker, process)
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


def test_preexisting_expected_output_does_not_credit_stale_file(tmp_path: Path) -> None:
    stale, launched = tmp_path / "stale.txt", tmp_path / "launched.txt"
    stale.write_text("old")
    output = tmp_path / "stale"
    args = arguments(output, f"from pathlib import Path; Path({str(launched)!r}).touch()")
    args[0:0] = ["--expected-output", str(stale)]
    process = start(args, tmp_path)
    _, stderr = finish(process)
    assert process.returncode == 1
    assert "expected output already exists" in stderr
    assert stale.read_text() == "old"
    assert not launched.exists()
    assert not output.exists()


@pytest.mark.skipif(os.name != "nt", reason="The process-family proof uses Windows Job Objects")
def test_direct_exit_does_not_credit_live_grandchild(tmp_path: Path) -> None:
    output, marker = tmp_path / "remaining_family", tmp_path / "family.json"
    process = start(arguments(output, family_code(marker, parent_exits=True)), tmp_path)
    finish(process)
    result = json.loads((output / "result.json").read_text())
    assert process.returncode == 1
    assert result["command_exit_code"] == 0
    assert result["status"] == "deadline"
    assert_family_stops(marker)


@pytest.mark.parametrize(("seconds", "status"), [(20, "complete"), (0.1, "deadline")])
def test_fast_exit_before_process_lookup(tmp_path: Path, seconds: float, status: str) -> None:
    output = tmp_path / "fast"
    inject = (
        f"import sys,time; sys.path.insert(0,{str(RUNNER.parents[1])!r}); from scripts import run_bounded as runner\n"
        "original=runner.publish\n"
        "def slow_initial(path,payload):\n"
        "    if path.name == 'run.json':\n"
        "        try: runner.psutil.Process(payload['command_pid']).wait(timeout=10)\n"
        "        except runner.psutil.NoSuchProcess: pass\n"
        "        time.sleep(0.3)\n"
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


@pytest.mark.skipif(os.name != "nt", reason="The regression uses Windows path case rules")
def test_reserved_output_name_is_case_insensitive(tmp_path: Path) -> None:
    output, launched = tmp_path / "reserved", tmp_path / "launched.txt"
    args = arguments(output, f"from pathlib import Path; Path({str(launched)!r}).touch()")
    args[0:0] = ["--expected-output", str(output / "RUN.JSON")]
    process = start(args, tmp_path)
    _, stderr = finish(process)
    assert process.returncode == 1
    assert "expected output cannot be a runner record" in stderr
    assert not launched.exists()
    assert not output.exists()


@pytest.mark.skipif(os.name != "nt", reason="The regression uses real Windows file aliases")
@pytest.mark.parametrize("alias", ["run.json.", "run.json "])
def test_windows_record_alias_is_not_an_output(tmp_path: Path, alias: str) -> None:
    output = tmp_path / "alias"
    expected = output / alias
    args = arguments(output, "pass")
    args[0:0] = ["--expected-output", str(expected)]
    process = start(args, tmp_path)
    finish(process)
    result = json.loads((output / "result.json").read_text())
    assert os.path.samefile(expected, output / "run.json")
    assert process.returncode == 1
    assert result["status"] == "invalid_output"
    assert result["command_exit_code"] == 0
    assert result["expected_outputs"][0]["runner_record_alias"]


def test_external_hard_link_to_record_is_not_an_output(tmp_path: Path) -> None:
    output, expected = tmp_path / "hard_link", tmp_path / "artifact.txt"
    record = output / "run.json"
    code = (
        f"import os,time; from pathlib import Path; source=Path({str(record)!r})\n"
        "while not source.exists(): time.sleep(0.01)\n"
        f"os.link(source,{str(expected)!r})"
    )
    args = arguments(output, code)
    args[0:0] = ["--expected-output", str(expected)]
    process = start(args, tmp_path)
    finish(process)
    result = json.loads((output / "result.json").read_text())
    assert os.path.samefile(expected, record)
    assert process.returncode == 1
    assert result["status"] == "invalid_output"
    assert result["command_exit_code"] == 0
