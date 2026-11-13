"""Run one command with resource limits and durable execution records."""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
import subprocess
import sys
import time
import traceback
from ctypes import wintypes
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import psutil

if sys.platform == "win32":
    from ctypes import WinDLL, WinError, get_last_error
else:
    from os import getpgid, killpg
    from signal import SIGKILL

POLL_SECONDS = 0.1
RECORD_NAMES = (
    "run.json",
    "run.json.tmp",
    "events.jsonl",
    "stdout.log",
    "stderr.log",
    "result.json",
    "result.json.tmp",
)
# MSVC link.exe starts vctip.exe to upload telemetry. It can outlive the build in the
# job and would hold a finished command open until its deadline, or forever.
TELEMETRY_IMAGE = "vctip.exe"
TELEMETRY_INSTALL_DIR = "microsoft visual studio"
TELEMETRY_GRACE_SECONDS = 10.0
# The command's hidden console stays open while vctip.exe is attached to it. It exits
# by itself after its last client, so the runner tolerates it but never stops it.
CONSOLE_HOST = PureWindowsPath(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "conhost.exe")


def utc_now() -> str:
    """Return an explicit UTC timestamp."""
    return datetime.now(UTC).isoformat()


class WindowsJob:
    """Keep descendants in an OS job until the supervisor exits."""

    kernel: Any
    handle: int

    def __init__(self) -> None:
        class BasicLimits(ctypes.Structure):
            _fields_ = [
                ("process_time", ctypes.c_int64),
                ("job_time", ctypes.c_int64),
                ("flags", wintypes.DWORD),
                ("minimum_working_set", ctypes.c_size_t),
                ("maximum_working_set", ctypes.c_size_t),
                ("active_processes", wintypes.DWORD),
                ("affinity", ctypes.c_size_t),
                ("priority", wintypes.DWORD),
                ("scheduling", wintypes.DWORD),
            ]

        class IoCounters(ctypes.Structure):
            _fields_ = [
                (name, ctypes.c_uint64) for name in ("read_ops", "write_ops", "other_ops", "read", "write", "other")
            ]

        class ExtendedLimits(ctypes.Structure):
            _fields_ = [
                ("basic", BasicLimits),
                ("io", IoCounters),
                ("process_memory", ctypes.c_size_t),
                ("job_memory", ctypes.c_size_t),
                ("peak_process_memory", ctypes.c_size_t),
                ("peak_job_memory", ctypes.c_size_t),
            ]

        self.kernel = WinDLL("kernel32", use_last_error=True)
        self.kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
        self.kernel.CreateJobObjectW.restype = wintypes.HANDLE
        self.kernel.GetCurrentProcess.restype = wintypes.HANDLE
        self.kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        self.kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
        self.kernel.QueryInformationJobObject.argtypes = [
            wintypes.HANDLE,
            ctypes.c_int,
            ctypes.c_void_p,
            wintypes.DWORD,
            ctypes.c_void_p,
        ]
        self.kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.kernel.OpenProcess.restype = wintypes.HANDLE
        self.kernel.IsProcessInJob.argtypes = [wintypes.HANDLE, wintypes.HANDLE, ctypes.POINTER(wintypes.BOOL)]
        self.kernel.QueryFullProcessImageNameW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            wintypes.LPWSTR,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise WinError(get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # Kill-on-close prevents orphans after hard supervisor death.
        if not self.kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = WinError(get_last_error())
            self.kernel.CloseHandle(self.handle)
            raise error
        if not self.kernel.AssignProcessToJobObject(self.handle, self.kernel.GetCurrentProcess()):
            error = WinError(get_last_error())
            self.kernel.CloseHandle(self.handle)
            raise error
        # The supervisor owns the only handle, so explicit close would also kill it.

    def pids(self) -> list[int]:
        """Read job members even after the direct command exits."""
        capacity = 64
        while True:
            buffer = ctypes.create_string_buffer(8 + capacity * ctypes.sizeof(ctypes.c_size_t))
            ok = self.kernel.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), None)
            if ok:
                count = ctypes.c_uint32.from_buffer(buffer, 4).value
                array = (ctypes.c_size_t * count).from_buffer(buffer, 8)
                return [int(pid) for pid in array if pid != os.getpid()]
            error = get_last_error()
            if error != 234:
                raise WinError(error)
            capacity *= 2

    def active_descendants(self) -> int:
        """Use the kernel count to exclude completed process objects."""
        buffer = ctypes.create_string_buffer(48)
        if not self.kernel.QueryInformationJobObject(self.handle, 1, buffer, len(buffer), None):
            raise WinError(get_last_error())
        return max(0, ctypes.c_uint32.from_buffer(buffer, 40).value - 1)

    def stop(self) -> None:
        """Terminate only processes that still belong to this job."""
        for pid in self.pids():
            handle = self.kernel.OpenProcess(0x100001 | 0x1000, False, pid)
            if not handle:
                if get_last_error() == 87:
                    continue
                raise WinError(get_last_error())
            try:
                belongs = wintypes.BOOL()
                if not self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(belongs)):
                    raise WinError(get_last_error())
                if belongs.value:
                    if not self.kernel.TerminateProcess(handle, 1) and self.kernel.WaitForSingleObject(handle, 0) != 0:
                        raise WinError(get_last_error())
                    if self.kernel.WaitForSingleObject(handle, 5000) != 0:
                        raise TimeoutError(f"An owned process does not exit: {pid}")
            finally:
                self.kernel.CloseHandle(handle)

    def stop_compiler_telemetry(self) -> list[dict[str, Any]]:
        """Terminate MSVC telemetry uploaders when nothing else remains in the job.

        Returns:
            The PID and image path of each terminated uploader. The list is empty, and no
            member is touched, when a live member is neither an uploader nor the console host.
        """
        handles: list[int] = []
        uploaders: list[tuple[int, int, str]] = []
        try:
            for pid in self.pids():
                handle = self.kernel.OpenProcess(0x100001 | 0x1000, False, pid)
                if not handle:
                    if get_last_error() == 87:
                        continue
                    raise WinError(get_last_error())
                handles.append(handle)
                belongs = wintypes.BOOL()
                if not self.kernel.IsProcessInJob(handle, self.handle, ctypes.byref(belongs)):
                    raise WinError(get_last_error())
                if not belongs.value or self.kernel.WaitForSingleObject(handle, 0) == 0:
                    continue
                # The open handle pins the process, so the path cannot belong to a reused PID.
                size = wintypes.DWORD(32768)
                buffer = ctypes.create_unicode_buffer(size.value)
                if not self.kernel.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
                    raise WinError(get_last_error())
                if is_compiler_telemetry(buffer.value):
                    uploaders.append((pid, handle, buffer.value))
                elif buffer.value.casefold() != str(CONSOLE_HOST).casefold():
                    return []
            stopped = []
            for pid, handle, image in uploaders:
                if not self.kernel.TerminateProcess(handle, 1) and self.kernel.WaitForSingleObject(handle, 0) != 0:
                    raise WinError(get_last_error())
                if self.kernel.WaitForSingleObject(handle, 5000) != 0:
                    raise TimeoutError(f"A telemetry process does not exit: {pid}")
                stopped.append({"pid": pid, "image": image})
            return stopped
        finally:
            for handle in handles:
                self.kernel.CloseHandle(handle)


def is_compiler_telemetry(image: str) -> bool:
    """Match only vctip.exe inside a Visual Studio installation."""
    parts = [part.casefold() for part in PureWindowsPath(image).parts]
    return len(parts) > 1 and parts[-1] == TELEMETRY_IMAGE and TELEMETRY_INSTALL_DIR in parts[:-1]


# Windows closes this private handle at process exit, after the final record flush.
_WINDOWS_JOB: WindowsJob | None = None


class Telemetry:
    """Keep telemetry errors separate from command supervision."""

    def __init__(self, output: Path) -> None:
        self.output = output
        self.errors: list[str] = []

    def error(self, message: str) -> None:
        """Warn once per distinct error without a loss of worker ownership."""
        if message not in self.errors:
            self.errors.append(message)
            try:
                print(f"Warning: telemetry I/O error: {message}", file=sys.stderr, flush=True)
            except OSError:
                pass

    def event(self, event: str, **fields: Any) -> None:
        """Append one event without a replace operation on an open status file."""
        try:
            with (self.output / "events.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"time": utc_now(), "event": event, **fields}, allow_nan=False) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        except OSError as error:
            self.error(f"{type(error).__name__}: {error}")


def publish(path: Path, payload: dict[str, Any]) -> None:
    """Publish one new record without replacement of an existing destination."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    if os.name == "nt":
        temporary.rename(path)
    else:
        os.link(temporary, path)
        temporary.unlink()


def group_processes(group: int) -> list[psutil.Process]:
    """Keep POSIX group members visible after their direct parent exits."""
    members = []
    for pid in psutil.pids():
        try:
            if getpgid(pid) == group:
                process = psutil.Process(pid)
                if process.status() != psutil.STATUS_ZOMBIE:
                    members.append(process)
        except (ProcessLookupError, psutil.NoSuchProcess):
            continue
    return members


def tree_rss(pid: int, job: WindowsJob | None) -> int:
    """Measure current aggregate RSS for the command and its descendants."""
    if job is None:
        members = group_processes(pid)
    else:
        members = []
        for pid in job.pids():
            try:
                members.append(psutil.Process(pid))
            except psutil.NoSuchProcess:
                continue
    total = 0
    for member in members:
        try:
            total += member.memory_info().rss
        except psutil.NoSuchProcess:
            continue
    return total


def stop_owned(child: subprocess.Popen[bytes], job: WindowsJob | None) -> dict[str, int | bool]:
    """Stop the command lifetime and wait for owned processes."""
    if job is None:
        members = group_processes(child.pid)
        try:
            killpg(child.pid, SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=10)
        _, alive = psutil.wait_procs(members, timeout=5)
        if any(process.status() != psutil.STATUS_ZOMBIE for process in alive):
            raise TimeoutError("Owned group processes do not exit within five seconds")
        return {
            "forced": bool(members),
            "active_processes_before": len(members),
            "active_processes_after": len(group_processes(child.pid)),
            "command_waited": True,
        }
    before = job.active_descendants()
    deadline = time.monotonic() + 10
    while job.active_descendants():
        try:
            job.stop()
        except PermissionError:
            if time.monotonic() > deadline:
                raise
            # A process can exit between the job snapshot and OpenProcess.
            time.sleep(POLL_SECONDS)
        if time.monotonic() > deadline:
            raise TimeoutError("Owned descendants do not exit within ten seconds")
    child.wait(timeout=10)
    return {
        "forced": bool(before),
        "active_processes_before": before,
        "active_processes_after": job.active_descendants(),
        "command_waited": True,
    }


def has_owned_descendants(child: subprocess.Popen[bytes], job: WindowsJob | None) -> bool:
    """Reject success while an owned descendant remains alive."""
    if job is not None:
        return bool(job.active_descendants())
    return bool(group_processes(child.pid))


def supervise(args: argparse.Namespace) -> int:
    """Own the command until exit, cleanup, and terminal output verification."""
    global _WINDOWS_JOB
    output = args.output_dir.resolve()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        raise ValueError("A command must follow --")
    for path in args.expected_output:
        if os.path.lexists(path):
            raise FileExistsError(f"An expected output already exists: {path}")
    expected = [path.resolve() for path in args.expected_output]
    for path in expected:
        if os.path.lexists(path):
            raise FileExistsError(f"An expected output already exists: {path}")
        if path.parent == output and os.path.normcase(path.name) in RECORD_NAMES:
            raise ValueError(f"An expected output cannot be a runner record: {path}")
    output.mkdir(parents=True, exist_ok=False)
    telemetry = Telemetry(output)
    child = None
    started = time.monotonic()
    result: dict[str, Any] = {
        "status": "supervisor_error",
        "command": command,
        "cwd": str(Path.cwd()),
        "supervisor_pid": os.getpid(),
        "started_utc": utc_now(),
        "max_seconds": args.max_seconds,
        "max_rss_bytes": int(args.max_rss_gib * 1024**3),
        "peak_tree_rss_bytes": 0,
        "rss_samples": 0,
        "command_exit_code": None,
        "containment": "windows_kill_on_close_job" if os.name == "nt" else "posix_process_group",
        "telemetry_cleanup": [],
    }
    try:
        if os.name == "nt":
            _WINDOWS_JOB = WindowsJob()
        with (output / "stdout.log").open("xb", buffering=0) as stdout:
            with (output / "stderr.log").open("xb", buffering=0) as stderr:
                child = subprocess.Popen(
                    command,
                    stdout=stdout,
                    stderr=stderr,
                    start_new_session=os.name != "nt",
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                result["command_pid"] = child.pid
                try:
                    publish(output / "run.json", {**result, "status": "running"})
                except OSError as error:
                    telemetry.error(f"Initial record: {type(error).__name__}: {error}")
                telemetry.event("start", command_pid=child.pid, supervisor_pid=os.getpid())
                last_event = -1.0
                command_exited: float | None = None
                while True:
                    live = child.poll() is None or has_owned_descendants(child, _WINDOWS_JOB)
                    if live and child.returncode is not None and _WINDOWS_JOB is not None:
                        # Give real descendants time to finish before a telemetry-only check.
                        if command_exited is None:
                            command_exited = time.monotonic()
                        if time.monotonic() - command_exited >= TELEMETRY_GRACE_SECONDS:
                            stopped = _WINDOWS_JOB.stop_compiler_telemetry()
                            if stopped:
                                result["telemetry_cleanup"].extend(stopped)
                                telemetry.event("telemetry_cleanup", processes=stopped)
                                live = has_owned_descendants(child, _WINDOWS_JOB)
                    elapsed = time.monotonic() - started
                    if args.max_seconds is not None and elapsed > args.max_seconds:
                        result["status"] = "deadline"
                        break
                    if not live:
                        result["status"] = "complete" if child.returncode == 0 else "command_failed"
                        break
                    rss = tree_rss(child.pid, _WINDOWS_JOB)
                    result["rss_samples"] += 1
                    result["peak_tree_rss_bytes"] = max(result["peak_tree_rss_bytes"], rss)
                    if elapsed - last_event >= 1:
                        telemetry.event("progress", elapsed_seconds=elapsed, tree_rss_bytes=rss)
                        last_event = elapsed
                    if rss > result["max_rss_bytes"]:
                        result["status"] = "memory_limit"
                        break
                    time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        result.update(status="interrupted", error="KeyboardInterrupt")
    except Exception as error:
        # This boundary must clean up a real worker after any supervisor failure.
        result.update(
            status="supervisor_error", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()
        )
    finally:
        if child is not None:
            try:
                result["cleanup"] = stop_owned(child, _WINDOWS_JOB)
            except (OSError, psutil.Error, subprocess.TimeoutExpired) as error:
                result.update(
                    status="supervisor_error",
                    cleanup_error=f"{type(error).__name__}: {error}",
                    cleanup_traceback=traceback.format_exc(),
                )
            result["command_exit_code"] = child.poll()
        result["elapsed_seconds"] = time.monotonic() - started
        result["finished_utc"] = utc_now()
        for filename in ("stdout.log", "stderr.log"):
            try:
                with (output / filename).open("ab") as stream:
                    stream.flush()
                    os.fsync(stream.fileno())
            except OSError as error:
                telemetry.error(f"Log flush: {type(error).__name__}: {error}")
        checks = []
        for path in expected:
            try:
                present = path.is_file()
                alias = present and any(
                    os.path.samefile(path, output / name) for name in RECORD_NAMES if (output / name).is_file()
                )
                checks.append(
                    {
                        "path": str(path),
                        "is_file": present,
                        "size_bytes": path.stat().st_size if present else None,
                        "runner_record_alias": alias,
                    }
                )
            except OSError as error:
                checks.append({"path": str(path), "is_file": False, "error": f"{type(error).__name__}: {error}"})
        result["expected_outputs"] = checks
        if result["status"] == "complete":
            if any(row.get("runner_record_alias") for row in checks):
                result["status"] = "invalid_output"
            elif not all(row["is_file"] for row in checks):
                result["status"] = "missing_output"
        telemetry.event("finish", status=result["status"], command_exit_code=result["command_exit_code"])
        result["telemetry_errors"] = telemetry.errors
        try:
            publish(output / "result.json", result)
        except OSError as error:
            print(f"The runner cannot publish the terminal record: {error}\n{json.dumps(result)}", file=sys.stderr)
            return 1
    return 0 if result["status"] == "complete" else 1


def main() -> int:
    """Parse one bounded command without a shell."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    deadline = parser.add_mutually_exclusive_group(required=True)
    deadline.add_argument("--max-seconds", type=float)
    deadline.add_argument("--no-deadline", action="store_true", help="Keep the memory limit without a wall deadline")
    parser.add_argument("--max-rss-gib", type=float, required=True)
    parser.add_argument("--expected-output", type=Path, action="append", default=[])
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    limits = [args.max_rss_gib]
    if args.max_seconds is not None:
        limits.append(args.max_seconds)
    if any(not math.isfinite(value) or value <= 0 for value in limits):
        parser.error("Resource limits must be finite and greater than zero")
    if not args.command or args.command[0] != "--" or len(args.command) < 2:
        parser.error("A command must follow --")
    try:
        return supervise(args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
