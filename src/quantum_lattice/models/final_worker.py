"""Single-launch, fail-closed supervisor for the isolated headless worker.

No privileged setting is changed. Hardware facts are read before launch and
throughout execution; the child runs once, in its own process group. Pauses and
ordinary guards request an atomic checkpoint with SIGINT, while severe heat or
a failed supervisor probe escalates promptly. Campaign-wide cumulative runtime
is owned by the research coordinator; this adds a per-launch wall deadline.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time
import uuid


EXPECTED_ONLINE = [0, 1, 3, 4, 5, 6, 7, 9, 10, 11]
FAULT = re.compile(r"hardware error|machine.check|mce:|NVRM:.*Xid|out of memory|oom.kill|segfault|general protection fault", re.I)


def _json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w") as handle:
        handle.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _capture(command):
    value = subprocess.run(command, capture_output=True, text=True, timeout=5)
    if value.returncode:
        raise RuntimeError(f"Hardware probe failed ({command[0]}): {value.stderr.strip()[:240]}")
    return value.stdout


def _cpu_list(value):
    result = set()
    for part in value.strip().split(","):
        if "-" in part:
            a, b = map(int, part.split("-"))
            result.update(range(a, b+1))
        else:
            result.add(int(part))
    return sorted(result)


class HardwareProbe:
    """Read-only Linux/AMD/NVIDIA guards; missing evidence is an error."""
    def kernel(self, cursor=None):
        command = ["journalctl", "-k", "-b", "--no-pager", "-o", "json", "--show-cursor"]
        command += ["--after-cursor", cursor] if cursor else ["-n", "1"]
        output = _capture(command)
        messages, next_cursor = [], cursor
        for line in output.splitlines():
            if line.startswith("-- cursor: "):
                next_cursor = line.split(": ", 1)[1].strip()
            elif line.startswith("{"):
                record = json.loads(line)
                message = record.get("MESSAGE", "")
                messages.append(str(message))
                next_cursor = record.get("__CURSOR", next_cursor)
        if not next_cursor:
            raise RuntimeError("Kernel journal frontier is unavailable; refusing an unmonitored launch")
        return next_cursor, messages if cursor else []

    def snapshot(self, cursor=None):
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        online = _cpu_list(Path("/sys/devices/system/cpu/online").read_text())
        boost = int(Path("/sys/devices/system/cpu/cpufreq/boost").read_text().strip())
        cpu_temperatures = []
        for hwmon in Path("/sys/class/hwmon").glob("hwmon*"):
            if (hwmon/"name").read_text().strip() == "k10temp":
                for source in hwmon.glob("temp*_input"):
                    cpu_temperatures.append(int(source.read_text().strip())/1000)
        if not cpu_temperatures:
            raise RuntimeError("AMD CPU temperature guard is unavailable")
        gpu = _capture(["nvidia-smi", "--query-gpu=uuid,temperature.gpu", "--format=csv,noheader,nounits"])
        rows = [line.strip().split(",") for line in gpu.splitlines() if line.strip()]
        if len(rows) != 1 or len(rows[0]) != 2:
            raise RuntimeError("Expected exactly one identifiable NVIDIA GPU")
        gpu_uuid, gpu_temperature = rows[0][0].strip(), float(rows[0][1])
        apps = _capture(["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader,nounits"])
        processes = []
        for line in apps.splitlines():
            if not line.strip() or "No running processes" in line:
                continue
            pid, app_gpu = line.split(",")
            processes.append({"pid": int(pid), "gpu_uuid": app_gpu.strip()})
        memory = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            memory[key] = int(value.split()[0])
        next_cursor, messages = self.kernel(cursor)
        return {"boot_id": boot, "online": online, "boost": boost,
                "affinity": sorted(os.sched_getaffinity(0)),
                "cpu_c": max(cpu_temperatures), "gpu_c": gpu_temperature,
                "gpu_uuid": gpu_uuid, "compute_processes": processes,
                "swap_used_kib": memory["SwapTotal"]-memory["SwapFree"],
                "kernel_cursor": next_cursor, "kernel_faults": [m[:1000] for m in messages if FAULT.search(m)]}


def _settings(hardware):
    defaults = {"expected_online": EXPECTED_ONLINE, "expected_boost": 0,
                "max_cpu_c": 80, "max_gpu_c": 82, "severe_cpu_c": 90,
                "severe_gpu_c": 90, "poll_seconds": 10, "drain_seconds": 300,
                "require_zero_swap": True}
    if not isinstance(hardware, dict) or set(hardware)-set(defaults)-{"gpu_uuid", "expected_boot_id"}:
        raise ValueError("Unknown worker hardware settings")
    config = {**defaults, **hardware}
    if not isinstance(config.get("gpu_uuid"), str) or not config["gpu_uuid"].startswith("GPU-"):
        raise ValueError("The actual worker GPU UUID must be pinned")
    if config["expected_online"] != EXPECTED_ONLINE or config["expected_boost"] != 0:
        raise ValueError("This worker requires the verified ten-CPU/boost-off workaround")
    for name in ("max_cpu_c", "max_gpu_c", "severe_cpu_c", "severe_gpu_c", "poll_seconds", "drain_seconds"):
        value = config[name]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not (config["max_cpu_c"] <= 80 < config["severe_cpu_c"] <= 95
            and config["max_gpu_c"] <= 82 < config["severe_gpu_c"] <= 95
            and config["poll_seconds"] <= 10 and config["drain_seconds"] <= 300):
        raise ValueError("Worker guard limits cannot be weakened")
    if type(config["require_zero_swap"]) is not bool or not config["require_zero_swap"]:
        raise ValueError("The qualified worker requires the zero-swap guard")
    return config


def _guard(snapshot, settings, boot_id, child_pid=None):
    """Return the first deterministic guard reason and escalation severity."""
    required = {"boot_id", "online", "boost", "affinity", "cpu_c", "gpu_c", "gpu_uuid",
                "compute_processes", "swap_used_kib", "kernel_cursor", "kernel_faults"}
    if not required.issubset(snapshot):
        return "missing_guard_evidence", True
    if (not isinstance(snapshot["kernel_cursor"], str) or not snapshot["kernel_cursor"]
            or not isinstance(snapshot["boot_id"], str) or not snapshot["boot_id"]
            or type(snapshot["swap_used_kib"]) is not int or snapshot["swap_used_kib"] < 0
            or not isinstance(snapshot["compute_processes"], list)
            or not isinstance(snapshot["kernel_faults"], list)):
        return "invalid_guard_evidence", True
    for key in ("cpu_c", "gpu_c"):
        if type(snapshot[key]) not in (float, int) or not math.isfinite(snapshot[key]):
            return "invalid_thermal_guard", True
    if snapshot["cpu_c"] >= settings["severe_cpu_c"] or snapshot["gpu_c"] >= settings["severe_gpu_c"]:
        return "severe_heat", True
    checks = [(snapshot["boot_id"] != boot_id, "boot_changed"),
              (snapshot["online"] != settings["expected_online"], "online_cpu_changed"),
              (snapshot["boost"] != settings["expected_boost"], "boost_changed"),
              (not snapshot["affinity"] or not set(snapshot["affinity"]).issubset(settings["expected_online"]), "unsafe_affinity"),
              (snapshot["gpu_uuid"] != settings["gpu_uuid"], "gpu_changed"),
              (snapshot["cpu_c"] >= settings["max_cpu_c"], "cpu_temperature"),
              (snapshot["gpu_c"] >= settings["max_gpu_c"], "gpu_temperature"),
              (snapshot["swap_used_kib"] != 0, "swap_in_use"),
              (bool(snapshot["kernel_faults"]), "new_kernel_fault")]
    for failed, reason in checks:
        if failed:
            return reason, False
    for process in snapshot["compute_processes"]:
        if process["gpu_uuid"] != settings["gpu_uuid"]:
            return "unexpected_compute_gpu", False
        if child_pid is None:
            return "gpu_already_busy", False
        try:
            if os.getpgid(process["pid"]) != child_pid:
                return "foreign_gpu_work", False
        except ProcessLookupError:
            continue
    return None, False


def _signal(child, signum):
    try:
        os.killpg(child.pid, signum)
    except ProcessLookupError:
        pass


def _supervise(command, repo_root, launch, max_wall_seconds, settings, stop_file,
               *, probe, clock=time, popen=subprocess.Popen, send_signal=_signal,
               stop_requested=None):
    """Injection seam for CPU-only fixtures; not exposed as a production CLI."""
    started = clock.monotonic()
    hard_deadline = started+max_wall_seconds
    deadline_request = hard_deadline-settings["drain_seconds"]-20
    state = {"state": "preflight", "command": command, "started_utc": _utc(),
             "max_wall_seconds": max_wall_seconds, "first_guard": None,
             "stop_reason": None, "child_pid": None, "launch_count": 0,
             "restart_policy": "never", "monitor_samples": 0}
    def event(kind, **details):
        record = {"event": kind, "utc": _utc(),
                  "active_wall_seconds": clock.monotonic()-started, **details}
        with (launch/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False)+"\n")
            handle.flush()
            os.fsync(handle.fileno())
    def save():
        state["active_wall_seconds"] = clock.monotonic()-started
        _json(launch/"state.json", state)
    save()
    if stop_file.exists() or (stop_requested is not None and stop_requested()):
        state.update(state="paused", stop_reason="user_pause")
        save()
        return state
    try:
        snapshot = probe.snapshot()
        boot = settings.get("expected_boot_id", snapshot["boot_id"])
        reason, severe = _guard(snapshot, settings, boot)
        if reason:
            raise RuntimeError(f"Worker preflight guard: {reason}")
        if clock.monotonic() >= deadline_request:
            raise RuntimeError("Insufficient launch budget after reserving checkpoint drain")
    except Exception as error:
        state.update(state="refused", stop_reason="preflight_failed", error=str(error))
        save()
        return state
    state["preflight"] = snapshot
    event("preflight_passed", snapshot=snapshot)
    cursor = snapshot["kernel_cursor"]
    if stop_file.exists() or (stop_requested is not None and stop_requested()):
        state.update(state="paused", stop_reason="user_pause")
        save()
        return state
    env = dict(os.environ)
    env.update({name: "1" for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS")})
    env["PYTHONUNBUFFERED"] = "1"
    env["QL_FINAL_SUPERVISOR_PID"] = str(os.getpid())
    with (launch/"stdout.log").open("wb") as stdout, (launch/"stderr.log").open("wb") as stderr:
        try:
            child = popen(command, cwd=repo_root, env=env, stdout=stdout, stderr=stderr, start_new_session=True)
        except Exception as error:
            state.update(state="failed_launch", stop_reason="child_launch_failed", error=str(error))
            save()
            return state
        state.update(state="running", child_pid=child.pid, boot_id=boot,
                     gpu_uuid=settings["gpu_uuid"], launch_count=1)
        stop_deadline = None
        escalated = False
        next_probe = clock.monotonic()
        def request(reason, severe=False, is_guard=False):
            nonlocal stop_deadline, escalated
            now = clock.monotonic()
            if is_guard and state["first_guard"] is None:
                state["first_guard"] = {"reason": reason, "utc": _utc(), "severe": severe}
                event("first_guard", **state["first_guard"])
            if state["stop_reason"] is None:
                state.update(stop_reason=reason, state="draining")
                stop_deadline = min(hard_deadline, now+settings["drain_seconds"])
                send_signal(child, signal.SIGINT)
                event("checkpoint_stop_requested", reason=reason, child_pid=child.pid)
            if severe and not escalated:
                send_signal(child, signal.SIGTERM)
                escalated = True
                stop_deadline = min(hard_deadline, now+3)
                event("termination_escalated", reason=reason, child_pid=child.pid)
            save()
        try:
            save()
            event("child_started", child_pid=child.pid)
            while child.poll() is None:
                try:
                    now = clock.monotonic()
                    if stop_file.exists() or (stop_requested is not None and stop_requested()):
                        request("user_pause")
                    if now >= deadline_request:
                        request("wall_budget")
                    if now >= next_probe:
                        snapshot = probe.snapshot(cursor)
                        cursor = snapshot["kernel_cursor"]
                        state.update(last_sample=snapshot, monitor_samples=state["monitor_samples"]+1)
                        reason, severe = _guard(snapshot, settings, boot, child.pid)
                        if reason:
                            request(reason, severe, is_guard=True)
                        event("hardware_sample", snapshot=snapshot)
                        next_probe = now+settings["poll_seconds"]
                        save()
                    if stop_deadline is not None and clock.monotonic() >= stop_deadline:
                        send_signal(child, signal.SIGKILL)
                        state["forced_kill"] = True
                        break
                    clock.sleep(min(.5, max(.01, hard_deadline-clock.monotonic())))
                except KeyboardInterrupt:
                    request("user_pause")
                except Exception as error:
                    state["monitor_error"] = str(error)
                    request("supervisor_probe_failed", severe=True, is_guard=True)
                    # Probing a broken monitor repeatedly would spin instead
                    # of allowing the bounded terminate deadline to progress.
                    next_probe = math.inf
            try:
                code = child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                send_signal(child, signal.SIGKILL)
                code = child.wait(timeout=5)
                state["forced_kill"] = True
        except BaseException as error:
            # Includes inability to record a guard (e.g. a full disk). Never
            # leave an unmonitored child alive after controller failure.
            send_signal(child, signal.SIGTERM)
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                send_signal(child, signal.SIGKILL)
                child.wait(timeout=5)
            state.update(state="failed_supervisor", stop_reason="supervisor_failure", error=str(error))
            try:
                save()
            except Exception:
                pass
            raise
        if state["first_guard"] is not None:
            terminal = "failed_guard"
        elif state.get("forced_kill"):
            terminal = "failed_drain"
        elif state["stop_reason"] in {"user_pause", "wall_budget"}:
            terminal = "paused"
        else:
            terminal = "complete" if code == 0 else "failed_child"
        state.update(state=terminal, returncode=code, finished_utc=_utc())
        event("finished", state=terminal, returncode=code)
        save()
        return state


def supervise(command, repo_root, run_dir, max_wall_seconds, hardware, *, stop_file=None):
    """Run one foreground child under the verified PC guards; never retry it.

    Run this supervisor in a persistent session/user service. It does not set
    up persistence, reboot autostart, hardware settings or a second queue.
    ``stop_file`` defaults to ``run_dir/PAUSE``. Delete that request explicitly
    before a later user-authorized continuation.
    """
    if not isinstance(command, list) or not command or any(not isinstance(s, str) or not s for s in command):
        raise ValueError("command must be a nonempty argv list")
    config = _settings(hardware)
    if (type(max_wall_seconds) not in (float, int) or not math.isfinite(max_wall_seconds)
            or not config["drain_seconds"]+21 < max_wall_seconds <= 4*24*3600):
        raise ValueError("Per-launch wall ceiling must reserve drain and remain within four days")
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("The persistent worker supervisor must run in the main thread")
    root = Path(repo_root).resolve()
    allowed = (root/"codex material/artifacts/final_benchmark").resolve()
    run = (root/run_dir).resolve() if not Path(run_dir).is_absolute() else Path(run_dir).resolve()
    if run == allowed or not run.is_relative_to(allowed):
        raise ValueError("Supervisor records must be below ignored final_benchmark artifacts")
    stop_file = run/"PAUSE" if stop_file is None else Path(stop_file).resolve()
    if not stop_file.is_relative_to(allowed):
        raise ValueError("Pause request must be inside final_benchmark artifacts")
    run.mkdir(parents=True, exist_ok=True)
    with (run/".supervisor.lock").open("a") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another final worker supervisor is active") from error
        launch = run/"launches"/uuid.uuid4().hex
        launch.mkdir(parents=True)
        _json(launch/"request.json", {"command": command, "hardware": config,
            "max_wall_seconds": max_wall_seconds, "stop_file": str(stop_file), "root": str(root)})
        old_affinity = os.sched_getaffinity(0)
        stopping = {"requested": False}
        old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def on_stop(signum, frame):
            stopping["requested"] = True
        try:
            for sig in old_handlers:
                signal.signal(sig, on_stop)
            os.sched_setaffinity(0, config["expected_online"])
            result = _supervise(command, root, launch, max_wall_seconds, config, stop_file,
                                probe=HardwareProbe(), stop_requested=lambda: stopping["requested"])
            _json(run/"supervisor-state.json", {**result, "launch_directory": str(launch.relative_to(root))})
            return result
        finally:
            try:
                os.sched_setaffinity(0, old_affinity)
            finally:
                for sig, old in old_handlers.items():
                    signal.signal(sig, old)
