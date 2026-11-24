"""Persistent NVIDIA GPU scheduler for trusted local users."""
import argparse
from contextlib import ExitStack, contextmanager
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from uuid import NAMESPACE_URL, uuid4, uuid5

BASE = Path(os.environ.get("XPFLOW_HOME", Path.home() / ".local/share/xpflow")).expanduser().resolve()
DEFAULTS = {
    "admit_util": 30, "admit_memory": 60, "admit_seconds": 10, "poll_seconds": 3,
    "stop_util": 78, "stop_memory": 80, "guard_seconds": 2, "observe_seconds": 60,
    "max_retries": 3, "max_evaluations": 8, "eval_gpus": [], "eval_fallback": True,
    "pause_evaluations": False, "min_free_bytes": 0, "checkpoint_watches": [],
}
ACTIVE = {"observing", "running", "postprocessing"}
TERMINAL = {"complete", "failed", "cancelled", "blocked"}


def put(path, value):
    temp = path.with_name(f".{path.name}.{uuid4().hex}.pending")
    fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as out:
            out.write(value + "\n")
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    put(path, json.dumps(value, ensure_ascii=False, allow_nan=False))


def settings(q):
    path = q / "config.json"
    raw = load(path) if path.exists() else {}
    if not isinstance(raw, dict) or raw.keys() - DEFAULTS.keys():
        raise ValueError("config.json must contain only documented settings")
    cfg = {**DEFAULTS, **raw}
    for key in ("admit_util", "admit_memory", "stop_util", "stop_memory"):
        if type(cfg[key]) is not int or not 1 <= cfg[key] <= 100:
            raise ValueError(f"{key} must be 1–100")
    if cfg["admit_util"] > cfg["stop_util"] or cfg["admit_memory"] > cfg["stop_memory"]:
        raise ValueError("Admission thresholds must not exceed runtime thresholds")
    for key in ("poll_seconds", "guard_seconds", "max_evaluations"):
        if type(cfg[key]) is not int or cfg[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("admit_seconds", "observe_seconds", "max_retries", "min_free_bytes"):
        if type(cfg[key]) is not int or cfg[key] < 0:
            raise ValueError(f"{key} must be a nonnegative integer")
    for key in ("eval_fallback", "pause_evaluations"):
        if type(cfg[key]) is not bool:
            raise ValueError(f"{key} must be a boolean")
    if not isinstance(cfg["eval_gpus"], list) or any(type(g) is not int or g < 0 for g in cfg["eval_gpus"]):
        raise ValueError("eval_gpus must be a list of nonnegative GPU indices")
    if not isinstance(cfg["checkpoint_watches"], list):
        raise ValueError("checkpoint_watches must be a list")
    for watch in cfg["checkpoint_watches"]:
        validate_watch(watch)
    return cfg


def validate_watch(watch):
    required = {"name", "cwd", "glob", "argv"}
    optional = {"after_gpu", "series", "retain_series_best", "quality_gate"}
    if not isinstance(watch, dict) or not required <= watch.keys() or watch.keys() - required - optional:
        raise ValueError("Each watch needs name, cwd, glob and argv, with optional after_gpu")
    if any(not isinstance(watch[k], str) or not watch[k] for k in ("name", "cwd", "glob")):
        raise ValueError("Watch name, cwd and glob must be nonempty strings")
    if not Path(watch["cwd"]).is_absolute() or Path(watch["glob"]).is_absolute() or ".." in Path(watch["glob"]).parts:
        raise ValueError("Watch cwd must be absolute and glob confined to cwd")
    validate_argv(watch["argv"])
    if not any("{checkpoint}" in arg for arg in watch["argv"]):
        raise ValueError("Watch argv must reference {checkpoint}")
    if "after_gpu" in watch:
        validate_argv(watch["after_gpu"])
    if "series" in watch and (not isinstance(watch["series"], str) or not watch["series"]):
        raise ValueError("Watch series must be a nonempty string")
    if "retain_series_best" in watch and type(watch["retain_series_best"]) is not bool:
        raise ValueError("Watch retain_series_best must be a boolean")
    if watch.get("retain_series_best") and not watch.get("series"):
        raise ValueError("Watch retention requires a series")
    if "quality_gate" in watch:
        validate_gate(watch["quality_gate"])


def validate_argv(value):
    if not isinstance(value, list) or not value or not value[0] or any(not isinstance(x, str) for x in value):
        raise ValueError("A command must be a nonempty JSON array of strings")
    return value


def queue_root():
    BASE.mkdir(parents=True, exist_ok=True, mode=0o700)
    return BASE


def requests(q):
    return list(q.glob("xpflow/*/xpflow.json"))


def enqueue(attempt, name, kind, args, cwd, **options):
    request = dict(name=name, kind=kind, args=list(args), cwd=str(cwd),
                   submitted_ns=time.time_ns(), **options)
    save(attempt / "xpflow.json", request)


@contextmanager
def exclusive(path):
    import fcntl
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield None
        else:
            yield stream


def gpu_lock_path(q, gpu):
    return q / f"gpu-{gpu}.lock"


def queue_order(kind, submitted_at):
    return kind != "train", submitted_at


def request_order(request):
    submitted = request["submitted_ns"]
    if request["kind"] == "eval":
        checkpoint = request.get("checkpoint")
        if checkpoint:
            try:
                return True, False, -Path(checkpoint).stat().st_mtime_ns, submitted
            except OSError:
                pass
        return True, True, submitted, submitted
    return False, bool(request.get("retry_of")), submitted, submitted


def gpu_samples(gpu=None):
    argv = ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits"]
    if gpu is not None:
        argv.extend(["-i", str(gpu)])
    raw = subprocess.check_output(argv, text=True, timeout=5)
    return [tuple(map(int, line.split(","))) for line in raw.splitlines()]


def idle(cfg=DEFAULTS):
    return {gpu for gpu, util, used, total in gpu_samples()
            if 0 <= util < cfg["admit_util"] and used >= 0 and total > 0
            and used * 100 < cfg["admit_memory"] * total}


def state(attempt, cfg=DEFAULTS):
    if (attempt / "exit").is_file():
        if (attempt / "terminal").is_file():
            return (attempt / "terminal").read_text().strip()
        return "complete" if int((attempt / "exit").read_text()) == 0 else "failed"
    if (attempt / "gpu.released").is_file():
        return "postprocessing"
    started = attempt / "xpflow.started"
    if not started.is_file():
        return "queued"
    return "observing" if time.time() - float(started.read_text()) < cfg["observe_seconds"] else "running"


def finish(attempt, code, reason=None, terminal=None):
    if reason:
        put(attempt / "failure", reason)
    if terminal:
        put(attempt / "terminal", terminal)
    put(attempt / "exit", str(code if code >= 0 else 128 - code))


def process_identity(pid):
    try:
        # /proc/[pid]/stat fields after the final ")" start with process state.
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] == "Z":
            return None
        boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        return {"pid": pid, "start": fields[19], "boot": boot}
    except (OSError, ValueError, IndexError):
        return None


def verified_process(identity):
    return bool(identity and process_identity(identity["pid"]) == identity)


def terminate_group(pid, identity):
    if not verified_process(identity):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    time.sleep(3)
    current = process_identity(pid)
    if current is not None and current != identity:
        return
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def stop_started_process(process, identity):
    """Stop and reap a child started by this worker."""
    if identity:
        terminate_group(process.pid, identity)
    if process.poll() is None:
        # This Popen owns the unreaped child even if procfs is unavailable.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    process.wait()


def run_phase(argv, cwd, env, log, locks, attempt, cfg, gpu):
    if (attempt / "cancel").exists():
        return 130, "cancelled_by_user"
    process = subprocess.Popen([arg.replace("{attempt}", str(attempt)) for arg in argv], cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                               start_new_session=True, pass_fds=tuple(lock.fileno() for lock in locks))
    identity = None
    try:
        identity = process_identity(process.pid)
        if identity:
            save(attempt / "process.json", identity)
        while process.poll() is None:
            if (attempt / "cancel").exists():
                stop_started_process(process, identity)
                return 130, "cancelled_by_user"
            if gpu is not None:
                try:
                    _, util, used, total = gpu_samples(gpu)[0]
                    if not 0 <= util <= 100 or not 0 <= used <= total or total <= 0:
                        raise ValueError("Invalid GPU sample")
                    reason = "gpu_red_line" if (util >= cfg["stop_util"]
                                                or used * 100 >= total * cfg["stop_memory"]) else None
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    reason = "gpu_query_failed"
                if reason:
                    if process.poll() is not None:
                        break
                    stop_started_process(process, identity)
                    return 3, reason
            try:
                process.wait(timeout=cfg["guard_seconds"])
            except subprocess.TimeoutExpired:
                pass
        return process.returncode, None
    except BaseException:
        stop_started_process(process, identity)
        raise
    finally:
        if process.poll() is not None:
            (attempt / "process.json").unlink(missing_ok=True)


def run_command(request, gpu, log, lock, gpu_lock, release_gpu, attempt, cfg):
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu),
               XPFLOW_GPU=str(gpu), XPFLOW_ATTEMPT=str(attempt), XPFLOW_PHASE="gpu")
    code, reason = run_phase(request["args"], request["cwd"], env, log, (lock, gpu_lock), attempt, cfg, gpu)
    if code == 0 and request.get("after_gpu"):
        release_gpu()
        put(attempt / "gpu.released", str(time.time()))
        env.update(CUDA_VISIBLE_DEVICES="", XPFLOW_PHASE="cpu")
        return run_phase(request["after_gpu"], request["cwd"], env, log, (lock,), attempt, cfg, None)
    return code, reason


def worker(attempt, gpu):
    q = attempt.parent.parent
    cfg = settings(q)
    with exclusive(attempt / "running.lock") as lock, ExitStack() as gpu_locks:
        if lock is None or state(attempt, cfg) != "queued" or (attempt / "cancel").exists():
            return
        request = load(attempt / "xpflow.json")
        gpu_lock = gpu_locks.enter_context(exclusive(gpu_lock_path(q, gpu)))
        if gpu_lock is None:
            return
        try:
            if gpu not in idle(cfg):
                return
        except (OSError, subprocess.SubprocessError, ValueError):
            return
        put(attempt / "gpu", str(gpu))
        put(attempt / "pid", str(os.getpid()))
        save(attempt / "worker.json", process_identity(os.getpid()))
        put(attempt / "xpflow.started", str(time.time()))
        with (attempt / f"{request['kind']}.log").open("ab", buffering=0) as log:
            try:
                checkpoint = request.get("checkpoint_sha256")
                if checkpoint and not checkpoint_matches(request):
                    finish(attempt, 1, "checkpoint_changed", "blocked")
                    return
                code, reason = run_command(request, gpu, log, lock, gpu_lock, gpu_locks.close, attempt, cfg)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                log.write(f"{error}\n".encode())
                code, reason = 1, "launch_error"
        terminal = "cancelled" if reason == "cancelled_by_user" else None
        if code == 0 and request.get("quality_gate"):
            if not passes_gate(read_report(attempt), request["quality_gate"]):
                code, reason, terminal = 1, "quality_gate_failed", "blocked"
        finish(attempt, code, reason or (f"exit_{code}" if code else None), terminal)


def read_report(attempt):
    try:
        report = load(attempt / "report.json")
        return report if isinstance(report, dict) else {}
    except (OSError, ValueError):
        return {}


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_gate(gate):
    if not isinstance(gate, dict) or not gate:
        raise ValueError("A quality gate must be a nonempty JSON object")
    for bounds in gate.values():
        if (not isinstance(bounds, dict) or not bounds or bounds.keys() - {"min", "max"}
                or not all(finite(x) for x in bounds.values())
                or bounds.get("min", -math.inf) > bounds.get("max", math.inf)):
            raise ValueError("Quality gate bounds require finite min and/or max")
    return gate


def passes_gate(report, gate):
    for metric, bounds in gate.items():
        value = report.get(metric)
        if not finite(value) or not bounds.get("min", -math.inf) <= value <= bounds.get("max", math.inf):
            return False
    return True


def ranked(q):
    rows = []
    for file in requests(q):
        attempt = file.parent
        status = state(attempt)
        if status != "complete" and not (status == "blocked" and (attempt / "failure").is_file()
                                         and (attempt / "failure").read_text().strip() == "quality_gate_failed"):
            continue
        request = load(file)
        if request["kind"] != "eval" or not request.get("series"):
            continue
        report = read_report(attempt)
        if all(finite(report.get(k)) and report[k] >= 0 for k in ("strict_success", "mean_error")):
            rows.append(dict(attempt=str(attempt), series=request["series"], name=request["name"],
                             strict_success=report["strict_success"], mean_error=report["mean_error"],
                             metrics={key: value for key, value in report.items() if finite(value)},
                             retain_series_best=request.get("retain_series_best", False), state=status))
    return sorted(rows, key=lambda row: (-row["strict_success"], row["mean_error"], row["attempt"]))


def retain_artifacts(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["series"], []).append(row)
    for group in groups.values():
        lowest = min(group, key=lambda row: (row["mean_error"], -row["strict_success"], row["attempt"]))
        for row in group:
            if row is lowest or row is group[0] or not row["retain_series_best"]:
                continue
            attempt = Path(row["attempt"])
            raw = attempt / "artifacts"
            if not raw.is_dir() or raw.is_symlink() or raw.resolve() != raw:
                continue
            with exclusive(attempt / "retention.lock") as lock:
                if lock is None:
                    continue
                if not (attempt / "rebuild.json").exists():
                    save(attempt / "rebuild.json", {"request": load(attempt / "xpflow.json"),
                         "report": read_report(attempt),
                         "entries": sorted(str(p.relative_to(raw)) for p in raw.rglob("*"))})
                shutil.rmtree(raw)


def retry_attempt(attempt, request, *, manual=False):
    path = attempt.with_name(uuid5(NAMESPACE_URL, f"xpflow:{attempt.name}:retry").hex)
    path.mkdir(exist_ok=True, mode=0o700)
    with exclusive(path / "enqueue.lock") as lock:
        if lock is None:
            return path
        if not (path / "xpflow.json").exists():
            new = {**request, "retry_of": str(attempt),
                   "retry_count": 0 if manual else request.get("retry_count", 0) + 1,
                   "submitted_ns": time.time_ns()}
            save(path / "xpflow.json", new)
    return path


def dependency_state(request):
    for parent in map(Path, request.get("depends_on", [])):
        status = state(parent)
        if status in TERMINAL - {"complete"}:
            return "blocked"
        if status != "complete":
            return "waiting"
    return "ready"


def reconcile(attempt):
    worker_file = attempt / "worker.json"
    if worker_file.is_file() and not verified_process(load(worker_file)):
        child_file = attempt / "process.json"
        if child_file.is_file():
            child = load(child_file)
            if verified_process(child):
                terminate_group(child["pid"], child)
    with exclusive(attempt / "running.lock") as lock:
        if lock is not None and state(attempt) in ACTIVE:
            cancelled = (attempt / "cancel").exists()
            finish(attempt, 130 if cancelled else 1, "cancelled_by_user" if cancelled else "worker_lost",
                   "cancelled" if cancelled else None)


def dispatch(q, free, children, env, can_start, on_start, cfg=DEFAULTS):
    evaluation_count = 0
    queued = []
    for file in requests(q):
        attempt = file.parent
        request = load(file)
        status = state(attempt, cfg)
        if status in ACTIVE:
            reconcile(attempt)
            status = state(attempt, cfg)
        if status in ACTIVE or (status == "queued" and attempt in children):
            if status != "postprocessing":
                gpu = attempt / "gpu"
                if gpu.is_file():
                    free.discard(int(gpu.read_text()))
                elif attempt in children:
                    free.discard(children[attempt][1])
            evaluation_count += request["kind"] == "eval"
            continue
        if status == "failed" and request.get("retry_count", 0) < request.get("max_retries", cfg["max_retries"]):
            retry_attempt(attempt, request)
        if status != "queued":
            continue
        with exclusive(attempt / "running.lock") as lock:
            if lock is None:
                gpu = attempt / "gpu"
                if gpu.is_file():
                    free.discard(int(gpu.read_text()))
                else:
                    free.clear()
                evaluation_count += request["kind"] == "eval"
                continue
            if (attempt / "cancel").exists():
                finish(attempt, 130, "cancelled_by_user", "cancelled")
                continue
            dependency = dependency_state(request)
            if dependency == "blocked":
                finish(attempt, 1, "dependency_failed", "blocked")
            elif dependency == "ready":
                queued.append((attempt, request))
    for gpu in tuple(free):
        with exclusive(gpu_lock_path(q, gpu)) as lock:
            if lock is None:
                free.discard(gpu)
    eval_gpus = set(cfg["eval_gpus"])
    for attempt, request in sorted(queued, key=lambda pair: (*request_order(pair[1]), pair[0])):
        is_eval = request["kind"] == "eval"
        if is_eval and evaluation_count >= cfg["max_evaluations"]:
            continue
        if not can_start(attempt, request):
            continue
        if is_eval:
            eligible = free & eval_gpus
            if not eligible and (cfg["eval_fallback"] or not eval_gpus):
                eligible = free
        else:
            eligible = free - eval_gpus
        if not eligible:
            continue
        gpu = min(eligible)
        free.remove(gpu)
        child = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_worker", str(attempt), str(gpu)],
                                 env=env, start_new_session=True)
        children[attempt] = (child, gpu)
        evaluation_count += is_eval
        on_start(request, gpu)


@lru_cache(maxsize=4096)
def checkpoint_digest(path, inode, size, modified_ns):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else hash_stream(stream)


def checkpoint_matches(request):
    path = Path(request["checkpoint"])
    try:
        before = path.stat()
        digest = checkpoint_digest(path, before.st_ino, before.st_size, before.st_mtime_ns)
        after = path.stat()
        return digest == request["checkpoint_sha256"] and (
            before.st_ino, before.st_size, before.st_mtime_ns
        ) == (after.st_ino, after.st_size, after.st_mtime_ns)
    except OSError:
        return False


def hash_stream(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def discover_checkpoints(q, watches):
    for watch in watches:
        cwd = Path(watch["cwd"])
        for candidate in cwd.glob(watch["glob"]):
            path = candidate.resolve()
            if not path.is_file() or not path.is_relative_to(cwd.resolve()):
                continue
            before = path.stat()
            if before.st_size == 0 or time.time_ns() - before.st_mtime_ns < 10_000_000_000:
                continue
            digest = checkpoint_digest(path, before.st_ino, before.st_size, before.st_mtime_ns)
            after = path.stat()
            if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
                continue
            key = json.dumps([watch, str(cwd), digest], sort_keys=True)
            attempt = q / "xpflow" / uuid5(NAMESPACE_URL, key).hex
            if (attempt / "xpflow.json").exists():
                continue
            attempt.mkdir(parents=True, exist_ok=True, mode=0o700)
            def expand(argv):
                return [arg.replace("{checkpoint}", str(path)) for arg in argv]
            options = {"checkpoint": str(path), "checkpoint_sha256": digest}
            options.update({k: watch[k] for k in ("series", "retain_series_best", "quality_gate") if k in watch})
            if watch.get("after_gpu"):
                options["after_gpu"] = expand(watch["after_gpu"])
            enqueue(attempt, watch["name"], "eval", expand(watch["argv"]), cwd, **options)


def serve(q):
    children = {}
    eligible_since = {}
    period = 3
    retention = None
    retention_at = 0
    with exclusive(q / "scheduler.lock") as lock:
        if lock is None:
            raise SystemExit("Another scheduler is already running for this queue")
        while True:
            try:
                cfg = settings(q)
                period = cfg["poll_seconds"]
                try:
                    current = idle(cfg)
                except (OSError, subprocess.SubprocessError, ValueError):
                    current = set()
                now = time.monotonic()
                eligible_since = {gpu: eligible_since.get(gpu, now) for gpu in current}
                free = {gpu for gpu, start in eligible_since.items() if now - start >= cfg["admit_seconds"]}
                children = {a: worker for a, worker in children.items() if worker[0].poll() is None}
                discover_checkpoints(q, cfg["checkpoint_watches"])
                storage_ready = shutil.disk_usage(q).free >= cfg["min_free_bytes"]
                def can_start(attempt, request):
                    return request["kind"] != "eval" or (storage_ready and not cfg["pause_evaluations"])
                dispatch(q, free, children, os.environ.copy(), can_start,
                         lambda _, gpu: eligible_since.pop(gpu, None), cfg)
                if now - retention_at >= 60 and (retention is None or retention.poll() is not None):
                    retention_at = now
                    retention = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "_retain", str(q)],
                                                 start_new_session=True)
            except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
                eligible_since.clear()
                print(f"xpflow: {error}", file=sys.stderr, flush=True)
            time.sleep(period)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    submit = commands.add_parser("submit")
    submit.add_argument("--name", required=True)
    submit.add_argument("--kind", choices=("train", "eval"), required=True)
    submit.add_argument("--cwd", type=Path, required=True)
    submit.add_argument("--checkpoint", type=Path)
    submit.add_argument("--after-gpu-json")
    submit.add_argument("--depends-on", action="append", type=Path, default=[])
    submit.add_argument("--max-retries", type=int)
    submit.add_argument("--quality-gate-json")
    submit.add_argument("--series")
    submit.add_argument("--retain-series-best", action="store_true")
    submit.add_argument("command", nargs=argparse.REMAINDER)
    for name in ("status", "idle", "serve", "leaderboard"):
        commands.add_parser(name)
    for name in ("retry", "cancel"):
        commands.add_parser(name).add_argument("attempt", type=Path)
    args = parser.parse_args()
    q = queue_root()
    try:
        cfg = settings(q)
        if args.action == "idle":
            print(json.dumps(sorted(idle(cfg))))
        elif args.action == "serve":
            serve(q)
        elif args.action == "leaderboard":
            for row in ranked(q):
                print(json.dumps(row, ensure_ascii=False))
        elif args.action == "submit":
            command = args.command[1:] if args.command[:1] == ["--"] else args.command
            if not command or not args.cwd.is_absolute() or not args.cwd.is_dir():
                parser.error("A command and an existing absolute working directory are required")
            if args.max_retries is not None and args.max_retries < 0:
                parser.error("--max-retries must be nonnegative")
            known = {p.parent.resolve() for p in requests(q)}
            dependencies = [dep.resolve() for dep in args.depends_on]
            if not set(dependencies) <= known:
                parser.error("Dependencies must be existing attempts from this queue")
            options = {"max_retries": args.max_retries if args.max_retries is not None else cfg["max_retries"]}
            if args.quality_gate_json:
                options["quality_gate"] = validate_gate(json.loads(args.quality_gate_json))
            if args.series:
                if args.kind != "eval":
                    parser.error("--series is for eval jobs")
                options["series"] = args.series
            if args.retain_series_best:
                if not args.series:
                    parser.error("--retain-series-best requires --series")
                options["retain_series_best"] = True
            if args.checkpoint:
                if args.kind != "eval" or not args.checkpoint.is_absolute() or not args.checkpoint.is_file():
                    parser.error("--checkpoint requires an existing absolute file and an eval job")
                options["checkpoint"] = str(args.checkpoint)
                info = args.checkpoint.stat()
                options["checkpoint_sha256"] = checkpoint_digest(
                    args.checkpoint, info.st_ino, info.st_size, info.st_mtime_ns)
            if args.after_gpu_json:
                if args.kind != "eval":
                    parser.error("--after-gpu-json is for eval jobs")
                options["after_gpu"] = validate_argv(json.loads(args.after_gpu_json))
            if dependencies:
                options["depends_on"] = list(map(str, dependencies))
            attempt = q / "xpflow" / uuid4().hex
            attempt.mkdir(parents=True, mode=0o700)
            enqueue(attempt, args.name, args.kind, command, args.cwd, **options)
            print(attempt)
        elif args.action in {"retry", "cancel"}:
            attempt = args.attempt.resolve()
            if attempt not in {p.parent.resolve() for p in requests(q)}:
                parser.error("Supply an attempt from this queue")
            status = state(attempt, cfg)
            if args.action == "retry":
                if status not in {"failed", "cancelled", "blocked"}:
                    parser.error("Supply a failed, cancelled, or blocked attempt")
                print(retry_attempt(attempt, load(attempt / "xpflow.json"), manual=True))
            else:
                if status in TERMINAL:
                    parser.error("The attempt has already ended")
                put(attempt / "cancel", "requested")
                print(attempt)
        else:
            entries = [(path, load(path)) for path in requests(q)]
            for path, request in sorted(entries, key=lambda pair: (*request_order(pair[1]), pair[0])):
                attempt = path.parent
                row = dict(name=request["name"], kind=request["kind"], state=state(attempt, cfg),
                           attempt=str(attempt))
                for field in ("retry_count", "retry_of", "checkpoint", "checkpoint_sha256", "depends_on"):
                    if field in request:
                        row[field] = request[field]
                for field in ("gpu", "exit", "failure", "gpu.released"):
                    marker = attempt / field
                    if marker.exists():
                        row[field] = marker.read_text().strip()
                print(json.dumps(row, ensure_ascii=False))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    if sys.argv[1:2] == ["_worker"]:
        worker(Path(sys.argv[2]), int(sys.argv[3]))
    elif sys.argv[1:2] == ["_retain"]:
        root = Path(sys.argv[2])
        with exclusive(root / "retention.lock") as lock:
            if lock is not None:
                retain_artifacts(ranked(root))
    else:
        main()
