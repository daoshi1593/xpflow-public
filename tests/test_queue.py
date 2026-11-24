import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import xpflow


class QueueTests(unittest.TestCase):
    def queue_job(self, root, name, kind="eval", **options):
        job = root / "xpflow" / name
        job.mkdir(parents=True)
        xpflow.enqueue(job, name, kind, ["echo", "demo"], root, **options)
        return job

    def test_enqueue_priority_and_state(self):
        with tempfile.TemporaryDirectory() as root:
            job = Path(root) / "job"
            job.mkdir()
            xpflow.enqueue(job, "demo", "train", ["echo", "hi"], root)
            self.assertEqual(xpflow.load(job / "xpflow.json")["args"], ["echo", "hi"])
            self.assertEqual(xpflow.state(job), "queued")
            self.assertLess(xpflow.queue_order("train", 99), xpflow.queue_order("eval", 1))
            xpflow.put(job / "exit", "0")
            self.assertEqual(xpflow.state(job), "complete")

    def test_strict_admission_boundaries(self):
        with patch.object(xpflow, "gpu_samples", return_value=[(0, 29, 59, 100), (1, 30, 0, 100), (2, 0, 60, 100)]):
            self.assertEqual(xpflow.idle(), {0})

    def test_only_own_queue_is_discovered(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            for folder in [root / "xpflow/a", root / "registered/b"]:
                folder.mkdir(parents=True)
                (folder / "xpflow.json").write_text("{}")
            self.assertEqual(xpflow.requests(root), [root / "xpflow/a/xpflow.json"])

    def test_quality_gate_requires_finite_metrics(self):
        gate = xpflow.validate_gate({"strict_success": {"min": 80}, "unsafe": {"max": 2}})
        self.assertTrue(xpflow.passes_gate({"strict_success": 80, "unsafe": 2}, gate))
        self.assertFalse(xpflow.passes_gate({"strict_success": 95, "unsafe": float("nan")}, gate))
        self.assertFalse(xpflow.passes_gate({"strict_success": 95}, gate))

    def test_watch_deduplicates_content_and_detects_replaced_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            ckpt = root / "model.pth"
            ckpt.write_bytes(b"first")
            epoch = time.time() - 20
            os.utime(ckpt, (epoch, epoch))
            watch = dict(name="eval", cwd=folder, glob="*.pth", argv=["eval", "{checkpoint}", "{attempt}"])
            xpflow.discover_checkpoints(root, [watch])
            xpflow.discover_checkpoints(root, [watch])
            files = xpflow.requests(root)
            self.assertEqual(len(files), 1)
            request = xpflow.load(files[0])
            self.assertTrue(xpflow.checkpoint_matches(request))
            self.assertEqual(request["args"][-1], "{attempt}")
            ckpt.write_bytes(b"next")
            self.assertFalse(xpflow.checkpoint_matches(request))

    def test_checkpoint_priority_and_training_retry_order(self):
        with tempfile.TemporaryDirectory() as folder:
            older, newer = (Path(folder) / n for n in ("old", "new"))
            for index, file in enumerate((older, newer), 1):
                file.touch()
                os.utime(file, (index, index))
            eval_order = lambda path: xpflow.request_order(dict(kind="eval", checkpoint=str(path), submitted_ns=99))
            self.assertLess(eval_order(newer), eval_order(older))
            self.assertLess(xpflow.request_order(dict(kind="train", submitted_ns=99)),
                            xpflow.request_order(dict(kind="train", submitted_ns=1, retry_of="first")))

    @unittest.skipUnless(sys.platform == "linux", "POSIX file locks")
    def test_restart_reserves_gpu_while_worker_is_starting(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            starting = self.queue_job(root, "starting", "train")
            waiting = self.queue_job(root, "waiting", "train")
            xpflow.put(starting / "gpu", "0")
            children = {}
            with xpflow.exclusive(starting / "running.lock"), patch.object(xpflow.subprocess, "Popen"):
                xpflow.dispatch(root, {0, 1}, children, {}, lambda *_: True, lambda *_: None)
            self.assertEqual(children[waiting][1], 1)
            self.assertNotIn(starting, children)

    @unittest.skipUnless(sys.platform == "linux", "POSIX file locks")
    def test_workers_cannot_start_on_an_already_locked_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = self.queue_job(root, "waiting", "train")
            with xpflow.exclusive(xpflow.gpu_lock_path(root, 0)), patch.object(xpflow, "run_command") as run:
                xpflow.worker(job, 0)
            run.assert_not_called()
            self.assertEqual(xpflow.state(job), "queued")

    @unittest.skipUnless(sys.platform == "linux", "Linux process groups")
    def test_state_write_failure_reaps_spawned_process_before_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = self.queue_job(root, "write-error", "train")
            request = xpflow.load(job / "xpflow.json")
            request["args"] = [sys.executable, "-c", "import time; time.sleep(30)"]
            xpflow.save(job / "xpflow.json", request)
            processes = []
            spawn = xpflow.subprocess.Popen
            save = xpflow.save

            def capture(*args, **kwargs):
                process = spawn(*args, **kwargs)
                processes.append(process)
                return process

            def fail_process_record(path, value):
                if path.name == "process.json":
                    raise OSError("simulated identity write failure")
                save(path, value)

            try:
                with (patch.object(xpflow, "gpu_samples", return_value=[(0, 0, 0, 100)]),
                      patch.object(xpflow.subprocess, "Popen", side_effect=capture),
                      patch.object(xpflow, "save", side_effect=fail_process_record),
                      patch.object(xpflow.time, "sleep")):
                    xpflow.worker(job, 0)
                self.assertEqual(xpflow.state(job), "failed")
                self.assertEqual(len(processes), 1)
                self.assertIsNotNone(processes[0].poll())
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=5)

    @unittest.skipUnless(sys.platform == "linux", "POSIX file locks")
    def test_retry_and_gpu_pool_during_cpu_postprocessing(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            first = self.queue_job(root, "first", "train")
            xpflow.finish(first, 1)
            request = xpflow.load(first / "xpflow.json")
            retry = xpflow.retry_attempt(first, request)
            self.assertEqual(retry, xpflow.retry_attempt(first, request))
            self.assertEqual(xpflow.load(retry / "xpflow.json")["retry_count"], 1)
            current = self.queue_job(root, "audit")
            xpflow.put(current / "gpu", "1")
            xpflow.put(current / "gpu.released", "1")
            waiting = self.queue_job(root, "waiting")
            children = {current: (object(), 1)}
            cfg = {**xpflow.DEFAULTS, "eval_gpus": [1], "eval_fallback": False}
            with patch.object(xpflow, "reconcile"), patch.object(xpflow.subprocess, "Popen") as popen:
                xpflow.dispatch(root, {0, 1}, children, {}, lambda *_: True, lambda *_: None, cfg)
            self.assertEqual(children[retry][1], 0)
            self.assertEqual(children[waiting][1], 1)
            self.assertEqual(popen.call_count, 2)

    @unittest.skipUnless(sys.platform == "linux", "POSIX file locks and process identities")
    def test_worker_runs_cpu_phase_with_empty_cuda_visibility(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            job = self.queue_job(root, "phases", series="demo",
                                 after_gpu=[sys.executable, "-c",
                                            "import os; assert not os.environ['CUDA_VISIBLE_DEVICES'];"
                                            "assert os.environ['XPFLOW_PHASE']=='cpu'"])
            request = xpflow.load(job / "xpflow.json")
            request["args"] = [sys.executable, "-c",
                               "import os; assert os.environ['CUDA_VISIBLE_DEVICES']=='0'"]
            xpflow.save(job / "xpflow.json", request)
            with patch.object(xpflow, "gpu_samples", return_value=[(0, 0, 0, 100)]):
                xpflow.worker(job, 0)
            self.assertEqual(xpflow.state(job), "complete")
            self.assertTrue((job / "gpu.released").is_file())

    @unittest.skipUnless(sys.platform == "linux", "POSIX file locks")
    def test_retention_keeps_two_winners_and_writes_rebuild_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            jobs = {}
            for name, strict, error, enabled in (
                ("strict", 100, 2, True), ("mean", 80, 1, True),
                ("loser", 50, 5, True), ("unmanaged", 30, 8, False),
            ):
                job = self.queue_job(root, name, series="demo", retain_series_best=enabled)
                jobs[name] = job
                (job / "artifacts").mkdir()
                xpflow.put(job / "artifacts" / "trace", "raw")
                xpflow.save(job / "report.json", {"strict_success": strict, "mean_error": error})
                xpflow.finish(job, 0)
            xpflow.retain_artifacts(xpflow.ranked(root))
            self.assertFalse((jobs["loser"] / "artifacts").exists())
            self.assertTrue((jobs["loser"] / "rebuild.json").is_file())
            for key in ("strict", "mean", "unmanaged"):
                self.assertTrue((jobs[key] / "artifacts" / "trace").is_file())

    @unittest.skipUnless(sys.platform == "linux", "Linux scheduler integration")
    def test_live_scheduler_reuses_gpu_before_cpu_phase_ends(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            mock = root / "nvidia-smi"
            mock.write_text("#!/bin/sh\nprintf '0, 0, 0, 100\\n'\n")
            mock.chmod(0o700)
            xpflow.save(root / "config.json", {"admit_seconds": 0, "poll_seconds": 1,
                                               "guard_seconds": 1, "max_retries": 0})
            released = root / "cpu-started"
            allow_exit = root / "allow-exit"
            audit = self.queue_job(root, "audit", after_gpu=[
                sys.executable, "-c",
                "import os,time,sys; from pathlib import Path;"
                "assert os.environ['XPFLOW_PHASE']=='cpu';"
                "Path(sys.argv[1]).touch();"
                "raise SystemExit(0 if any(Path(sys.argv[2]).exists() or time.sleep(.1) for _ in range(200)) else 1)",
                str(released), str(allow_exit),
            ])
            env = dict(os.environ, PATH=f"{root}{os.pathsep}{os.environ['PATH']}", XPFLOW_HOME=folder,
                       PYTHONDONTWRITEBYTECODE="1")
            def wait_for(condition):
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if condition():
                        return
                    time.sleep(.05)
                self.fail("Timed out waiting for the scheduler")

            with subprocess.Popen([sys.executable, xpflow.__file__, "serve"], env=env) as scheduler:
                try:
                    wait_for(released.exists)
                    self.assertEqual(xpflow.state(audit), "postprocessing")
                    train = self.queue_job(root, "train", "train")
                    wait_for(lambda: xpflow.state(train) == "complete")
                    self.assertEqual(xpflow.state(audit), "postprocessing")
                    allow_exit.touch()
                    wait_for(lambda: xpflow.state(audit) == "complete")
                finally:
                    allow_exit.touch()
                    scheduler.terminate()


if __name__ == "__main__":
    unittest.main()
