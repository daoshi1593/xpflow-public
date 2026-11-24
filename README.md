# XPFlow

XPFlow is an open-source NVIDIA GPU scheduler for research teams that train and evaluate models on a shared Linux workstation. It runs your existing commands, waits for a GPU with enough headroom, and automatically queues evaluations as training writes checkpoints.

Use it when training, batch inference, and evaluation compete for the same cards, and you want comparable results without keeping every raw rollout or per-frame dump.

- **Schedule around live GPU load.** Configure how long utilization and memory use must stay below admission thresholds. A separate runtime guard stops the job when either reading reaches its red line. Reserve cards for evaluation or batch inference and decide whether those jobs can borrow training cards.
- **Evaluate checkpoints as they appear.** Watch a directory, deduplicate checkpoints by content, and run your evaluator. An optional CPU phase can check or summarize results after the GPU is released. Metric gates can block downstream jobs.
- **Keep the raw results that matter.** Rank evaluations within a series. With retention enabled, keep the raw outputs of the strict-success and mean-error winners and leave reports and rebuild manifests for the other attempts.

**Supported today:** single-node scheduling, one GPU per GPU phase. **Planned:** multi-node scheduling.

## Quick start

Use Linux with procfs and POSIX file locks, Python 3.10+, NVIDIA drivers and `nvidia-smi`. The queue uses a local filesystem with atomic rename. Run the scheduler and clients under the same Unix account.

```bash
git clone https://github.com/daoshi1593/xpflow-public.git
cd xpflow-public
```

```bash
export XPFLOW_HOME="$HOME/.local/share/xpflow"
python3 xpflow.py serve
```

In a second terminal at the repository root, submit a smoke-test command and inspect the queue.

```bash
export XPFLOW_HOME="$HOME/.local/share/xpflow"
python3 xpflow.py submit --name demo --kind train --cwd "$PWD" -- \
  python3 -c 'import os; print("Assigned GPU:", os.environ["CUDA_VISIBLE_DEVICES"])'
python3 xpflow.py status
python3 xpflow.py idle
```

One foreground scheduler holds the queue lock. Detached workers survive scheduler restarts. Every attempt has its own request, combined log and state files. Status emits JSON lines with state, GPU, exit, failure and retry lineage. `idle` shows instantaneous eligibility. Admission also requires sustained low readings and a worker-side recheck.

## Try automatic checkpoint evaluation

Run this from the repository root on a Linux host with an eligible NVIDIA GPU. It uses an isolated queue and a toy checkpoint so you can try the workflow without a training framework or dataset. The toy evaluator reports fixed metrics and does no CUDA computation.

```bash
export XPFLOW_HOME="$(mktemp -d)"
python3 - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["XPFLOW_HOME"])
(root / "checkpoints").mkdir()
(root / "evaluate.py").write_text(
    "from pathlib import Path\nimport sys\n"
    "print('Evaluated', Path(sys.argv[1]).read_text())\n"
)
(root / "audit.py").write_text(
    "import json, os\nfrom pathlib import Path\n"
    "assert os.environ['CUDA_VISIBLE_DEVICES'] == ''\n"
    "attempt = Path(os.environ['XPFLOW_ATTEMPT'])\n"
    "(attempt / 'report.json').write_text(json.dumps("
    "{'strict_success': 105, 'mean_error': 0.012, 'unsafe': 0}))\n"
)
(root / "config.json").write_text(json.dumps({
    "checkpoint_watches": [{
        "name": "demo-eval",
        "cwd": str(root),
        "glob": "checkpoints/*.ckpt",
        "argv": ["python3", "evaluate.py", "{checkpoint}"],
        "after_gpu": ["python3", "audit.py"],
        "series": "demo-v1",
        "quality_gate": {"strict_success": {"min": 100}, "unsafe": {"max": 0}},
    }],
}))
PY
printf 'demo model\n' > "$XPFLOW_HOME/checkpoints/step-1.tmp"
mv "$XPFLOW_HOME/checkpoints/step-1.tmp" "$XPFLOW_HOME/checkpoints/step-1.ckpt"
printf 'In a second terminal use XPFLOW_HOME=%s\n' "$XPFLOW_HOME"
python3 xpflow.py serve
```

In a second terminal at the repository root, set and export the printed `XPFLOW_HOME`, then run the following commands. Give the watcher at least 10 seconds after publication and allow time for GPU admission. Once the attempt completes, the leaderboard shows `demo-v1` with 105 strict successes and a mean error of 0.012.

```bash
python3 xpflow.py status
python3 xpflow.py leaderboard
```

For a real project, point the watch at your checkpoint directory, replace both commands with your scripts, and write the evaluation metrics to `report.json`. To enable cleanup, write disposable outputs under the attempt's `artifacts/` directory and set `retain_series_best` on the watch.

## Scheduling and recovery

- Training precedes evaluation. New training submissions precede training retries, with FIFO within each class. Evaluations with `--checkpoint /absolute/file` prioritize newer checkpoint modification times. Evaluations without checkpoint metadata run FIFO.
- XPFlow stops a GPU phase at either runtime threshold or if GPU sampling fails. The measurements cover the whole card. Other activity and the job's own computation can trip the guard.
- Failed attempts retry up to the configured limit. Override it with `--max-retries N`, including zero. Run `python3 xpflow.py retry /absolute/path/to/ended-attempt` to retry manually. Repeated retry requests for one predecessor reuse its successor. A manual retry starts a fresh allowance when it creates a successor. The command handles any model checkpoint resumption.
- Run `python3 xpflow.py cancel /absolute/path/to/attempt` to request termination of the job's process group and suppress automatic retry for the attempt. Recovery verifies Linux boot ID and process start time before terminating a recorded orphan. An unidentified process holding the worker lock can keep a card reserved until it exits.
- Repeat `--depends-on /absolute/attempt` to require successful prerequisites. Dependencies refer to exact attempts. A failed, cancelled or blocked prerequisite blocks its dependents.

## Evaluation options

```bash
python3 xpflow.py submit --name regression --kind eval --cwd "$PWD" \
  --checkpoint "$PWD/models/model.pth" \
  --after-gpu-json '["python3","audit.py"]' \
  --series regression-v1 --retain-series-best \
  --quality-gate-json '{"strict_success":{"min":100},"unsafe":{"max":10}}' \
  -- python3 evaluate.py models/model.pth
```

- **Phase handoff.** The optional CPU command starts after the GPU command exits successfully. XPFlow releases the GPU between phases. The GPU command must finish its GPU child work before exiting. Both commands share a log and count as one evaluation. XPFlow exports `CUDA_VISIBLE_DEVICES`, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `XPFLOW_GPU`, `XPFLOW_ATTEMPT` and `XPFLOW_PHASE`. The CPU phase receives empty `CUDA_VISIBLE_DEVICES` and must avoid GPU work. In argv, `{attempt}` expands to the current attempt path, including on retry.
- **Reports and quality gates.** Write a numeric metrics object to `$XPFLOW_ATTEMPT/report.json`. Gates apply inclusive `min` or `max` bounds. Missing or nonfinite metrics fail the gate and block dependents. Evaluation results that fail a gate can still appear in the leaderboard.
- **Leaderboard and retention.** For `--series NAME`, provide nonnegative finite `strict_success` and `mean_error`. The leaderboard ranks strict success descending, then mean error ascending. Use a separate series for each evaluation protocol and unit convention. With explicit `--retain-series-best`, the background sweep keeps raw files for the strict-success and mean-error winners. Only disposable output belongs in an opted-in attempt's `artifacts/` directory. For each nonwinner the worker writes its request, report and artifact inventory to `rebuild.json` before removing that directory. Rebuilding requires the original inputs.
- **Checkpoint discovery.** Watches enqueue each watch/content pair once across restarts. XPFlow waits at least ten seconds after file modification and checks the discovered content hash again before running. Publish completed files by atomic rename. Changing a watch creates a new watch identity.

## Configuration

Put `config.json` under `XPFLOW_HOME`. The scheduler rereads it every poll. Workers read it on startup.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `admit_util`, `admit_memory` | 30, 60 | Strict upper bounds on GPU utilization and memory percentages |
| `admit_seconds`, `poll_seconds` | 10, 3 | Admission window and scheduler poll period |
| `stop_util`, `stop_memory` | 78, 80 | Stop at or above either percentage during GPU work |
| `guard_seconds`, `observe_seconds` | 2, 60 | Worker polling interval and initial observing window |
| `max_retries`, `max_evaluations` | 3, 8 | Automatic retries per chain and concurrent evaluations |
| `eval_gpus`, `eval_fallback` | [], true | Evaluation-preferred physical GPU indices excluded from training, and whether evaluations can fall back to other cards |
| `pause_evaluations`, `min_free_bytes` | false, 0 | Pause new evaluations or require free space on the queue filesystem |
| `checkpoint_watches` | [] | Automatic checkpoint evaluation rules |

### GPU admission and runtime red lines

Admission checks two readings from `nvidia-smi` for each physical GPU: reported utilization, and `100 × used memory / total memory`. **Both** readings must be strictly below their `admit_*` thresholds at each scheduler sample for at least `admit_seconds`. The worker rechecks the readings before starting the command.

Once the GPU phase starts, **either** reading at or above its `stop_*` threshold trips the red line. With the defaults, a utilization reading of 78% or a memory reading of 80% is enough. A missing or invalid GPU sample also stops the GPU phase. The worker sends SIGTERM to its job's process group, waits three seconds, then escalates to SIGKILL if the group is still present. A failed attempt may retry under `max_retries`. Use `--max-retries 0` to keep a failed job from automatically running again.

The red line applies to the **whole card**, including the assigned job's own load and any external GPU activity. Thresholds are integers from 1 to 100, with an inclusive stop comparison, so even `stop_util: 100` trips on a 100% reading. Both checks are active. Changes to red lines apply to workers started after the change.

### Put training and inference on separate GPUs

On a four-GPU host, this example puts training on GPUs 0–2 and batch inference or evaluation on GPU 3. Save it as `$XPFLOW_HOME/config.json`:

```json
{
  "admit_util": 20,
  "admit_memory": 60,
  "admit_seconds": 10,
  "poll_seconds": 3,
  "stop_util": 90,
  "stop_memory": 85,
  "guard_seconds": 2,
  "eval_gpus": [3],
  "eval_fallback": false,
  "max_evaluations": 2
}
```

```bash
python3 xpflow.py submit --name train --kind train --cwd "$PWD" -- python3 train.py
python3 xpflow.py submit --name infer --kind eval --cwd "$PWD" -- python3 infer.py
```

Use `--kind eval` for inference jobs routed to the evaluation pool. With `eval_fallback: false`, they wait for a reserved card. Set it to `true` to let them use a free training-pool card when no reserved card is available. Training always excludes GPUs listed in `eval_gpus`. An empty list makes all eligible GPUs available to both job kinds. These are queue-wide pools. The scheduler picks the lowest-index eligible card in the chosen pool, and the submit CLI exposes pool selection through the job kind.

GPU indices in `config.json` are physical `nvidia-smi` indices. If XPFlow picks GPU 3, the child gets `XPFLOW_GPU=3` and `CUDA_VISIBLE_DEVICES=3`; a CUDA framework normally sees that single visible device as `cuda:0`. Use `XPFLOW_GPU` for physical-card logging and the framework's local index for allocations. `xpflow.py status` reports the physical assignment. With a CPU postprocessing command, a GPU can admit another job while the audit runs, subject to the parallel evaluation limit.

### Watch checkpoints for automatic evaluation

```json
{
  "checkpoint_watches": [
    {
      "name": "regression",
      "cwd": "/absolute/project",
      "glob": "checkpoints/*.pth",
      "argv": ["python3", "evaluate.py", "{checkpoint}"],
      "after_gpu": ["python3", "audit.py"],
      "series": "regression-v1",
      "retain_series_best": true
    }
  ]
}
```

The glob is relative to `cwd`. `{checkpoint}` expands to the discovered path. `after_gpu`, `series`, `retain_series_best` and `quality_gate` are optional. A watch's `quality_gate` takes the same metric bounds as `--quality-gate-json`. The leaderboard includes additional finite numeric report metrics, such as collision counts or speed summaries.

## Limits and tests

XPFlow runs trusted local commands using the scheduler account. It assigns one GPU per GPU phase. Independent queues and external processes do not share its reservations. MIG scheduling, multi-GPU jobs, CPU-only submissions, a web service and multi-user isolation are outside the current implementation. Runtime logs and job requests can contain commands and paths. Keep the queue private.

Run `python3 -m unittest discover -s tests -v` for CPU-only tests. Validate real GPU behavior on the deployment host.

## Contributing

For bug reports, include your Linux and Python versions, the relevant XPFlow configuration, and a minimal command or checkpoint watch that reproduces the issue. Remove private paths or credentials from logs before sharing them. For code changes, include the relevant test results and describe any real-GPU checks you ran.

## License and origin

[MIT](LICENSE). XPFlow is independently implemented. Its scheduling design draws inspiration from [gflow](https://github.com/AndPuQing/gflow). See [project origin](docs/ORIGIN.md).
