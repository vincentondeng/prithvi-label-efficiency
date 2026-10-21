"""Run a priority-ordered queue of training jobs, one job per GPU at a time.
Skips finished jobs (result file exists), kills runs that exceed the time limit,
records failures and keeps going."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

STAGES = {
    1: dict(fractions=[0.10, 1.0], seeds=range(0, 5)),
    2: dict(fractions=[0.25, 0.50, 0.05, 0.01], seeds=range(0, 5)),
    3: dict(fractions=[0.10, 0.05, 0.01], seeds=range(5, 10)),
}
INITS = ["pretrained", "scratch"]

p = argparse.ArgumentParser()
p.add_argument("--stages", type=int, nargs="+", default=[1])
p.add_argument("--fractions", type=float, nargs="*", default=None, help="restrict to these fractions")
p.add_argument("--steps", type=int, default=4000)
p.add_argument("--batch-size", type=int, default=4)
p.add_argument("--gpus", type=int, default=2)
p.add_argument("--timeout-min", type=float, default=45)
p.add_argument("--max-jobs", type=int, default=None, help="only run the first N pending jobs (testing)")
p.add_argument("--out-dir", default="results")
p.add_argument("--root", default="data/sen1floods11")
p.add_argument("--splits-dir", default="data/splits")
p.add_argument("--dry-run", action="store_true")
args = p.parse_args()

jobs = []
for st in args.stages:
    cfg = STAGES[st]
    fracs = [f for f in cfg["fractions"] if args.fractions is None or f in args.fractions]
    for seed in cfg["seeds"]:
        for frac in fracs:
            for init in INITS:
                jobs.append((init, frac, seed))

out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
Path("logs").mkdir(exist_ok=True)


def result_path(job):
    init, frac, seed = job
    return out / f"{init}_f{frac:g}_s{seed}.json"


pending = [j for j in jobs if not result_path(j).exists()]
print(f"{len(jobs)} jobs in the selected stages, {len(jobs) - len(pending)} already done, {len(pending)} pending")
if args.max_jobs:
    pending = pending[: args.max_jobs]
if args.dry_run:
    for j in pending:
        print("  ", j)
    sys.exit(0)

slots, failed, n_done = {}, [], 0
t_start = time.time()


def stamp():
    return time.strftime("%H:%M:%S")


while pending or slots:
    for gpu in range(args.gpus):
        if gpu not in slots and pending:
            init, frac, seed = job = pending.pop(0)
            log = open(f"logs/{init}_f{frac:g}_s{seed}.log", "w")
            cmd = [sys.executable, "scripts/train.py", "--init", init, "--fraction", str(frac), "--seed", str(seed),
                   "--steps", str(args.steps), "--batch-size", str(args.batch_size), "--eval-every", "0",
                   "--root", args.root, "--splits-dir", args.splits_dir, "--out-dir", args.out_dir]
            proc = subprocess.Popen(cmd, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu)),
                                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            slots[gpu] = (proc, job, time.time(), log)
            print(f"[{stamp()}] gpu{gpu} start {job} | {len(pending)} still queued", flush=True)
    time.sleep(5)
    for gpu, (proc, job, t0, log) in list(slots.items()):
        rc = proc.poll()
        if rc is None and time.time() - t0 > args.timeout_min * 60:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            rc = "timeout"
        if rc is not None:
            log.close()
            ok = rc == 0 and result_path(job).exists()
            minutes = (time.time() - t0) / 60
            if ok:
                n_done += 1
            else:
                failed.append((job, rc))
                with open(out / "failed.txt", "a") as f:
                    f.write(f"{job} rc={rc}\n")
            print(f"[{stamp()}] gpu{gpu} {'done' if ok else 'FAILED rc=' + str(rc)} {job} in {minutes:.1f} min", flush=True)
            del slots[gpu]

print(f"finished: {n_done} done, {len(failed)} failed, {(time.time() - t_start) / 60:.1f} min total")
for job, rc in failed:
    print("  failed:", job, rc)
