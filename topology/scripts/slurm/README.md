# SLURM job templates for topology

All SLURM templates here run on MIT Engaging / ORCD. They source `~/.bashrc`
and use the `deepcirc` activation helper (see `docs/engaging_cluster_setup.md`
§2.3 for the helper definition). Outputs go to `$DEEPCIRC_SCRATCH`.

## One-time setup

Before submitting any array job, create the log directory on scratch:

```bash
mkdir -p $DEEPCIRC_SCRATCH/logs
```

Verify the topology repo is up to date on the cluster:

```bash
cd $DEEPCIRC_REPO && git pull
```

Verify Phase 0 has been run (target list YAML must exist):

```bash
ls -l $DEEPCIRC_REPO/topology/configs/target_functions.yaml
```

If missing, run Phase 0 first:

```bash
cd $DEEPCIRC_REPO/topology/scripts
python 00_select_targets.py
```

---

## p1_ppo.sbatch — Phase 1 (topology generation)

One PPO+GAT topology run per target Boolean function. Array indices 0..19
correspond to target IDs 1..20 in `configs/target_functions.yaml` (in document
order: G1 first, then G2, then G3).

### Group submissions

```bash
cd $DEEPCIRC_REPO/topology/scripts/slurm

# G1 — pilot (4 targets: 0x2B, 0x17, 0x6D, 0xEE)
sbatch --array=0-3%1 p1_ppo.sbatch

# G2 — scale-up (6 new targets: 0x4D, 0xCC, 0xE8, 0x33, 0x66, 0x3C)
sbatch --array=4-9%1 p1_ppo.sbatch

# G3 — completion (10 new targets, indices 10..19)
sbatch --array=10-19%1 p1_ppo.sbatch
```

### About the `%1` (serial) concurrency

The `mit_normal_gpu` partition's QoS caps each user at 32 CPUs concurrent
(verified via `sacctmgr show qos mit_normal_gpu`). Our SLURM script
requests `--cpus-per-task=32` per task, which uses the full budget for one
task; tasks run sequentially. Per-task wall-clock is the upstream-claimed
~1 h, so total wall-clock per group is roughly:

| group | tasks | total wall-clock |
|---|---|---|
| G1 | 4 | ~4 h |
| G2 | 6 | ~6 h |
| G3 | 10 | ~10 h |

If you want more parallelism, use `mit_preemptable` instead — longer
walltime (48 h), larger GPU pool, but jobs can be preempted. To do that:

```bash
# Edit p1_ppo.sbatch: change `--partition=mit_normal_gpu` →
#                            `--partition=mit_preemptable`
# Add `#SBATCH --requeue` so preempted jobs auto-restart.
# Then submit with higher concurrency, e.g. %4.
```

### Resubmitting a single failed task

If, e.g., task 2 (target 0x6D) crashes:

```bash
sbatch --array=2 p1_ppo.sbatch
```

### Resource profile per task

| field | value |
|---|---|
| Partition | `mit_normal_gpu` |
| GPU | 1× L40S |
| CPUs | 40 (drives the 80 parallel env workers) |
| Memory | 64 GB |
| Walltime | 2 h (typical run is <1 h; headroom for slow seeds) |

### Outputs per target

```
$DEEPCIRC_SCRATCH/runs/stage1/<HEX>/
├── seed_verilog/
│   └── <HEX>.v                                            # auto-generated from hex
├── trained_masked/
│   ├── trained_final_shared_registry.pkl                  # registry of all unique topologies discovered
│   ├── trained_registry_summary.csv                       # hash + energy + size per topology
│   ├── trained_episode_metrics.csv                        # per-episode best-energy + wall-time
│   ├── tb_compat_steps.csv                                # step-level metrics
│   └── slurm-*.log
└── run_metadata.json                                      # CLI args + import versions + timestamps
```

### Log paths

stdout → `$DEEPCIRC_SCRATCH/logs/p1_<arrayJobId>_<taskId>.out`
stderr → `$DEEPCIRC_SCRATCH/logs/p1_<arrayJobId>_<taskId>.err`

### Monitoring while running

```bash
# All your jobs
squeue -u $USER

# Only the topology array
squeue -u $USER -n p1_ppo

# Tail a running task's stdout
tail -f $DEEPCIRC_SCRATCH/logs/p1_<JOBID>_0.out

# Check registry growth on a finished or running task
python -c "
import pickle
with open('$DEEPCIRC_SCRATCH/runs/stage1/0x2B/trained_masked/trained_final_shared_registry.pkl', 'rb') as f:
    reg = pickle.load(f)
print(f'unique topology hashes: {len(reg)}')
print(f'total entries: {sum(len(b) for b in reg.values())}')
"
```

---

## Future templates (placeholder)

- `p2_random_baselines.sbatch` — Phase 2 random NIG baseline generation (CPU)
- `p8_p10_analyses.sbatch` — Phase 8–10 cross-topology analyses (CPU)

These will be added as their respective phases come online. See
`PROJECT_STATE_TOPOLOGY.md` §3 for the full phase sequence.

---

## p1_5_parse.sbatch — Phase 1.5 (registry parsing, CPU)

Wraps upstream's `shared_registry_parsing_HDL_main.py`, producing
`optimal_topologies.pkl` per target. Required before P3.

```bash
sbatch --array=0-3%4 p1_5_parse.sbatch         # G1
sbatch --array=4-9%4 p1_5_parse.sbatch         # G2 new targets
sbatch --array=10-19%4 p1_5_parse.sbatch       # G3 new targets
```

CPU-only on `mit_normal`; ~minutes per task.

---

## p4_mlp_train.sbatch — Phase 4 (MLP training, mit_normal_gpu)

Conservative path: serial on `mit_normal_gpu`. Use this for the G1 pilot.

```bash
GROUP=G1 sbatch --array=0-91%1 p4_mlp_train.sbatch
```

Caps: 32 CPUs / 2 GPUs / 515 G mem per user → `%1` (one task at a time).
Walltime 6 h per task (the partition cap). For G1's ~92 topologies, total
wall-clock ~23 hours.

---

## p4_mlp_train_preemptable.sbatch — Phase 4 (MLP training, mit_preemptable)

Faster path: `mit_preemptable` partition with `--requeue`. Use for G2 / G3
once the pipeline is validated on G1.

```bash
GROUP=G1 sbatch --array=0-91%4   p4_mlp_train_preemptable.sbatch
GROUP=G2 sbatch --array=0-249%4  p4_mlp_train_preemptable.sbatch
GROUP=G3 sbatch --array=0-499%4  p4_mlp_train_preemptable.sbatch
```

Caps: 1024 CPUs / **4 GPUs** / 4 TB mem per user → `%4` (4 tasks parallel,
limited by GPU cap). Walltime 12 h per task. Preemption-safe via the
wrapper's idempotency + atomic dir-rename semantics.

For G1: ~6 h. For G2: ~16 h. For G3: ~31 h.

To find N for a group's array range:
```bash
N=$(jq .n_total $DEEPCIRC_SCRATCH/population/G1/population_summary.json)
sbatch --array=0-$((N-1))%4 p4_mlp_train_preemptable.sbatch
```

If a task is preempted, SLURM auto-resubmits (because of `--requeue`); the
wrapper detects the leftover `<topology_id>.tmp/` from the killed run, nukes
it, and restarts that topology from scratch. Already-completed topologies
are skipped via the idempotency check.
