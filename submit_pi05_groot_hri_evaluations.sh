#!/usr/bin/env bash
# One Slurm allocation, sequential models; each runner owns environment/client cleanup.
# Dry-run is the default. Smoke: HRI_NUM_TRIALS=2 bash "$0" --submit
# Resume: set the same HRI_RUN_ID; valid scored seeds with videos are skipped.
set -Eeuo pipefail
export HRI_BASELINE_ROOT="${HRI_BASELINE_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)}"
export HRI_NUM_TRIALS="${HRI_NUM_TRIALS:-50}"
export HRI_START_SEED="${HRI_START_SEED:-0}"
export HRI_RUN_ID="${HRI_RUN_ID:-pi05_groot_hri_$(date -u +%Y%m%d_%H%M%S)}"
export HRI_TASKS="${HRI_TASKS:-hri_000 hri_001 hri_002 hri_003 hri_004 hri_005 hri_006 hri_007 hri_008 hri_009}"
export HRI_MODELS="${HRI_MODELS:-pi05 groot}"
export HRI_JOB_TIME="${HRI_JOB_TIME:-3-00:00:00}"
mode="${1:---dry-run}"
[[ "$#" -le 1 ]] || { echo "Usage: $0 [--dry-run|--submit|--run]" >&2; exit 2; }
case "$mode" in
  --submit)
    bash "${HRI_BASELINE_ROOT}/submit_pi05_groot_hri_evaluations.sh" --dry-run
    # Export an explicit root: sbatch copies this script into its spool directory.
    exec sbatch --parsable --job-name=pi05_groot_hri --partition=cluster02 \
      --gres=gpu:rtx5090:1 --cpus-per-task=4 --mem=48G --time="${HRI_JOB_TIME}" \
      --chdir="${HRI_BASELINE_ROOT}" --export=ALL \
      --output="${HRI_BASELINE_ROOT}/slurm-pi05-groot-hri-%j.out" \
      --error="${HRI_BASELINE_ROOT}/slurm-pi05-groot-hri-%j.err" \
      "${HRI_BASELINE_ROOT}/submit_pi05_groot_hri_evaluations.sh" --run
    ;;
  --dry-run|--run) ;;
  *) echo "Usage: $0 [--dry-run|--submit|--run]" >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1
exec "${HRI_BASELINE_ROOT}/.conda/lerobot-inference/bin/python" -u - "$mode" <<'PY'
"""Submission-level orchestration; policy/environment behavior stays in the runners."""
import ast
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

root = Path(os.environ['HRI_BASELINE_ROOT']).resolve()
workspace = root.parent
motionforge = Path(os.environ.get('MOTIONFORGE_ROOT', workspace / 'MotionForge')).resolve()
dry = sys.argv[1] == '--dry-run'
run_id = os.environ['HRI_RUN_ID']
if not re.fullmatch(r'[A-Za-z0-9_.-]+', run_id) or run_id in {'.', '..'}:
    raise ValueError('HRI_RUN_ID must be a simple directory name')
count = int(os.environ['HRI_NUM_TRIALS'])
start = int(os.environ['HRI_START_SEED'])
if count < 1 or start < 0:
    raise ValueError('Require HRI_NUM_TRIALS >= 1 and HRI_START_SEED >= 0')
tasks = os.environ['HRI_TASKS'].split()
models = os.environ['HRI_MODELS'].split()
if not tasks or len(set(tasks)) != len(tasks) or any(not re.fullmatch(r'hri_00[0-9]', t) for t in tasks):
    raise ValueError('Select unique HRI000-HRI009 tasks')
if not models or len(set(models)) != len(models) or any(m not in {'pi05', 'groot'} for m in models):
    raise ValueError('HRI_MODELS must select unique pi05/groot entries')
desired = set(range(start, start + count))
job = root / 'results' / 'human_robot_interaction' / run_id
groot_root = root / 'Isaac-GR00T'
deployment = groot_root / 'gr00t_trt_deployments' / 'gr00t_trt_deployment_gr00t1.7-HRI-80000'
dataset = Path(os.environ.get('GROOT_HRI_DATASET', workspace / 'datasets/lerobot_hri_old'))
backbone = workspace / 'ckpts/nvidia/Cosmos-Reason2-2B'
ffmpeg = Path(os.environ.get('GROOT_FFMPEG_PREFIX', workspace / 'ICRA2027-baseline/.conda/lerobot-baselines'))
specs = {
    'pi05': dict(prefix='PI05', runner=root / 'pi05/run_hri000_hri009_evaluation.sh',
                 model=workspace / 'ckpts/pi05-HRI-80000/pretrained_model',
                 out=root / 'pi05/output', port=35796),
    'groot': dict(prefix='GROOT', runner=groot_root / 'examples/MotionForge/run_hri000_hri009_evaluation.sh',
                  model=workspace / 'ckpts/gr00t1.7-HRI-80000',
                  out=groot_root / 'examples/MotionForge/outputs', port=35896),
}
for spec in specs.values():
    spec['out'] = spec['out'] / 'human_robot_interaction/hri000_hri009/level2_fixed/server_scheduled' / run_id


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def log(message):
    print(f'[HRI-CHAIN {datetime.now(timezone.utc).isoformat(timespec="seconds")}] {message}', flush=True)


def save(path, data):
    # All generated metadata is confined to this run; replace atomically on resume.
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(data)
    temp.replace(path)


def environment(name, seeds, segment, preflight=False):
    spec = specs[name]
    p = spec['prefix']
    env = os.environ.copy()
    # Do not inherit another suite's run selection, ports, or checkpoint overrides.
    for key in list(env):
        if key.startswith((p + '_EVAL_', 'MOTIONFORGE_LIGHTING_', 'MOTIONFORGE_BACKGROUND_', 'MOTIONFORGE_VISUAL_ASSET_')):
            del env[key]
    env.update({
        'MOTIONFORGE_ROOT': str(motionforge), 'MOTIONFORGE_DEFAULT_DEVICE': 'cpu',
        'MOTIONFORGE_HRI006_DEVICE': 'cuda:0', 'MOTIONFORGE_LIGHTING_MODE': 'fixed',
        'MOTIONFORGE_PYTHON': str(workspace / 'isaacsim/python.sh'),
        'LEROBOT_ROOT': str(root / 'lerobot'), 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONOPTIMIZE': '',
        'DRY_RUN': '1' if preflight else '0', p + '_SCRIPT_DIR': str(spec['runner'].parent),
        p + '_MODEL_PATH': str(spec['model']), p + '_EVAL_NUM_TRIALS': str(len(seeds)),
        p + '_EVAL_START_SEED': str(seeds[0]), p + '_EVAL_ATTEMPTS_PER_WORKER': '50',
        p + '_EVAL_HRI006_ATTEMPTS_PER_WORKER': '1', p + '_EVAL_OBS_PORT': str(spec['port']),
        p + '_EVAL_ACT_PORT': str(spec['port'] + 2), p + '_EVAL_RUN_ID': segment,
        p + '_EVAL_OUTPUT_ROOT': str(spec['out'] / 'attempts'),
        p + '_EVAL_TASKS': ' '.join(tasks),
    })
    if name == 'groot':
        env.update(GROOT_ACCEL_MODE='pytorch' if preflight else 'trt_full_pipeline',
                   GROOT_TRT_ENGINE_PATH=str(deployment / 'engines'),
                   GROOT_FFMPEG_PREFIX=str(ffmpeg),
                   LD_PRELOAD=str(ffmpeg / 'lib/libstdc++.so.6') + (' ' + env['LD_PRELOAD'] if env.get('LD_PRELOAD') else ''),
                   GROOT_BACKBONE_MODEL_PATH=str(backbone), GR00T_BACKBONE_MODEL_PATH=str(backbone),
                   LD_LIBRARY_PATH=str(ffmpeg / 'lib') + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''))
    return env


def scored(name, task):
    """Count only valid official results backed by the exact recorded outcome video."""
    found = {}
    base = specs[name]['out'] / 'attempts'
    for path in sorted(base.glob(f'*/{task}/server.log')):
        videos = {}
        for line in path.open(errors='replace'):
            match = re.search(r'trial_video trial=(\d+)/\d+ outcome=(success|failure) path=(.+)', line)
            if match:
                videos[int(match[1])] = (match[2], Path(match[3].strip()))
            match = re.search(r'trial_result trial=(\d+)/\d+ seed=(\d+).*? result=(\{.*\})', line)
            if not match:
                continue
            seed = int(match[2])
            if seed not in desired:
                continue
            try:
                result = ast.literal_eval(match[3])
            except (ValueError, SyntaxError):
                continue  # Incomplete line from an interrupted writer is not a result.
            official = result.get('metrics', {}).get('official_result', {})
            outcome = official.get('outcome')
            video = videos.get(int(match[1]))
            if (official.get('counts_toward_score') is True and official.get('requires_rerun') is False
                    and official.get('seed') == seed and outcome in {'success', 'failure'}
                    and video and video[0] == outcome and video[1].is_file() and video[1].stat().st_size > 0):
                if seed in found:
                    raise RuntimeError(f'Duplicate valid seed: {name}/{task}/{seed}; inspect existing attempts')
                found[seed] = {'success': outcome == 'success', 'video': str(video[1]), 'server_log': str(path)}
    return found


def seed_ranges(missing):
    groups = []
    for seed in sorted(missing):
        if not groups or seed != groups[-1][-1] + 1:
            groups.append([])
        groups[-1].append(seed)
    return groups


active = None


def stop_child():
    if active is not None and active.poll() is None:
        os.killpg(active.pid, signal.SIGTERM)
        try:
            active.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(active.pid, signal.SIGKILL)
            active.wait()


def cancelled(signum, frame):
    stop_child()
    raise SystemExit(128 + signum)


signal.signal(signal.SIGTERM, cancelled)
signal.signal(signal.SIGINT, cancelled)


def run(command, env, cwd, timeout, label):
    global active
    log(f'START {label}: {shlex.join(map(str, command))}')
    active = subprocess.Popen(command, env=env, cwd=cwd, start_new_session=True)
    started = time.monotonic()
    try:
        while True:
            try:
                rc = active.wait(timeout=60)
                log(f'END {label} rc={rc}')
                return rc
            except subprocess.TimeoutExpired:
                if time.monotonic() - started > timeout:
                    log(f'TIMEOUT {label}; terminating process group')
                    stop_child()
                    return 124
                log(f'RUNNING {label} elapsed={int(time.monotonic() - started)}s')
    finally:
        stop_child()
        active = None


manifest = {'run_id': run_id, 'models': models, 'tasks': tasks, 'trials_per_task': count,
            'start_seed': start, 'checkpoints': {}, 'benchmarks': {}, 'runners': {}}
for name in models:
    spec = specs[name]
    manifest['checkpoints'][name] = {
        'path': str(spec['model']), 'config_sha256': sha(spec['model'] / 'config.json'),
        'weights': {p.name: [p.stat().st_size, p.stat().st_mtime_ns]
                    for p in sorted(spec['model'].glob('*.safetensors'))},
    }
    manifest['runners'][name] = sha(spec['runner'])
for task in tasks:
    manifest['benchmarks'][task] = sha(motionforge / f'configs/benchmarks/human_robot_interaction/{task}_rgb_gr00t_zmq.yaml')
manifest_path = job / 'manifest.json'
if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
    raise RuntimeError('Run manifest differs; choose a new HRI_RUN_ID instead of mixing configurations')
if not manifest_path.exists() and any(specs[n]['out'].exists() for n in models):
    raise RuntimeError('Existing model output has no matching run manifest; inspect before resuming')

for name in models:
    cmd = ['bash', str(specs[name]['runner'])]
    # The CPU preflight validates GR00T checkpoint/commands without claiming TRT readiness.
    result = subprocess.run(cmd, env=environment(name, sorted(desired), 'preflight', True), cwd=root,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        print(result.stdout)
        raise RuntimeError(f'{name} preflight failed rc={result.returncode}')
    log(f'PREFLIGHT_OK {name}; tasks={len(tasks)} trials={count}; HRI006 worker=1')
    for task in tasks:
        log(f'PLAN {name}/{task}: completed={len(scored(name, task))}/{count}, missing={seed_ranges(desired - scored(name, task).keys())}')

if 'groot' in models:
    runtime = subprocess.run([str(groot_root / '.venv/bin/python'), '-c', 'import torchcodec; print(torchcodec.__version__)'],
                             env=environment('groot', sorted(desired), 'codec_preflight', True),
                             capture_output=True, text=True)
    if runtime.returncode:
        raise RuntimeError('GR00T TorchCodec runtime failed: ' + runtime.stderr)
    log('TORCHCODEC_OK version=' + runtime.stdout.strip())
    for filename in ['info.json', 'modality.json', 'episodes.jsonl', 'tasks.jsonl', 'stats.json']:
        if not (dataset / 'meta' / filename).is_file():
            raise FileNotFoundError(dataset / 'meta' / filename)
    log(f'TRT calibration/verification dataset={dataset}; engine={deployment}')
if dry:
    log('DRY_RUN_OK: no jobs, engines, or result directories created')
    raise SystemExit(0)
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Use --submit to allocate a GPU before --run')
job.mkdir(parents=True, exist_ok=True)
run_lock = (job / '.run.lock').open('a')
try:
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise RuntimeError('This HRI_RUN_ID already has an active job; refusing duplicate trials')
save(manifest_path, json.dumps(manifest, indent=2) + '\n')
overall = 0
trt_ready = 'groot' not in models

if 'groot' in models:
    env = environment('groot', sorted(desired), 'trt_build')
    env.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONPATH=str(groot_root))
    stamp = deployment / 'verified_hri_checkpoint.json'
    signature = {'checkpoint': manifest['checkpoints']['groot'],
                 'model_index_sha256': sha(specs['groot']['model'] / 'model.safetensors.index.json'),
                 'processor_sha256': sha(specs['groot']['model'] / 'processor_config.json'),
                 'mode': 'full_pipeline', 'gpu': 'rtx5090'}
    engines = ['state_encoder.engine', 'action_encoder.engine', 'dit_bf16.engine', 'action_decoder.engine',
               'vit.engine', 'llm_bf16.engine', 'vl_self_attention.engine', 'export_metadata.json']
    complete = all((deployment / 'engines' / f).is_file() and (deployment / 'engines' / f).stat().st_size > 0 for f in engines)
    trt_ready = complete and stamp.exists() and json.loads(stamp.read_text()) == signature
    if not trt_ready:
        # Verification returns a cosine value; enforce a threshold instead of accepting a printed warning.
        code = '''import sys, json
from pathlib import Path
sys.path.insert(0, "scripts/deployment")
from build_trt_pipeline import PipelineConfig, main
from verify_n1d7_trt import VerifyConfig, main as verify
from gr00t.data.embodiment_tags import EmbodimentTag
model, dataset, output, steps = sys.argv[1:]
main(PipelineConfig(model_path=model, dataset_path=dataset, output_dir=output,
    embodiment_tag="NEW_EMBODIMENT", export_mode="full_pipeline", video_backend="torchcodec", steps=steps))
cosine = verify(VerifyConfig(model_path=model, dataset_path=dataset, engine_dir=output+"/engines",
    mode="n17_full_pipeline", embodiment_tag=EmbodimentTag.NEW_EMBODIMENT, batch_size=1, video_backend="torchcodec"))
if not cosine > 0.99:
    raise RuntimeError(f"TensorRT verification cosine {cosine} <= 0.99")
print("HRI_TRT_VERIFY_OK", cosine, flush=True)
'''
        command = [str(groot_root / '.venv/bin/python'), '-u', '-c', code, str(specs['groot']['model']), str(dataset), str(deployment), 'export,build']
        rc = run(command, env, groot_root, int(os.environ.get('GROOT_BUILD_TIMEOUT_S', '7200')), 'GR00T TensorRT build/verify')
        trt_ready = rc == 0
        if trt_ready:
            save(stamp, json.dumps(signature, indent=2) + '\n')
        else:
            overall = 1
            log('GR00T TensorRT failed; Pi0.5 will still run, GR00T evaluation remains incomplete')

summary = {}


def update_summary():
    for name in models:
        summary[name] = {}
        for task in tasks:
            records = scored(name, task)
            summary[name][task] = {'completed': len(records), 'expected': count,
                                   'successes': sum(v['success'] for v in records.values()),
                                   'missing_seeds': sorted(desired - records.keys()), 'trials': records}
    save(job / 'progress.json', json.dumps(summary, indent=2) + '\n')
    lines = ['model task completed expected successes status']
    for name, entries in summary.items():
        for task, data in entries.items():
            status = 'completed' if data['completed'] == count else 'incomplete'
            lines.append(f"{name} {task} {data['completed']} {count} {data['successes']} {status}")
    save(job / 'success_rates.txt', '\n'.join(lines) + '\n')


for name in models:
    if name == 'groot' and not trt_ready:
        update_summary()
        continue
    for task in tasks:
        missing = desired - scored(name, task).keys()
        if not missing:
            log(f'SKIP {name}/{task}: all {count} valid scored seeds already recorded')
        for seeds in seed_ranges(missing):
            segment = f'{task}_seeds_{seeds[0]}_{seeds[-1]}_{time.time_ns()}'
            env = environment(name, seeds, segment)
            env[specs[name]['prefix'] + '_EVAL_TASKS'] = task
            rc = run(['bash', str(specs[name]['runner'])], env, root,
                     int(os.environ.get('HRI_STAGE_TIMEOUT_S', '14500')), f'{name}/{task}/seeds={seeds[0]}-{seeds[-1]}')
            if rc:
                overall = 1
                log(f'FAILED {name}/{task} rc={rc}; continuing with the next task')
            update_summary()
        update_summary()
if any(data['missing_seeds'] for entries in summary.values() for data in entries.values()):
    overall = 1
log(f'FINISHED exit={overall}; summary={job / "success_rates.txt"}')
raise SystemExit(overall)
PY
