"""Inspect saved FP32 masters/Adam/RNG without modifying the checkpoint or learner."""
import argparse
import contextlib
import gzip
import hashlib
import json
import tempfile
import subprocess
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def audit(folder, base_folder=None):
    import torch
    from safetensors import safe_open
    torch.set_num_threads(4)
    row = json.loads((folder / 'audit.json').read_text())
    state_path = folder / 'resume.pt.gz'
    if sha(state_path) != row['resume_sha256']:
        raise RuntimeError('Saved optimizer/RNG state hash mismatch')
    with tempfile.TemporaryDirectory(prefix='.audit-state-reader-', dir=folder) as temporary:
        native = Path(temporary) / 'resume.pt'
        with native.open('wb') as output:
            result = subprocess.run(['pigz', '-d', '-c', str(state_path)],
                stdout=output, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError('Saved state audit decompression failed')
        state = torch.load(native, map_location='cpu', weights_only=False, mmap=True)
    names = state['names']
    masters = state['masters']
    optimizer = state['optimizer']
    problems = []
    if len(names) != len(masters) or len(set(names)) != len(names):
        problems.append('Parameter/master identities are incomplete or duplicated')
    if len(masters) != row['eligible_parameter_tensors']:
        problems.append('Saved master tensor count differs from measured learner')
    if sum(p.numel() for p in masters) != row['trainable_parameter_count']:
        problems.append('Saved master parameter count differs from measured learner')
    if any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in masters):
        problems.append('Master weights are not CPU FP32')
    if any(not bool(torch.isfinite(p).all()) for p in masters):
        problems.append('Nonfinite saved master weights')
    moments = optimizer['state']
    recipe = {
        'lr': 1e-6, 'betas': (0.9, 0.98), 'eps': 1e-8,
        'weight_decay': 0.1, 'maximize': False, 'foreach': False,
    }
    for group in optimizer['param_groups']:
        for key, expected in recipe.items():
            if group.get(key) != expected:
                problems.append('Saved optimizer recipe mismatch: ' + key)
    parameter_indices = [index for group in optimizer['param_groups'] for index in group['params']]
    if sorted(parameter_indices) != list(range(len(masters))):
        problems.append('Optimizer groups do not contain all masters exactly once')
    counters = sorted({int(v['step']) for v in moments.values()})
    if counters != [row['step']]:
        problems.append('Adam step counters differ from measured update')
    if len(moments) != row['gradient_parameter_tensors']:
        problems.append('Adam moment count differs from native gradient count')
    for index, values in moments.items():
        if not (0 <= index < len(masters)):
            problems.append('Adam state names an unknown master')
            continue
        for name in ('exp_avg', 'exp_avg_sq'):
            value = values[name]
            if value.dtype != torch.float32 or value.shape != masters[index].shape:
                problems.append('Adam moment dtype or shape differs from its master')
            if not bool(torch.isfinite(value).all()):
                problems.append('Nonfinite saved Adam moments')
    rng_valid = all(isinstance(state[k], torch.Tensor) and state[k].dtype == torch.uint8
                    and state[k].numel() for k in ('cpu_rng', 'cuda_rng'))
    if not rng_valid:
        problems.append('CPU/CUDA RNG state missing or invalid')
    lookup = dict(zip(names, masters))
    checked = 0
    unrounded = 0
    changed_serving_tensors = 0
    serving_changes = []
    rounding_examples = []
    manifest = json.loads((folder / 'weight_manifest.json').read_text())
    base_streams = contextlib.ExitStack()
    base_lookup = {}
    if base_folder:
        for path in base_folder.glob('*.safetensors'):
            stream = base_streams.enter_context(safe_open(path, framework='pt', device='cpu'))
            base_lookup.update({name: stream for name in stream.keys()})
    for filename, expected in manifest.items():
        if sha(folder / filename) != expected:
            problems.append('Serving weight hash mismatch: ' + filename)
        with safe_open(folder / filename, framework='pt', device='cpu') as stream:
            for name in stream.keys():
                if name not in lookup:
                    problems.append('Serving tensor has no saved FP32 master: ' + name)
                    continue
                master = lookup[name]
                served = stream.get_tensor(name)
                rounded = master.to(dtype=served.dtype)
                if not torch.equal(rounded, served):
                    problems.append('Serving export differs from rounded master: ' + name)
                if base_folder:
                    if name not in base_lookup:
                        problems.append('Serving tensor missing from pinned base: ' + name)
                    elif not torch.equal(served, base_lookup[name].get_tensor(name)):
                        changed_serving_tensors += 1
                        if len(serving_changes) < 5:
                            serving_changes.append(name)
                if master.dtype == torch.float32 and not torch.equal(master, rounded.float()):
                    unrounded += 1
                    if len(rounding_examples) < 5:
                        rounding_examples.append(name)
                checked += 1
    base_streams.close()
    if not checked:
        problems.append('No serving/master tensor comparison completed')
    if base_folder and not changed_serving_tensors:
        problems.append('No numerical serving weight changes relative to pinned base')
    return {
        'checkpoint': str(folder), 'step': row['step'],
        'resume_sha256': row['resume_sha256'],
        'saved_master_tensors': len(masters),
        'saved_master_parameters': sum(p.numel() for p in masters),
        'master_dtype': 'FP32', 'adam_moment_tensors': len(moments),
        'adam_step_counters': counters, 'cpu_cuda_rng_valid': bool(rng_valid),
        'saved_adam_recipe': recipe,
        'serving_master_tensors_compared': checked,
        'masters_with_precision_beyond_serving_export': unrounded,
        'unrounded_master_examples': rounding_examples,
        'base_serving_model': str(base_folder) if base_folder else None,
        'numerically_changed_serving_tensors_from_base': changed_serving_tensors if base_folder else None,
        'serving_change_examples': serving_changes,
        'problems': problems, 'verified': not problems,
        'read_only': True, 'optimizer_update_performed': False,
        'note': 'This validates saved state. Actual subsequent optimizer continuity requires the next update audit.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--base-model', type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.checkpoint, args.base_model), indent=2))
