"""Diagnose and audit an already-saved update. Performs no backward or optimizer step."""
import gc
import gzip
import json
import sys
import tempfile
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from common import RUN, MODEL, config, load, save, sha
from train import base, process, logprobs
from transformers import AutoProcessor


def cast_parameters(model, dtype):
    for parameter in model.parameters():
        parameter.data = parameter.data.to(dtype=dtype)


def main():
    folder = RUN / 'train/roft/checkpoint-001'
    if (folder / 'audit.json').exists():
        raise RuntimeError('Existing audit must not be overwritten')
    torch.set_num_threads(8)
    state_sha = sha(folder / 'resume.pt.gz')
    with tempfile.TemporaryDirectory(prefix='.recovery-reader-', dir=folder) as temporary:
        native = Path(temporary) / 'resume.pt'
        with native.open('wb') as output:
            result = subprocess.run(['pigz', '-d', '-c', str(folder / 'resume.pt.gz')],
                stdout=output, stderr=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError('Saved recovery state decompression failed')
        state = torch.load(native, map_location='cpu', weights_only=False, mmap=True)
    counters = sorted({int(v['step']) for v in state['optimizer']['state'].values()})
    if counters != [1]:
        raise RuntimeError('Already-saved optimizer update does not have step1')
    records = load(RUN / 'jobs/roft-001-targets.json')
    microbatches = [json.loads(line) for line in (RUN / 'train/roft/microbatches.jsonl').read_text().splitlines()]
    if [r['id'] for r in records] != [r['id'] for r in microbatches]:
        raise RuntimeError('All admitted backwards must be recorded exactly once')
    total = sum(r['backward_tokens'] for r in microbatches)
    if total != load(RUN / 'train/roft/selection-001.json')['selected_tokens']:
        raise RuntimeError('Saved backward ledger does not match selected targets')
    processor = AutoProcessor.from_pretrained(str(MODEL))
    first = records[0]
    short = dict(first, token_ids=first['token_ids'][:16], supervised_positions=list(range(16)))
    probe, labels, _ = process(processor, short)
    model = base()
    names = [n for n, p in model.named_parameters()]
    if names != state['names']:
        raise RuntimeError('Saved FP32 master parameter identities differ')
    changed = 0
    subsystems = set()
    for (name, parameter), value in zip(model.named_parameters(), state['masters']):
        if value.dtype != torch.float32 or value.shape != parameter.shape:
            raise RuntimeError('Saved native master identity or dtype differs')
        if not torch.equal(parameter.detach().cpu(), value):
            changed += 1
            subsystems.add('vision' if '.visual.' in name else 'language')
        parameter.data.copy_(value.to(parameter.device))
    model.eval()
    native_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    # Recreate the exact failed export/probe: this incorrectly rounds rotary buffers.
    model.to(dtype=torch.bfloat16)
    model.to(dtype=torch.float32)
    bad_buffers = {n: b.detach().clone() for n, b in model.named_buffers()}
    bad = logprobs(model, probe, labels, False).cpu()
    # Fix only the diagnostic model's buffers. Parameter serving values stay identical.
    for name, value in model.named_buffers():
        value.data.copy_(native_buffers[name])
    good = logprobs(model, probe, labels, False).cpu()
    buffer_errors = {n: float((native_buffers[n].float() - value.float()).abs().max())
        for n, value in bad_buffers.items() if not torch.equal(native_buffers[n], value)}
    parameter_count = sum(p.numel() for p in model.parameters())
    del model
    gc.collect()
    torch.cuda.empty_cache()
    reloaded = base(folder)
    reloaded.eval()
    actual = logprobs(reloaded, probe, labels, False).cpu()
    torch.testing.assert_close(actual, good, rtol=.02, atol=.04)
    max_good = float((actual - good).abs().max())
    max_bad = float((actual - bad).abs().max())
    if not buffer_errors or max_bad <= max_good:
        raise RuntimeError('Buffer-rounding cause not established against corrected reload')
    for name, expected in load(folder / 'weight_manifest.json').items():
        if sha(folder / name) != expected:
            raise RuntimeError('Saved serving weights changed')
    receipt = {'at': time.time(), 'diagnosis': 'model.to(BF16) rounded nonpersistent rotary buffers; saved checkpoint regenerates them at native FP32 precision',
        'nonpersistent_buffer_rounding_errors': buffer_errors,
        'failed_probe_reproduced_max_logprob_error': max_bad,
        'correct_buffer_probe_reload_max_logprob_error': max_good,
        'same_existing_serving_weights': True, 'same_existing_optimizer_rng_state': True,
        'resume_sha256': state_sha, 'backward_tokens_replayed': 0,
        'optimizer_updates_performed_during_recovery': 0,
        'original_failure_retained': True, 'numerical_reload_verified': True}
    save(folder / 'recovery_reload_receipt.json', receipt)
    audit = {'arm': 'roft', 'step': 1, 'qualification': False, 'supervised_tokens': total,
        'optimizer_updates': 1, 'actual_changed_tensors': changed,
        'gradient_parameter_tensors': len(state['optimizer']['state']),
        'eligible_parameter_tensors': len(names), 'trainable_parameter_count': parameter_count,
        'full_parameter_training': True, 'changed_subsystems': sorted(subsystems),
        'loss': sum(r['loss_contribution'] for r in microbatches),
        'gradient_norm_after_data_scale_before_clipping': None,
        'gradient_norm_provenance': 'not persisted before failed original probe; cannot reconstruct unclipped norm',
        'data_gradient_scale': config()['data_gradient_scale'],
        'optimizer_step_counters': counters, 'resumed_optimizer_step_counters': [],
        'weight_manifest_sha256': sha(folder / 'weight_manifest.json'),
        'resume_sha256': state_sha, 'numerical_reload_verified': True,
        'reload_max_logprob_error': max_good, 'rollout_logprob_mean_errors': [],
        'target_ids': [r['id'] for r in records], 'fp32_gradient_accumulation': True,
        'native_leaf_gradient_storage': 'FP32', 'compute_precision': 'BF16 autocast',
        'fp32_master_weights_and_adam_state': True, 'model_precision': 'BF16',
        'replayed_targets': 0, 'cumulative_backward_tokens': total,
        'selection_sha256': sha(RUN / 'train/roft/selection-001.json'),
        'recovered_saved_update': True, 'recovery_receipt_sha256': sha(folder / 'recovery_reload_receipt.json')}
    save(folder / 'audit.json', audit)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
