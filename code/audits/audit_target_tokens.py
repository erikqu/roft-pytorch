"""Independently check actual token labels for a saved target batch on CPU."""
import argparse
import collections
import json
from pathlib import Path


def main():
    from transformers import AutoProcessor
    p = argparse.ArgumentParser()
    p.add_argument('root', type=Path)
    p.add_argument('--step', type=int, required=True)
    p.add_argument('--model', type=Path, required=True)
    args = p.parse_args()
    root = args.root
    tokenizer = AutoProcessor.from_pretrained(args.model).tokenizer
    targets = json.loads((root / 'jobs' / f'roft-{args.step:03d}-targets.json').read_text())
    selection = json.loads((root / 'train' / 'roft' / f'selection-{args.step:03d}.json').read_text())
    problems = []
    boundaries = []
    counts = collections.Counter()
    backward_tokens = 0
    for target in targets:
        identity = target['id']
        raw = json.loads((root / 'generations' / (identity + '.json')).read_text())
        text = target['preprocessed_target']
        encoded = tokenizer.encode(text, add_special_tokens=False)
        positions = target.get('supervised_positions', list(range(len(encoded))))
        if encoded != target['token_ids']:
            problems.append({'id': identity, 'reason': 'Target text and actual token IDs differ'})
        if raw['token_ids'] != target['raw_generation_token_ids']:
            problems.append({'id': identity, 'reason': 'Raw generation token IDs changed'})
        if not text.endswith('<|im_end|>\n'):
            problems.append({'id': identity, 'reason': 'End-of-turn newline missing'})
        if not text.startswith('<think>') or '</think>' not in text:
            counts['without_nonempty_thinking'] += 1
        else:
            thought = text.split('</think>', 1)[0].removeprefix('<think>').strip()
            if thought:
                counts['nonempty_thinking_retained'] += 1
            else:
                problems.append({'id': identity, 'reason': 'Leading empty thinking block retained'})
        if positions != list(range(len(positions))):
            problems.append({'id': identity, 'reason': 'Noncontiguous supervision mask'})
        if len(positions) != len(encoded):
            boundaries.append(identity)
        backward_tokens += len(positions)
        counts['targets'] += 1
        counts['tokens_after_preprocessing'] += len(encoded)
        counts['raw_generated_tokens'] += len(raw['token_ids'])
    if len(boundaries) > 1:
        problems.append('More than one partially supervised boundary target')
    if backward_tokens != selection['selected_tokens']:
        problems.append('Actual supervised token count differs from selection ledger')
    result = {'step': args.step, 'counts': dict(counts),
        'actual_supervised_tokens': backward_tokens,
        'boundary_targets': boundaries, 'problems': problems,
        'verified': not problems, 'read_only': True,
        'optimizer_update_counted': False,
        'note': 'Only the later saved numerical update proves these tokens completed training.'}
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
