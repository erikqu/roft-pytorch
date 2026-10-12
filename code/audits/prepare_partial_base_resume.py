"""Resume frozen base histories using original completed action evidence."""
import json
from pathlib import Path, PurePosixPath
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUN, load, save, sha
from tool_protocol import history_message, parse_calls
from agent_env import AgentEnvironment


def main():
    jobs = load(RUN / 'jobs/evaluation-base-blender-generated-fictional-face-angular-features-model-seed-17.json')
    recovered = []
    completed = []
    unstarted = []
    for job in jobs:
        identity = job['id']
        generation = RUN / 'generations' / (identity + '.json')
        folder = RUN / 'episodes' / identity
        if generation.exists():
            completed.append(identity)
            continue
        files = sorted((RUN / 'raw_turns').glob(identity + '-*.json'))
        if not files:
            unstarted.append(identity)
            continue
        turns = []
        for index, path in enumerate(files):
            raw = load(path)
            if raw['id'] != identity + f'/turn-{index:03d}' or raw['checkpoint_sha256'] is not None or raw['seed'] != job['seed'] + index * 1000003:
                raise RuntimeError('Original base raw identity or seed differs')
            turn = dict(raw)
            if index + 1 < len(files):
                nxt = load(files[index + 1])
                observation = nxt['sampling_messages'][-1]
                if observation['role'] != 'tool':
                    raise RuntimeError('Original tool feedback missing')
                turn['observation'] = observation['content']
            else:
                results = []
                for ordinal, call in enumerate(parse_calls(raw['text'])):
                    name, args = call['name'], call['arguments']
                    if name == 'bash':
                        receipt_path = folder / f'tool-{index:03d}-{ordinal:02d}.receipt.json'
                        if not receipt_path.exists():
                            recovery = load(folder / f'tool-{index:03d}-{ordinal:02d}.lost_observation_recovery.json')
                            if recovery['raw_sha256'] != sha(path) or recovery['id'] != raw['id'] or recovery['ordinal'] != ordinal or recovery['action_reexecuted']:
                                raise RuntimeError('Lost observation recovery identity differs')
                            observed = recovery['observation']
                            if observed['observation_status'] != 'lost_during_supervisor_interruption' or any(observed[k] is not None for k in ('exit_code', 'stdout', 'stderr', 'python_traceback_observed')):
                                raise RuntimeError('Lost tool output must remain unknown')
                            fact = observed['observed_file_facts']
                            target = folder / 'app' / str(PurePosixPath(fact['script_path']).relative_to('/app'))
                            if sha(target) != fact['script_sha256']:
                                raise RuntimeError('Recovered script changed before continuation')
                            results.append(observed)
                            continue
                        receipt = load(receipt_path)
                        stdout = folder / f'tool-{index:03d}-{ordinal:02d}.stdout'
                        stderr = folder / f'tool-{index:03d}-{ordinal:02d}.stderr'
                        if stdout.stat().st_size != receipt['stdout_bytes'] or stderr.stat().st_size != receipt['stderr_bytes']:
                            raise RuntimeError('Completed action output size differs')
                        out, err = stdout.read_bytes(), stderr.read_bytes()
                        results.append({'tool': 'bash', 'exit_code': receipt['exit_code'],
                            'stdout': out.decode(errors='replace'), 'stderr': err.decode(errors='replace'),
                            'python_traceback_observed': b'Traceback (most recent call last)' in err,
                            'note': 'A Blender exit_code=0 can still contain a Python traceback; inspect stderr.'})
                    elif name == 'write_file':
                        requested = PurePosixPath(args['path'])
                        if not requested.is_absolute() or not requested.is_relative_to('/app') or '..' in requested.parts:
                            raise RuntimeError('Unexpected last write path')
                        target = folder / 'app' / str(requested.relative_to('/app'))
                        if not target.resolve().is_relative_to((folder / 'app').resolve()) or target.read_bytes() != args['content'].encode():
                            raise RuntimeError('Original last write does not match sampled bytes')
                        results.append({'tool': name, 'path': args['path'],
                                        'bytes': target.stat().st_size, 'sha256': sha(target)})
                    else:
                        raise RuntimeError('Unsupported final action; separate qualification required')
                turn['observation'] = json.dumps(results, ensure_ascii=False)
            turns.append(turn)
        evidence = {'at': time.time(), 'raw_turn_hashes': {p.name: sha(p) for p in files},
            'sampled_actions_repeated': False, 'raw_tokens_replaced': False,
            'existing_task_volume_and_container': True,
            'last_actions_completed_evidence_verified': not any(r.get('observation_status') == 'lost_during_supervisor_interruption' for r in results),
            'lost_tool_observation_count': sum(r.get('observation_status') == 'lost_during_supervisor_interruption' for r in results)}
        environment = AgentEnvironment.resume(job, evidence)
        # Read-only attach; do not call close/act/finish on the retained container.
        messages = turns[-1]['sampling_messages'] + [history_message(turns[-1]['text']),
                    {'role': 'tool', 'content': turns[-1]['observation']}]
        save(folder / 'partial_resume.json', {'job': job, 'checkpoint_sha256': None,
            'messages': messages, 'turns': turns, 'total_tokens': sum(len(t['token_ids']) for t in turns),
            'context_overflow_proof': None, 'recovery_evidence': evidence})
        recovered.append({'id': identity, 'original_turns': len(turns),
            'original_container': environment.name, 'next_seed': job['seed'] + len(turns) * 1000003,
            'raw_turn_hashes': evidence['raw_turn_hashes']})
    if len(completed) + len(recovered) + len(unstarted) != 34:
        raise RuntimeError('Frozen continuation membership differs from admitted full set')
    receipt = {'at': time.time(), 'verified': True, 'completed_preserved': completed,
        'partial_histories': recovered, 'unstarted': unstarted,
        'actions_reexecuted': False, 'training_repeated': False, 'raw_tokens_unchanged': True}
    save(RUN / 'base_partial_resume_preparation.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'partial_histories'}), flush=True)


if __name__ == '__main__':
    main()
