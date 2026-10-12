"""Isolated ROFT/GRPO experiment identities and durable artifacts."""
import hashlib
import json
import os
from pathlib import Path
import time

CODE = Path(__file__).resolve().parent
RUN = Path(os.environ.get('ROFT_RUN', CODE / 'artifacts')).resolve()
MODEL = Path(os.environ.get('ROFT_MODEL', RUN / 'model'))
SEED = 20261008
TRAIN_SLUGS = ['cowboy-character', 'astronaut-complete-standing-suit',
    'fictional-face-soft-features', 'right-hand-open-palm', 'right-foot-flat-resting-pose',
    'suburban-ranch-house-complete-exterior', 'white-house-complete-exterior',
    'skyscraper-exposed-platforms-and-pillars', 'clock-tower', 'pagoda']
TRANSFER_SLUGS = ['fictional-face-angular-features', 'left-hand-relaxed-fingers',
    'left-foot-raised-heel-pose', 'gazebo', 'greenhouse', 'stone-gatehouse']

def task(slug): return 'blender-generated-' + slug + '-model'
def load(path): return json.loads(Path(path).read_text())
def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)
def append(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as f:
        f.write(json.dumps(data, allow_nan=False) + '\n'); f.flush(); os.fsync(f.fileno())
def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''): result.update(block)
    return result.hexdigest()
def digest(value): return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()
def status(stage, **kwargs):
    record = dict(stage=stage, updated_at=time.time(), **kwargs)
    save(RUN / 'status.json', record); print(json.dumps(record), flush=True)
def config(): return load(RUN / 'protocol.json')
def prompt_rows(): return {r['task']: r for r in load(RUN / 'prompts.json')}
def messages(row):
    return [{'role':'system','content':'You are a Blender modeling agent working in a task container. Use the available tools to inspect files, make focused edits, and run commands. Solve the task, validate when practical, then give a brief final answer. Work in /app. Blender is installed; use blender --background --threads 4 --python-exit-code 1 --python /app/script.py. Save the final scene yourself as /app/result.blend; no automatic save is performed.'}, {'role':'user', 'content':[{'type':'text','text':row['text']}] +
        [{'type':'image','url':str(RUN / name)} for name in row['images']]}]
def visible(text): return text.split('</think>', 1)[1].strip() if '</think>' in text else ''

def reflection_instruction():
    original = (CODE / 'official/failed_system.txt').read_text().strip()
    return original.replace('software engineer', 'Blender modeling agent').replace(
        'coding task', 'Blender modeling task').replace('test suite', 'published verifier') + (
        '\n\nGround claims in the recorded evidence. A program that did not execute '
        'does not establish that its proposed geometry or proportions were correct. '
        'Do not invent an executed repair, hidden reference geometry, or unobserved '
        'visual differences. Distinguish output-budget termination, Python errors, '
        'execution success and modeling quality.')

def reflection_feedback(grade):
    execution = grade['execution']
    summary = {'execution': execution['status'], 'saved': execution['saved'],
        'error_type': execution.get('error_type'), 'error': str(execution.get('error', ''))[:3000],
        'traceback': str(execution.get('traceback', ''))[-3000:],
        'finish_reason': grade['finish_reason'], 'generated_tokens': grade['output_tokens'],
        'termination_reason': grade.get('termination_reason', grade['finish_reason']),
        'reward': grade.get('reward'), 'geometry': grade.get('geometry'),
        'visual': grade.get('visual'), 'applied_caps': grade.get('applied_caps', []),
        'raw_verdict': grade.get('raw_verdict', 'unknown'),
        'verifier_failure':grade.get('verifier_failure'),
        'last_tool_observation': execution.get('last_tool_observation',''),
        'file_exists':execution.get('file_exists',execution.get('saved',False)),
        'blend_header_valid':execution.get('blend_header_valid'),
        'validation_status':grade.get('validation_status','unknown'),
        'evidence_observed': 'Recorded execution and actual verifier outputs; reward is not a causal explanation.'}
    return 'Recorded feedback for your attempt:\n' + json.dumps(summary, ensure_ascii=False)

def validate_prepared():
    manifest = load(RUN / 'manifest.json')
    for name, expected in manifest['files'].items():
        if sha(RUN / name) != expected: raise RuntimeError('Frozen input changed: ' + name)
    for name, expected in manifest['code'].items():
        if sha(CODE / name) != expected: raise RuntimeError('Frozen code changed: ' + name)
    for name, expected in manifest.get('official', {}).items():
        if sha(CODE / 'official' / name) != expected: raise RuntimeError('Pinned paper asset changed: ' + name)
    return manifest
