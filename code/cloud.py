"""Reuse only the explicitly retained H100. No allocation/shutdown API exists here."""
from pathlib import Path
import sys
import subprocess
import json
import base64
import datetime
import os
import re
import shlex
from common import RUN, load

ROOT = Path(__file__).resolve().parent
def api(endpoint):
    if endpoint != 'instances':
        raise RuntimeError('Only read-only instance inventory is available')
    key = os.environ.get('LAMBDA_API_KEY')
    if not key: raise RuntimeError('Lambda API credential unavailable')
    headers = 'header = "Authorization: Basic ' + base64.b64encode((key + ':').encode()).decode() + '"\n'
    result = subprocess.run(['curl', '--config', '-', '--silent', '--show-error', '--max-time', '30',
        'https://cloud.lambda.ai/api/v1/instances'], input=headers, text=True, capture_output=True, check=True)
    answer = json.loads(result.stdout)
    if 'data' not in answer: raise RuntimeError('Lambda inventory unavailable')
    return answer['data']

REMOTE = os.environ.get('ROFT_REMOTE', '/home/ubuntu/roft-pytorch')
PYTHON = sys.executable
NODE_PYTHON = os.environ.get('ROFT_NODE_PYTHON', REMOTE + '/.venv/bin/python')


def owned():
    a = load(RUN / 'allocation.json')
    n = next((r for r in api('instances') if r['id'] == a['instance_id']), None)
    if n and n['name'] != a['name']:
        raise RuntimeError('Retained node ownership mismatch')
    return n


def ssh():
    a = load(RUN / 'allocation.json')
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10',
            '-o', 'UserKnownHostsFile=' + str(RUN / 'lambda/known_hosts'),
            '-o', 'StrictHostKeyChecking=accept-new', a.get('ssh_user', 'ubuntu') + '@' + a['ip']]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def terminate(*args, **kwargs):
    raise RuntimeError('Node shutdown requires a later explicit user request; unavailable to this experiment')
