"""Use the existing renewable Azure helper without persisting credentials."""
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import tomllib

def azure_config():
    path=Path(os.environ['EXPERIMENT_AZURE_CONFIG']).expanduser()
    return tomllib.loads(path.read_text())['model_providers']['azure']

class AzureToken:
    def __init__(self,config):self.auth=config['auth'];self.lock=threading.Lock();self.token=None;self.expires=0
    def __call__(self):
        with self.lock:
            if self.token and time.monotonic()<self.expires:return self.token
            r=subprocess.run([self.auth['command']],capture_output=True,text=True,timeout=self.auth.get('timeout_ms',25000)/1000)
            if r.returncode:raise RuntimeError('Azure helper failed; credentials were not logged')
            raw=r.stdout.strip()
            try:
                parsed=json.loads(raw);token=parsed.get('access_token') or parsed.get('token') or parsed.get('bearer_token')
            except ValueError:token=raw
            if not isinstance(token,str) or not token:raise RuntimeError('No Azure token returned')
            self.token=token;self.expires=time.monotonic()+240
            return token
