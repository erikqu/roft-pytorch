"""Loopback-only Azure judging bridge through an SSH reverse tunnel.

Credentials stay on the original local host. The new experiment has its own
spend/cache history; no GRPO ledger or configuration is changed.
"""
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading
import time

from common import RUN, load, save, config

sys.path.insert(0,str(RUN/'source_harness/scripts'))
sys.path.insert(0,str(RUN/'source_harness/vendor'))
from judge_profile import configured_judge
from azure_judge import transport_for
from auth import AzureToken, azure_config

CONFIG=load(RUN/'source_harness/configs/reward.json')
CONFIG['judge_budget_usd']=config()['judge_budget_usd']
CONFIG['judge_azure_limits']['max_concurrent']=4
JUDGE=configured_judge(CONFIG)
AUTH=AzureToken(azure_config())
SEM=threading.BoundedSemaphore(4)
STATE=RUN/'judging';STATE.mkdir(exist_ok=True)

class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args): pass

    def do_POST(self):
        if self.path!='/judge':self.send_error(404);return
        try:
            length=int(self.headers['Content-Length'])
            if length>50*1024*1024:raise ValueError('Oversized evidence request')
            body=json.loads(self.rfile.read(length))
            raw=base64.b64decode(body['request_bytes'],validate=True)
            request=json.loads(raw)
            budget_class=body['budget_class']
            if budget_class not in ['training','evaluation']:raise ValueError('Invalid budget class')
            budget=config()['judge_'+budget_class+'_cap_usd']
            ledger=STATE/(budget_class+'-spend.jsonl')
            evidence={k:base64.b64decode(v,validate=True) for k,v in body['evidence'].items()}
            JUDGE.validate_request(request,evidence,raw)
            identity=JUDGE.digest(raw)
            def reserve_retry(payload, attempt):
                for ordinal in range(1,request['passes']+1):
                    expected,maximum=JUDGE.request_for(request,evidence,ordinal)
                    if expected==payload:
                        JUDGE.reserve(ledger,f'uncertain-{identity}-{attempt}-{time.time_ns()}',
                                      maximum,budget,CONFIG['judge_model'])
                        return
                raise RuntimeError('Retry did not match the bound request')
            transport=transport_for(CONFIG,STATE/'azure-transport',reserve_retry=reserve_retry,token_provider=AUTH)
            with SEM:
                attestation=JUDGE.judge_request(request,evidence,request_bytes=raw,
                    cache_dir=STATE/'cache',ledger_path=ledger,budget_usd=budget,
                    api_key=None,transport=transport,cache_wait_timeout_s=900)
            data=json.dumps(attestation).encode()
            self.send_response(200);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(data)));self.end_headers();self.wfile.write(data)
            print('JUDGMENT_COMPLETE',identity,flush=True)
        except Exception as error:
            # Never include credential helper output or HTTP authorization in logs.
            save(STATE/f'failure-{time.time_ns()}.json',{'type':type(error).__name__,'message':str(error)[:400]})
            print('JUDGMENT_FAILED',type(error).__name__,str(error)[:200],flush=True)
            self.send_error(503,'Judgment failed; inspect preserved host diagnostics')

if __name__=='__main__':
    port = config()['judge_port']
    print('JUDGE_BRIDGE_READY 127.0.0.1:'+str(port),flush=True)
    ThreadingHTTPServer(('127.0.0.1',port),Handler).serve_forever()
