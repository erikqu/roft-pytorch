"""Persistent candidate container with native tools and durable completion evidence."""
import json
from pathlib import Path,PurePosixPath
import shutil
import time
import subprocess
from common import RUN,config,save,sha,load
from grade import IMAGE,docker
from tool_protocol import parse_calls

class AgentEnvironment:
    @classmethod
    def resume(cls,job,receipt):
        obj=cls.__new__(cls);obj.job=job;obj.folder=RUN/'episodes'/job['id'];obj.app=obj.folder/'app'
        obj.name='roft-v5-agent-'+job['id'];obj.started=time.monotonic();obj.closed=False;obj.observations=[]
        if not obj.app.exists():raise RuntimeError('Original partial task volume missing')
        p=docker(['inspect','--format','{{.State.Running}}',obj.name],10)
        if p.returncode or p.stdout.strip()!=b'true':raise RuntimeError('Original partial container unavailable')
        started=docker(['inspect','--format','{{.State.StartedAt}}',obj.name],10)
        if started.returncode:raise RuntimeError('Original partial container start evidence missing')
        from datetime import datetime
        import re
        timestamp=started.stdout.decode().strip().replace('Z','+00:00')
        timestamp=re.sub(r'(\.\d{6})\d+(?=[+-])',r'\1',timestamp)
        wall_started=datetime.fromisoformat(timestamp).timestamp()
        obj.started=time.monotonic()-(time.time()-wall_started)
        obj.recovery_receipt=receipt
        return obj
    def __init__(self,job):
        self.job=job;self.folder=RUN/'episodes'/job['id'];self.app=self.folder/'app'
        self.folder.mkdir(parents=True,exist_ok=True)
        if self.app.exists():raise RuntimeError('Existing unfinished episode requires audited recovery')
        shutil.copytree(RUN/'tasks'/job['task']/'environment/workspace',self.app)
        (self.app/'result.blend').unlink(missing_ok=True)
        self.name='roft-v5-agent-'+job['id'];self.started=time.monotonic();self.closed=False;self.observations=[]
        args=['run','-d','--init','--name',self.name,'--network','none','--cpus','4','--memory','8g',
            '--pids-limit','256','--cap-drop','ALL','--security-opt','no-new-privileges','--shm-size','256m',
            '-e','OMP_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=1','-v',str(self.app)+':/app','-w','/app',
            '--entrypoint','/bin/sh',IMAGE,'-c','sleep infinity']
        p=docker(args,45)
        if p.returncode:raise RuntimeError('Candidate container launch failed: '+p.stderr.decode(errors='replace')[:1000])
    def act(self,text,turn):
        try:calls=parse_calls(text)
        except ValueError as error:
            result='Invalid native tool call: '+str(error)+'. Use the tool syntax in the system message.'
            self.observations.append(result);return result,False,None
        results=[]
        for ordinal,call in enumerate(calls):
            name,args=call['name'],call['arguments']
            if name=='finish':
                observation=json.dumps(results+[{'tool':'finish','summary':args['summary']}])
                return observation,True,'finished'
            if name=='write_file':
                requested=PurePosixPath(args['path'])
                if not requested.is_absolute() or not requested.is_relative_to('/app') or '..' in requested.parts:
                    observation={'tool':name,'error':'write_file path must be under /app'}
                else:
                    target=self.app/str(requested.relative_to('/app'))
                    # Resolve traversal/symlink escapes into the task volume before host write.
                    resolved=target.resolve()
                    if not resolved.is_relative_to(self.app.resolve()):
                        observation={'tool':name,'error':'write_file path leaves /app'}
                    else:
                        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(args['content'])
                        observation={'tool':name,'path':args['path'],'bytes':target.stat().st_size,'sha256':sha(target)}
                results.append(observation);continue
            limit=180
            if config().get('max_episode_seconds'):
                left=config()['max_episode_seconds']-(time.monotonic()-self.started)
                if left<=0:return 'Episode time exhausted.',True,'episode_time_limit'
                limit=min(limit,max(1,int(left)))
            p=docker(['exec',self.name,'timeout','--signal=TERM','--kill-after=5',str(limit),
                '/bin/sh','-lc','cd /app\n'+args['command']],limit+20)
            (self.folder/f'tool-{turn:03d}-{ordinal:02d}.stdout').write_bytes(p.stdout)
            (self.folder/f'tool-{turn:03d}-{ordinal:02d}.stderr').write_bytes(p.stderr)
            save(self.folder/f'tool-{turn:03d}-{ordinal:02d}.receipt.json',
                {'tool':'bash','exit_code':p.returncode,'stdout_bytes':len(p.stdout),
                 'stderr_bytes':len(p.stderr),'at':time.time()})
            if b'Error response from daemon' in p.stderr:raise RuntimeError('Docker infrastructure failure; preserve actual attempt')
            results.append({'tool':'bash','exit_code':p.returncode,'stdout':p.stdout.decode(errors='replace'),
                'stderr':p.stderr.decode(errors='replace'),
                'python_traceback_observed':b'Traceback (most recent call last)' in p.stderr,
                'note':'A Blender exit_code=0 can still contain a Python traceback; inspect stderr.'})
        observation=json.dumps(results,ensure_ascii=False);self.observations.append(observation)
        return observation,False,None
    def finish(self,reason,output_tokens):
        file=self.app/'result.blend';exists=file.is_file()
        header=file.open('rb').read(12) if exists else b''
        valid=header.startswith(b'BLENDER') and file.stat().st_size>12
        result={'status':'file_present_unverified' if valid else 'invalid_submission_header' if exists else 'missing_submission',
            'saved':valid,'file_exists':exists,'blend_header_valid':valid,'scene_openability':'not yet verified',
            'termination_reason':reason,'source_finish_reason':reason,'source_output_tokens':output_tokens,
            'seconds':time.monotonic()-self.started,'tool_observations':len(self.observations),
            'error':'Saved file is not a Blender scene' if exists and not valid else 'No saved result.blend' if not exists else '',
            'last_tool_observation':self.observations[-1] if self.observations else '',
            'saved_scene_sha256':sha(file) if exists else None}
        if hasattr(self,'recovery_receipt'):
            result['partial_source_recovery']=self.recovery_receipt
            result['elapsed_time_source']='Original persistent task container start; includes outage'
        save(self.folder/'execution.json',result)
        self.close();return result
    def close(self):
        if self.closed:return
        try:
            p=docker(['rm','-f',self.name],30)
            self.closed=p.returncode==0
            if not self.closed:raise RuntimeError('Container cleanup returned nonzero')
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            # Cleanup cannot discard a completed sampled attempt or fabricate its reward.
            save(self.folder/'cleanup_pending.json',{'container':self.name,'at':time.time(),
                'error':type(error).__name__,'execution_evidence_preserved':True,'retry_generation':False})
