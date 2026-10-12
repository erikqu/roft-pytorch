"""Persistent judging, backup and bounded work supervisor. Never stops the node."""
import fcntl
import os
import shlex
import signal
import subprocess
import time
from common import *
from cloud import ssh,owned,REMOTE,PYTHON,NODE_PYTHON,utc
from prepare import seal


def remote(command,timeout=45):
    return subprocess.run(ssh()+[command],capture_output=True,text=True,check=True,timeout=timeout).stdout


def sync(metadata=True):
    target=RUN/'remote';target.mkdir(exist_ok=True)
    args=['rsync','-az','--timeout=30','--exclude=model/','--exclude=tasks/','--exclude=source_harness/',
        '--exclude=public_tasks/','--exclude=.verified-resume-*/',
        '--exclude=.audit-state-reader-*/','--exclude=.recovery-reader-*/']
    if metadata:args+=['--exclude=*.safetensors','--exclude=resume.pt.gz*','--exclude=app/']
    command=args+['-e',shlex.join(ssh()[:-1]),ssh()[-1]+':'+REMOTE+'/run/',str(target)+'/']
    p=subprocess.run(command,capture_output=True,text=True,timeout=120)
    if p.returncode:raise RuntimeError('Backup transport failed: '+p.stderr[:400])
    subprocess.run(['rsync','-az','--timeout=30','-e',shlex.join(ssh()[:-1]),ssh()[-1]+':'+REMOTE+'/worker.log',
        str(RUN/'logs/worker.log')],capture_output=True,timeout=45,check=True)

def backup_final_states():
    # Preserve final full weights/FP32 optimizer/RNG states, then independently hash
    # verify against the remote model's update audit. No derived/no-op results.
    from online_state import recover
    for arm in config()['arms']:
        complete=load(RUN/'remote/train'/arm/'complete.json')
        folder=Path(complete['checkpoint']);local=RUN/'remote/train'/arm/folder.name
        size_code="from pathlib import Path;p=Path("+repr(str(folder))+");print(sum(x.stat().st_size for x in p.iterdir() if x.is_file()))"
        size=int(remote('python3 -c '+shlex.quote(size_code)).strip())
        if __import__('shutil').disk_usage(RUN).free<size+5*1024**3:
            raise RuntimeError('Insufficient host space for complete final state backup')
        subprocess.run(['rsync','-a','--timeout=60','--exclude=.verified-resume-*/',
            '--exclude=.audit-state-reader-*/','--exclude=.recovery-reader-*/','-e',shlex.join(ssh()[:-1]),
            ssh()[-1]+':'+str(folder)+'/',str(local)+'/'],check=True,timeout=3600)
        audit=load(local/'audit.json')
        if sha(local/'resume.pt.gz')!=audit['resume_sha256']:raise RuntimeError('Optimizer/RNG backup hash failed')
        for name,expected in load(local/'weight_manifest.json').items():
            if sha(local/name)!=expected:raise RuntimeError('Full model backup hash failed')
        save(RUN/f'{arm}-final-state-backup.json',{'at':time.time(),'verified':True,
            'checkpoint':str(local),'weight_manifest_sha256':sha(local/'weight_manifest.json'),
            'resume_sha256':audit['resume_sha256'],'full_weights_optimizer_rng':True})


def main():
    (RUN/'logs').mkdir(exist_ok=True)
    with (RUN/'controller.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        save(RUN/'controller_pid.json',{'pid':os.getpid(),'at':time.time()})
        n=owned()
        if not n or n['status']!='active':raise RuntimeError('Retained node unavailable')
        # v5 owns its new tool-interface qualification; the unchanged native
        # full-model math was independently verified in v4 and will be audited
        # again on the first admitted fresh update.
        seal();validate_prepared()
        transport=shlex.join(ssh()[:-1]);destination=ssh()[-1]+':'+REMOTE
        remote('mkdir -p '+REMOTE+'/code '+REMOTE+'/run '+REMOTE+'/model')
        check = 'from pathlib import Path;root=Path(' + repr(REMOTE) + ');required=[root/"run"/n for n in ["tasks","public_tasks","source_harness"]]+[root/"model"];missing=[str(p) for p in required if not p.is_dir()];assert not missing, missing'
        remote('python3 -c ' + shlex.quote(check))
        subprocess.run(['rsync','-a','--exclude=artifacts','--exclude=__pycache__','-e',transport,str(CODE)+'/',destination+'/code/'],check=True)
        for name in ['manifest.json','protocol.json','prompts.json','dataset_identity.json','source_dataset_manifest.json']:
            subprocess.run(['rsync','-a','-e',transport,str(RUN/name),destination+'/run/'+name],check=True)
        start=time.time()
        window=load(RUN/'work_window.json') if (RUN/'work_window.json').exists() else {'started_unix':start,'deadline_unix':start+config()['hard_work_seconds'],
            'authorization':'User explicitly requested fixes and try again on the retained node',
            'allocation_launch_unix':load(RUN/'allocation.json')['launch_unix'],
            'keep_node_running':True,'automatic_shutdown':False,'rate_usd_hour':load(RUN/'allocation.json')['rate_usd_hour']}
        if not (RUN/'work_window.json').exists():save(RUN/'work_window.json',window)
        if time.time()>=window['deadline_unix']:raise RuntimeError('Original work window expired; never reset')
        subprocess.run(['rsync','-a','-e',transport,str(RUN/'work_window.json'),destination+'/run/'],check=True)
        bridge=tunnel=None
        try:
            bridge=subprocess.Popen([PYTHON,'-u',str(CODE/'judge_bridge.py')],stdout=(RUN/'logs/judge_bridge.log').open('ab'),stderr=subprocess.STDOUT)
            port=config()['judge_port']
            tunnel=subprocess.Popen(ssh()[:-1]+['-o','ExitOnForwardFailure=yes','-N','-R',f'{port}:127.0.0.1:{port}',ssh()[-1]],
                stdout=subprocess.DEVNULL,stderr=(RUN/'logs/tunnel.log').open('ab'))
            time.sleep(2)
            if bridge.poll() is not None or tunnel.poll() is not None:raise RuntimeError('Judge transport not live')
            env={'ROFT_RUN':REMOTE+'/run','ROFT_MODEL':REMOTE+'/model','TOKENIZERS_PARALLELISM':'false',
                'LD_LIBRARY_PATH':os.environ.get('ROFT_NODE_LD_LIBRARY_PATH',''),
                'PATH':os.environ.get('ROFT_NODE_PATH',str(Path(NODE_PYTHON).parent)+':/usr/local/cuda/bin:/usr/bin:/bin')}
            launch="import subprocess,os;env=dict(os.environ);env.update("+repr(env)+");log=open('"+REMOTE+"/worker.log','ab');p=subprocess.Popen(["+repr(NODE_PYTHON)+",'-u','"+REMOTE+"/code/worker.py'],env=env,stdout=log,stderr=log,start_new_session=True);print(p.pid)"
            pid=int(remote('python3 -c '+shlex.quote(launch)).strip())
            save(RUN/'worker_launch.json',{'pid':pid,'at':time.time(),'instance_id':n['id'],'bridge_pid':bridge.pid,'tunnel_pid':tunnel.pid})
            while time.time()<window['deadline_unix']:
                try:sync()
                except (subprocess.TimeoutExpired,RuntimeError,subprocess.CalledProcessError) as error:
                    append(RUN/'backup_failures.jsonl',{'at':time.time(),'error':str(error)[:400],'worker_stopped':False})
                if (RUN/'remote/worker_complete.json').exists():
                    backup_final_states()
                    from backup_raw import main as backup_raw
                    backup_raw()
                    break
                if (RUN/'remote/failure.json').exists():raise RuntimeError('Remote worker failed; preserved failure.json')
                if bridge.poll() is not None or tunnel.poll() is not None:raise RuntimeError('Judge transport exited')
                time.sleep(20)
            else:raise TimeoutError('New run work window exhausted; node retained')
        except BaseException as error:
            save(RUN/'controller_failure.json',{'at':time.time(),'type':type(error).__name__,'error':str(error),'keep_node_running':True})
            raise
        finally:
            # Deadline/failure stops only this run's verified process group.
            if (RUN/'worker_launch.json').exists() and not (RUN/'remote/worker_complete.json').exists():
                worker_pid=load(RUN/'worker_launch.json')['pid']
                stop="import os,signal;from pathlib import Path;p=Path('/proc/"+str(worker_pid)+"/cmdline');"+\
                    "os.killpg("+str(worker_pid)+",signal.SIGTERM) if p.exists() and "+repr((REMOTE+'/code/worker.py').encode())+" in p.read_bytes() else None"
                try:remote('python3 -c '+shlex.quote(stop))
                except Exception as e:append(RUN/'cleanup_failures.jsonl',{'error':str(e)[:400]})
                time.sleep(2)
            try:sync()
            except Exception as e:append(RUN/'backup_failures.jsonl',{'at':time.time(),'error':str(e)[:400]})
            if tunnel:tunnel.terminate()
            if bridge:bridge.terminate()
            n=owned();save(RUN/'node_kept_running.json',{'at':time.time(),'state':n['status'] if n else 'missing',
                'automatic_shutdown':False,'reason':'Explicit user instruction','still_billable':bool(n)})
if __name__=='__main__':main()
