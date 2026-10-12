"""Run candidates in the copied gym; keep host judging on the local machine."""
import base64
import concurrent.futures
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
import urllib.request

from common import RUN, load, save, sha, config
import ast,re
def script(text):
    blocks=re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```",text,re.S)
    if len(blocks)>1:raise ValueError("Expected exactly one Python program")
    code=blocks[0].strip() if blocks else text.strip()
    ast.parse(code)
    if not code:raise ValueError("Empty Python program")
    return code+"\n"

DOCKER = ['docker'] if os.environ.get('ROFT_LOCAL_DOCKER')=='1' else ['sudo','-n','docker']
IMAGE = os.environ.get('ROFT_BLENDER_IMAGE', 'arle-blender-pilot:4.5.12')

def docker(args, timeout):
    process = subprocess.run(DOCKER+args, capture_output=True, timeout=timeout)
    return process

def grade(row):
    """Resume evidence/judging without generating a new candidate."""
    episode = RUN/'episodes'/row['id']
    if (episode/'result.json').exists():
        result = load(episode/'result.json')
        if result['response_sha256'] != row['response_sha256']: raise RuntimeError('Candidate changed on resume')
        return result
    unavailable = episode/'grading_unavailable.json'
    if unavailable.exists():
        from grading_state import validate_unavailable
        validate_unavailable(row, load(unavailable),
                             (episode/'grader/verifier/judge-request.json').read_bytes())
        # A refusal is missing feedback, never a numerical reward or a candidate reroll.
        return None
    episode.mkdir(parents=True, exist_ok=True)
    task = RUN/'tasks'/row['task']; app = episode/'app'; logs = episode/'grader'
    if not app.exists(): shutil.copytree(task/'environment/workspace', app)
    logs.mkdir(exist_ok=True)
    save(episode/'response.json', row)
    execution = episode/'execution.json'
    if not execution.exists():
        try: code = script(row['final_text'])
        except (ValueError, SyntaxError) as error:
            import traceback
            save(execution, {'saved':False,'status':'candidate_error','error_type':type(error).__name__,
                'error':str(error),'traceback':traceback.format_exc()[-12000:],
                'source_finish_reason':row['finish_reason'],'source_output_tokens':row['output_tokens']})
        else:
            candidate = episode/'candidate.py';candidate.write_text(code)
            name = 'roft-blender-exec-'+row['id']
            args=['run','--rm','--init','--name',name,'--network','none','--cpus','4','--memory','8g',
                  '--pids-limit','256','--cap-drop','ALL','--security-opt','no-new-privileges','--shm-size','256m',
                  '-e','OMP_NUM_THREADS=4','-e','OPENBLAS_NUM_THREADS=1',
                  '-v',f'{app}:/app','-v',f'{candidate}:/candidate.py:ro',
                  '-v',f'{RUN / "source_harness/scripts/oneshot_runner.py"}:/oneshot_runner.py:ro',
                  '--entrypoint','timeout',IMAGE,'--signal=TERM','--kill-after=5','180',
                  'blender','--background','--factory-startup','--threads','4','--python','/oneshot_runner.py']
            try: process=docker(args,210)
            except subprocess.TimeoutExpired:
                docker(['rm','-f',name],30)
                save(execution,{'saved':False,'status':'candidate_timeout'})
            else:
                (episode/'execution.stdout').write_bytes(process.stdout)
                (episode/'execution.stderr').write_bytes(process.stderr)
                if process.returncode in [125,126,127]: raise RuntimeError('Candidate container infrastructure error')
                result=load(app/'oneshot-execution.json') if (app/'oneshot-execution.json').exists() else {'saved':False,'status':'candidate_error'}
                result['return_code']=process.returncode;save(execution,result)
    name='roft-blender-grade-'+row['id']
    mounts=['-v',f'{app}:/app:ro','-v',f'{task / "tests"}:/tests:ro','-v',f'{logs}:/logs',
            '-v',f'{RUN / "source_harness/scripts/grade.py"}:/grade.py:ro',
            '-v',f'{Path(__file__).parent / "verifier_blender_launcher.py"}:/opt/roft-verifier-bin/blender:ro',
            '-v',f'{Path(__file__).parent / "verifier_surface_guard.py"}:/opt/roft-verifier-guard.py:ro']
    sealed=['run','--rm','--name',name,'--network','none','--cpus','8','--memory','64g',
            '--cap-drop','ALL','--security-opt','no-new-privileges','-e','OMP_NUM_THREADS=4',
            '-e','OPENBLAS_NUM_THREADS=1',
            '-e','PATH=/opt/roft-verifier-bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
            *mounts,IMAGE,'python3','/grade.py']
    if not (logs/'verifier/evaluation.json').exists():
        try: p=docker([*sealed,'/tests','/app','/logs'],900)
        except subprocess.TimeoutExpired:
            docker(['rm','-f',name],30);raise
        (logs/'evidence.stdout').write_bytes(p.stdout);(logs/'evidence.stderr').write_bytes(p.stderr)
        if p.returncode: raise RuntimeError('Sealed verifier evidence failed; retain evidence logs')
    request_path=logs/'verifier/judge-request.json'
    if request_path.exists() and not (logs/'verifier/judge-attestation.json').exists():
        raw=request_path.read_bytes(); request=json.loads(raw)
        save(logs/'verifier/judge-execution-profile.json',{
             'provider':'azure','model':'gpt-6-luna','reasoning_effort':'xhigh',
             'max_output_tokens':16384,'passes':request['passes'],
             'request_sha256':__import__('hashlib').sha256(raw).hexdigest(),
             'host_credentials_forwarded':False,'budget_scope':config()['run_identity']})
        evidence={}
        for item in request['evidence']:
            filename=item['filename']
            if Path(filename).name!=filename:raise RuntimeError('Invalid evidence filename')
            evidence[filename]=base64.b64encode((logs/'verifier/judge-evidence'/filename).read_bytes()).decode()
        body=json.dumps({'request_bytes':base64.b64encode(raw).decode(),'evidence':evidence,'budget_class':'evaluation' if row.get('phase')=='evaluation' else 'training'}).encode()
        req=urllib.request.Request('http://127.0.0.1:'+str(config()['judge_port'])+'/judge',data=body,headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=900) as response: attestation=json.load(response)
        save(logs/'verifier/judge-attestation.json',attestation)
    p=docker([*sealed,'--finalize-shared','/tests','/app','/logs'],150)
    (logs/'finalize.stdout').write_bytes(p.stdout);(logs/'finalize.stderr').write_bytes(p.stderr)
    if p.returncode:raise RuntimeError('Sealed finalization failed')
    reward=load(logs/'pilot-reward.json');technical=load(logs/'verifier/technical.json')
    evaluation=load(logs/'verifier/evaluation.json')
    value=reward['reward'];assert isinstance(value,(int,float)) and math.isfinite(value) and 0<=value<=1
    # Combined reward is exactly the released finalization result, never reconstructed.
    result=dict(row,reward=value,geometry=technical['deterministic'],visual=evaluation.get('llm_visual'),execution=load(execution),
                applied_caps=evaluation.get('applied_caps',[]),graded_at=time.time(),
                reward_file_sha256=sha(logs/'verifier/reward.json'),episode_path=str(episode))
    # Raw binary verdict is unavailable in this continuous geometry benchmark.
    result['raw_verdict']='unknown'
    failure_path=logs/'verifier/candidate-failure.json'
    result['verifier_failure']=load(failure_path) if failure_path.exists() else None
    result['validation_status']='invalid_submission' if result['verifier_failure'] else 'verifier_completed'
    if result['verifier_failure']:
        result['execution']=dict(result['execution'],status='verifier_rejected_submission',saved=False,verifier_failure=result['verifier_failure'])
    result['candidate_feedback_images']=[]
    for ordinal in (1,2):
        source=logs/'verifier/judge-evidence'/f'candidate-{ordinal:02d}.png'
        if source.exists():
            target=episode/f'candidate-feedback-{ordinal:02d}.png'
            shutil.copy2(source,target)
            result['candidate_feedback_images'].append(str(target.relative_to(RUN)))
    result['candidate_feedback_image_sha256']=[sha(RUN/p) for p in result['candidate_feedback_images']]
    save(episode/'result.json',result)
    print('GRADED',row['id'],row['task'],value,flush=True)
    return result

def batch(rows, workers=2):
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(grade,r) for r in rows]
        return [f.result() for f in futures]
