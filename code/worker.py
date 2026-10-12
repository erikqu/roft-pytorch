"""Serialized single-H100 paper-derived online updates, then fresh paired evaluation."""
import fcntl
import os
import random
import signal
import subprocess
import sys
import traceback
from common import *
from token_budget import select_fresh


def command(script,*args,training=True):
    window=load(RUN/'work_window.json');deadline=window['deadline_unix']-(config()['evaluation_reserve_seconds'] if training else 0)
    remaining=deadline-time.time()
    if remaining<=0:raise TimeoutError('Authorized new-run work window exhausted')
    env=dict(os.environ)
    if script in ('train.py','qualification_math.py','long_sequence_qualification.py'):
        env['PYTORCH_ALLOC_CONF']='expandable_segments:True'
    proc=subprocess.Popen([sys.executable,'-u',str(CODE/script),*map(str,args)],start_new_session=True,env=env)
    try:
        proc.wait(timeout=remaining)
        if proc.returncode:raise subprocess.CalledProcessError(proc.returncode,script)
    except BaseException:
        if proc.poll() is None:
            os.killpg(proc.pid,signal.SIGTERM)
            try:proc.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
        raise


def sample(jobs,checkpoint=None,grading=False,training=True):
    path=RUN/'jobs'/(jobs[0]['id']+'.json');save(path,jobs)
    args=['--checkpoint',checkpoint] if checkpoint else []
    command('generate.py',path,*args,*(['--grade'] if grading else []),training=training)
    return [load(RUN/'generations'/(j['id']+'.json')) for j in jobs]


class TaskOrder:
    def __init__(self,arm):
        self.rng=random.Random(SEED+(0 if arm=='roft' else 100000));self.tasks=[];self.epoch=0
    def next(self):
        if not self.tasks:
            self.tasks=list(config()['train']);self.rng.shuffle(self.tasks);self.epoch+=1
        return self.tasks.pop()


def train_arm(arm):
    from transformers import AutoProcessor
    from reflection_targets import accepted_target
    from objectives import group_advantages
    cfg=config();order=TaskOrder(arm);previous=None;cumulative=0;step=0;degenerate=0;consumed=set()
    if list((RUN/'train'/arm).glob('checkpoint-*')):
        from online_state import recover
        audits,cumulative,last,consumed=recover(RUN,cfg['total_supervised_tokens_per_arm'],arm)
        if arm!='roft':raise RuntimeError('GRPO sampler recovery needs a separate audited path')
        step=len(audits);previous=str(last) if last else None
        chain=[]
        for prior in audits:
            if prior['step']!=len(chain)+1:raise RuntimeError('Recovered update log is noncontiguous')
            chain.append(json.dumps(prior,allow_nan=False))
        log=RUN/'train'/arm/'updates.jsonl'
        if log.exists():
            existing=[json.loads(line) for line in log.read_text().splitlines()]
            if [r['step'] for r in existing]!=list(range(1,len(existing)+1)):
                raise RuntimeError('Existing update log contains duplicates or gaps')
            if len(existing)>len(audits):raise RuntimeError('Update log exceeds audited saved chain')
            for row,prior in zip(existing,audits):
                if row['target_ids']!=prior['target_ids'] or row['resume_sha256']!=prior['resume_sha256']:
                    raise RuntimeError('Existing update log differs from saved checkpoint chain')
        log.write_text('\n'.join(chain)+'\n')
        for prior in audits:
            jobs=load(RUN/'jobs'/f'roft-{prior["step"]:03d}-attempt-000.json')
            if len(jobs)!=cfg['source_attempts_per_update']:raise RuntimeError('Recovered source membership changed')
            for job in jobs:
                if order.next()!=job['task']:raise RuntimeError('Recovered task sampler order differs')
            records=load(RUN/'jobs'/f'roft-{prior["step"]:03d}-targets.json')
            lengths=sorted(len(r['token_ids']) for r in records)
            low=lengths[len(lengths)//2]<100 or sum(n<=10 for n in lengths)/len(lengths)>.1
            degenerate=degenerate+1 if low else 0
        save(RUN/'train'/arm/'recovery_state.json',{'at':time.time(),'completed_updates':step,
            'completed_backward_tokens':cumulative,'consumed_target_ids':sorted(consumed),
            'sampler_order_verified':True,'source_checkpoint':previous,'backward_replayed':False})
    tokenizer=AutoProcessor.from_pretrained(str(MODEL)).tokenizer
    while cumulative<cfg['total_supervised_tokens_per_arm']:
        step+=1;start=time.monotonic();records=[];attempt_ids=[];excluded=[]
        if arm=='roft':
            jobs=[]
            for i in range(cfg['source_attempts_per_update']):
                jobs.append({'id':f'roft-{step:03d}-attempt-{i:03d}','task':order.next(),
                    'seed':SEED+step*10000+i,'phase':'training','arm':arm,'step':step,'policy_version':step-1})
            attempts=sample(jobs,previous,True)
            attempts.sort(key=lambda r:r['seconds_since_generation_start'])
            attempt_ids=[r['id'] for r in attempts]
            reflection_jobs=[]
            for i,attempt in enumerate(attempts):
                if not (RUN/'episodes'/attempt['id']/'result.json').exists():
                    excluded.append({'attempt_id':attempt['id'],'reason':'actual_verifier_feedback_unavailable'});continue
                for repeat in range(cfg['reflections_per_attempt']):
                    reflection_jobs.append({'id':attempt['id']+f'-reflection-{repeat}','task':attempt['task'],
                        'seed':SEED+10000000+step*10000+i*4+repeat,'phase':'reflection','arm':arm,
                        'step':step,'reflection':True,'attempt_id':attempt['id'],'policy_version':step-1})
            raw=sample(reflection_jobs,previous)
            raw.sort(key=lambda r:r['seconds_since_generation_start'])
            for row in raw:
                accepted,reason=accepted_target(row,tokenizer)
                if reason:excluded.append({'id':row['id'],'reason':reason})
                elif len(row['prompt_token_ids'])+len(accepted['token_ids'])>cfg['max_training_tokens']:
                    excluded.append({'id':row['id'],'reason':'training_sequence_bound'})
                else:records.append(accepted)
            lengths=sorted(len(r['token_ids']) for r in records)
            if not lengths:raise RuntimeError('No complete reflections; do not replace with action targets')
            low=lengths[len(lengths)//2]<100 or sum(n<=10 for n in lengths)/len(lengths)>.1
            degenerate=degenerate+1 if low else 0
            if degenerate>=2:raise RuntimeError('Paper reflection degeneracy stop triggered')
        else:
            groups=[];ordinal=0
            while len(groups)<32:
                name=order.next();group_id=f'grpo-{step:03d}-group-{ordinal:03d}';ordinal+=1
                jobs=[{'id':group_id+f'-attempt-{i}','task':name,'seed':SEED+20000000+step*100000+ordinal*8+i,
                    'phase':'training','arm':arm,'step':step,'group_id':group_id,'policy_version':step-1} for i in range(8)]
                attempts=sample(jobs,previous,True);attempt_ids.extend(r['id'] for r in attempts)
                grades=[load(RUN/'episodes'/r['id']/'result.json') if (RUN/'episodes'/r['id']/'result.json').exists() else None for r in attempts]
                if any(g is None for g in grades):
                    raise RuntimeError('GRPO group missing actual verifier reward; no zero substitution')
                rewards=[g['reward'] for g in grades]
                if len(set(rewards))==1:
                    excluded.append({'group_id':group_id,'reason':'identical_actual_rewards','rewards':rewards});continue
                advantages=group_advantages(rewards,[group_id]*8).tolist()
                groups.append((attempts,advantages))
                status('grpo_dynamic_groups',accepted_groups=len(groups),attempted_groups=ordinal,step=step)
            for attempts,advantages in groups:
                for attempt,adv in zip(attempts,advantages):
                    length=sum(len(t['token_ids']) for t in attempt['turns'])
                    if not length:raise RuntimeError('Empty agentic GRPO trajectory')
                    for t,turn in enumerate(attempt['turns']):
                        records.append(dict(turn,id=attempt['id']+f'-turn-{t:03d}',task=attempt['task'],
                            group_id=attempt['group_id'],attempt_id=attempt['id'],advantage=adv,
                            trajectory_weight=1/(len(groups)*8*length),trajectory_generated_tokens=length))
        quota=cfg['tokens_per_update'][step-1] if step<=len(cfg['tokens_per_update']) else cfg['total_supervised_tokens_per_arm']-cumulative
        targets,selection=select_fresh(records,min(quota,cfg['total_supervised_tokens_per_arm']-cumulative),step)
        if set(r['id'] for r in targets)&consumed:raise RuntimeError('Old target replay forbidden')
        consumed.update(r['id'] for r in targets)
        save(RUN/'train'/arm/f'selection-{step:03d}.json',dict(selection,excluded=excluded,attempt_ids=attempt_ids,
            policy_version=step-1,replayed_targets=0,source_checkpoint=previous))
        path=RUN/'jobs'/f'{arm}-{step:03d}-targets.json';save(path,targets)
        extra=['--previous',previous] if previous else []
        command('train.py','update','--arm',arm,'--step',step,'--records',path,*extra)
        previous=str(RUN/'train'/arm/f'checkpoint-{step:03d}')
        audit=load(Path(previous)/'audit.json');cumulative+=audit['supervised_tokens']
        audit.update(cumulative_backward_tokens=cumulative,cycle_seconds=time.monotonic()-start,
            selection_sha256=sha(RUN/'train'/arm/f'selection-{step:03d}.json'))
        save(Path(previous)/'audit.json',audit);append(RUN/'train'/arm/'updates.jsonl',audit)
        status('verified_full_parameter_update',arm=arm,step=step,backward_tokens=cumulative,
            target=cfg['total_supervised_tokens_per_arm'],actual_changed_tensors=audit['actual_changed_tensors'])
    if cumulative!=cfg['total_supervised_tokens_per_arm']:raise RuntimeError('Exact final backward budget mismatch')
    save(RUN/'train'/arm/'complete.json',{'arm':arm,'checkpoint':previous,'backward_tokens':cumulative,'updates':step})
    return previous


def evaluate(arm,checkpoint):
    cfg=config();jobs=[]
    for split,names,seeds in [('transfer',cfg['transfer'],cfg['transfer_seeds']),
                              ('within_task',cfg['within_task_tasks'],cfg['within_task_seeds'])]:
        for name in names:
            for seed in seeds:
                jobs.append({'id':f'evaluation-{arm}-{name}-seed-{seed}','task':name,'seed':seed,
                    'phase':'evaluation','split':split,'arm':arm})
    sample(jobs,checkpoint,True,training=False)
    save(RUN/'evaluation'/f'{arm}-complete.json',{'arm':arm,'checkpoint':checkpoint,
        'provenance':'fresh inference','jobs':[j['id'] for j in jobs],'protocol_sha256':sha(RUN/'protocol.json')})


def main():
    def stop(signum,frame):raise InterruptedError('Run supervisor requested bounded work stop; node retained')
    signal.signal(signal.SIGTERM,stop)
    with (RUN/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        validate_prepared();save(RUN/'worker_pid.json',{'pid':os.getpid(),'started':time.time()})
        try:
            if not (RUN/'qualification/native_gradient_equivalence.json').exists():command('qualification_math.py')
            if not (RUN/'qualification/complete.json').exists():
                qualify_native()
            checkpoints={}
            for arm in config()['arms']:checkpoints[arm]=train_arm(arm)
            # Evaluate the requested ROFT arm before its baseline; no GRPO launch dependency.
            for arm in config()['arms']+['base']:evaluate(arm,checkpoints.get(arm))
            command('report.py',training=False)
            save(RUN/'worker_complete.json',{'at':time.time(),'token_equal_training':True,
                'fresh_evaluation_all_arms':True,'keep_node_running':True,'automatic_shutdown':False})
        except BaseException as error:
            save(RUN/'failure.json',{'at':time.time(),'type':type(error).__name__,
                'error':str(error),'traceback':traceback.format_exc()[-10000:],'keep_node_running':True})
            raise


def qualify_native():
    command('long_sequence_qualification.py')
    jobs=[{'id':'qualification-native-agent','task':config()['train'][0],'seed':SEED,
        'phase':'qualification','arm':'base','qualification_agent':True}]
    sample(jobs,None,True)
    row=load(RUN/'generations/qualification-native-agent.json')
    if len(row['turns'])<2 or not any('observation' in t for t in row['turns']):
        raise RuntimeError('Native tool qualification needs actual tool execution')
    actual=load(RUN/'episodes'/row['id']/'result.json')
    if not actual['execution']['saved'] or actual.get('verifier_failure'):
        raise RuntimeError('Native qualification must produce a verifiably readable Blender scene')
    reflection=[{'id':'qualification-native-reflection','task':row['task'],'seed':SEED+1,
        'phase':'qualification','reflection':True,'attempt_id':row['id']}]
    sample(reflection,None)
    from transformers import AutoProcessor
    from reflection_targets import accepted_target
    from train import process
    processor=AutoProcessor.from_pretrained(str(MODEL))
    raw=load(RUN/'generations/qualification-native-reflection.json');target,error=accepted_target(raw,processor.tokenizer)
    if error:raise RuntimeError('Qualification reflection rejected: '+error)
    batch,labels,n=process(processor,target)
    save(RUN/'qualification/complete.json',{'passed':True,'actual_native_tool_execution':True,
        'actual_readable_blender_scene':True,'fresh_thinking_reflection':True,
        'sampled_prefix_exact':True,'all_target_thinking_eot_supervised':True,'reflection_target_tokens':n})
if __name__=='__main__':main()
