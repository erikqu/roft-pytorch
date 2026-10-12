"""Fresh full-weight multimodal sampling and genuine multi-turn tool execution."""
import argparse
import concurrent.futures
from pathlib import Path
import time
from common import *
from reflection_targets import bounded,trajectory
from tool_protocol import TOOLS,history_message
from context_bound import ascii_token_byte_bound,proven_ascii_context_overflow

def reflection_messages(row,attempt,grade):
    task=dict(row,text=bounded(row['text'],4000))
    feedback=bounded(reflection_feedback(grade),3000)
    # The task context must be the exact instruction used in the attempted solution.
    task=dict(task,text=bounded(attempt.get('task_description',row['text']),4000))
    result=[{'role':'system','content':reflection_instruction()}]+messages(task)[1:]+[
        {'role':'user','content':[{'type':'text','text':'Recorded interaction:\n'+trajectory(attempt)+
            '\nFinal files/patch summary:\n'+bounded(attempt.get('file_summary',''),2000)+'\n'+feedback}]}]
    for name in grade.get('candidate_feedback_images',[]):
        result[-1]['content'].append({'type':'image','url':str(RUN/name)})
    return result

def job_messages(job):
    if 'sampling_messages' in job:return job['sampling_messages']
    row=prompt_rows()[job['task']]
    if job.get('qualification_agent'):
        row=dict(row,text='Test the agent environment: inspect /app/cameras.json, write /app/probe.py that imports bpy, creates a cube, and calls bpy.ops.wm.save_as_mainfile(filepath=\"/app/result.blend\"). Run Blender with --background --python-exit-code 1 --python /app/probe.py. Check the saved file and finish. Use the provided native tools; keep this interface probe brief.')
    if job.get('reflection'):
        return reflection_messages(row,load(RUN/'generations'/(job['attempt_id']+'.json')),
            load(RUN/'episodes'/job['attempt_id']/'result.json'))
    if job.get('qualification'):row=dict(row,text='Describe the visible reference shapes briefly.')
    return messages(row)

def checkpoint_hash(checkpoint):
    return digest(load(Path(checkpoint)/'weight_manifest.json')) if checkpoint else None

def generate(jobs,checkpoint=None,grading=False):
    import os
    os.environ['VLLM_WORKER_MULTIPROC_METHOD']='spawn'
    from vllm import LLM,SamplingParams
    from transformers import AutoProcessor
    from PIL import Image
    from agent_env import AgentEnvironment
    cfg=config();processor=AutoProcessor.from_pretrained(str(MODEL));identity=checkpoint_hash(checkpoint)
    token_byte_bound=ascii_token_byte_bound(Path(MODEL)/'tokenizer.json')
    pending=[];results={}
    for job in jobs:
        path=RUN/'generations'/(job['id']+'.json')
        if path.exists():
            r=load(path)
            if r['checkpoint_sha256']!=identity or r['seed']!=job['seed']:raise RuntimeError('Resume identity changed')
            results[job['id']]=r
        else:pending.append(job)
    if grading and results:
        from grade import batch
        batch(list(results.values()),cfg['grade_workers'])
    if not pending:return [results[j['id']] for j in jobs]
    status('loading_inference',jobs=len(pending),checkpoint=checkpoint)
    llm=LLM(model=str(checkpoint or MODEL),dtype='bfloat16',tensor_parallel_size=1,
        max_model_len=cfg['max_context_tokens'],max_num_seqs=cfg['max_num_seqs'],
        max_num_batched_tokens=cfg['max_num_batched_tokens'],gpu_memory_utilization=cfg['gpu_memory_utilization'],
        limit_mm_per_prompt={'image':cfg['max_images_per_prompt'],'video':0},enable_prefix_caching=True,
        compilation_config={'mode':0,'cudagraph_mode':'FULL_DECODE_ONLY'},generation_config='vllm',seed=SEED)
    pool=concurrent.futures.ThreadPoolExecutor(max_workers=max(cfg['grade_workers'],cfg['max_num_seqs']))
    live={};tool_futures={};grade_futures=[];iterator=iter(pending);started=time.monotonic()
    if results and any((RUN/'episodes'/j['id']/'partial_resume.json').exists() for j in pending):
        anchors=[(RUN/'generations'/(r['id']+'.json')).stat().st_mtime-r['seconds_since_generation_start']
            for r in results.values()]
        started=time.monotonic()-(time.time()-min(anchors))
        save(RUN/'partial_generation_time_anchor.json',{'at':time.time(),
            'original_generation_start_unix_from_completed_artifacts':min(anchors),
            'preserved_completion_order_across_recovery':True,'outage_included':True,
            'timestamp_source':'completed generation file mtime minus recorded generation elapsed time'})
    def enqueue(state):
        job=state['job'];turn=len(state['turns']);msg=state['messages']
        text=processor.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,
            enable_thinking=True,tools=[] if job.get('reflection') else TOOLS)
        image_names=list(prompt_rows()[job['task']]['images'])
        if job.get('reflection'):
            image_names+=load(RUN/'episodes'/job['attempt_id']/'result.json').get('candidate_feedback_images',[])
        state['images']=image_names
        bound=120 if job.get('qualification') else cfg['max_reflection_tokens'] if job.get('reflection') else min(
            cfg['max_response_tokens'],cfg['max_solver_total_tokens']-state['total_tokens'])
        packed=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',enable_thinking=True,tools=[] if job.get('reflection') else TOOLS)
        free_context=cfg['max_context_tokens']-packed['input_ids'].shape[-1]
        if free_context<=0:raise RuntimeError('No remaining context capacity; episode context cutoff must be recorded')
        bound=min(bound,free_context)
        params=SamplingParams(max_tokens=bound,temperature=cfg['reflection_temperature'] if job.get('reflection') else cfg['temperature'],
            top_p=cfg['top_p'],top_k=cfg['top_k'],seed=job['seed']+turn*1000003,logprobs=0)
        rid=job['id']+f'/turn-{turn:03d}';live[rid]=state
        llm.llm_engine.add_request(rid,{'prompt':text,'multi_modal_data':{'image':[
            Image.open(RUN/p).convert('RGB') for p in image_names]}},params)
    def add_job(job):
        state={'job':job,'messages':job_messages(job),'turns':[],'total_tokens':0}
        resume=RUN/'episodes'/job['id']/'partial_resume.json'
        if resume.exists():
            saved=load(resume)
            if saved['checkpoint_sha256']!=identity or saved['job']!=job:
                raise RuntimeError('Partial recovery checkpoint/job identity mismatch')
            state.update(messages=saved['messages'],turns=saved['turns'],total_tokens=saved['total_tokens'])
            state['images']=list(prompt_rows()[job['task']]['images'])
            state['env']=AgentEnvironment.resume(job,saved['recovery_evidence'])
            state['env'].observations=[t['observation'] for t in saved['turns'] if 'observation' in t]
            if saved.get('context_overflow_proof'):
                complete(state,'context_limit')
                try:add_job(next(iterator))
                except StopIteration:pass
                return
        elif not job.get('reflection') and not job.get('qualification'):state['env']=AgentEnvironment(job)
        enqueue(state)
    def complete(state,reason):
        job=state['job'];turns=state['turns'];last=turns[-1]
        token_ids=[t for r in turns for t in r['token_ids']]
        text='\n\n'.join(r['text'] for r in turns)
        record=dict(job,text=text,task_description=state['messages'][1]['content'][0]['text'],final_text=last['text'].split('</think>',1)[-1].strip(),
            token_ids=token_ids,output_tokens=len(token_ids),turns=turns,
            finish_reason=last['finish_reason'] if job.get('reflection') else ('stop' if reason=='finished' else 'length'),
            termination_reason=reason,checkpoint_sha256=identity,protocol_sha256=sha(RUN/'protocol.json'),
            image_sha256=[sha(RUN/p) for p in state['images']],response_sha256=digest({'turns':turns}),
            prompt_sha256=turns[0]['prompt_sha256'],backend='vllm',provenance='fresh inference',
            seconds_since_generation_start=time.monotonic()-started)
        if job.get('reflection') or job.get('qualification'):
            record.update(prompt_token_ids=last['prompt_token_ids'],sampled_raw_logprobs=last['sampled_raw_logprobs'],
                sampling_messages=last['sampling_messages'])
        else:
            record['execution']=state['env'].finish(reason,len(token_ids))
            scripts=[]
            for p in sorted(state['env'].app.rglob('*.py')):
                if p.is_file():scripts.append(str(p.relative_to(state['env'].app))+'\n'+p.read_text(errors='replace'))
            record['file_summary']=bounded('\n\n'.join(scripts),2000)
        save(RUN/'generations'/(job['id']+'.json'),record);results[job['id']]=record
        print('GENERATED',job['id'],len(token_ids),len(turns),reason,flush=True)
        if grading:
            from grade import grade
            grade_futures.append(pool.submit(grade,record))
    try:
        for _ in range(min(cfg['max_num_seqs'],len(pending))):
            try:add_job(next(iterator))
            except StopIteration:break
        active_jobs=min(cfg['max_num_seqs'],len(pending))
        while live or tool_futures:
            for output in llm.llm_engine.step() if live else []:
                if not output.finished:continue
                state=live.pop(output.request_id);job=state['job'];result=output.outputs[0]
                ids=list(result.token_ids)
                lp=[float(scores[t].logprob) for t,scores in zip(ids,result.logprobs)]
                if len(lp)!=len(ids):raise RuntimeError('Missing sampled raw log probabilities')
                turn={'id':output.request_id,'text':result.text,'token_ids':ids,
                    'sampled_raw_logprobs':lp,'prompt_token_ids':list(output.prompt_token_ids),
                    'sampling_messages':state['messages'].copy(),'finish_reason':result.finish_reason,
                    'checkpoint_sha256':identity,'seed':job['seed']+len(state['turns'])*1000003,
                    'prompt_sha256':digest({'ids':list(output.prompt_token_ids),'images':[sha(RUN/p) for p in state['images']]})}
                # Raw tokens persisted before parsing, tool execution or grading.
                save(RUN/'raw_turns'/(job['id']+f'-{len(state["turns"]):03d}.json'),turn)
                state['turns'].append(turn);state['total_tokens']+=len(ids)
                if job.get('reflection') or job.get('qualification'):
                    complete(state,result.finish_reason);active_jobs-=1
                    try:add_job(next(iterator));active_jobs+=1
                    except StopIteration:pass
                    continue
                state['messages']=state['messages']+[history_message(result.text)]
                f=pool.submit(state['env'].act,result.text,len(state['turns'])-1);tool_futures[f]=state
            for f in list(tool_futures):
                if not f.done():continue
                state=tool_futures.pop(f);observation,done,reason=f.result()
                proof=proven_ascii_context_overflow(observation,token_byte_bound,cfg['max_context_tokens'])
                if proof:
                    save(RUN/'episodes'/state['job']['id']/f'context-bound-{len(state["turns"])-1:03d}.json',proof)
                    # No next context is sampled. Retain the enormous exact raw
                    # stdout/stderr externally and the paper's bounded feedback view.
                    state['turns'][-1]['external_tool_observation']=proof
                    observation=bounded(observation,600)
                    if state['env'].observations:
                        state['env'].observations[-1]=observation
                    done=True;reason='context_limit'
                state['turns'][-1]['observation']=observation
                state['messages']=state['messages']+[{'role':'tool','content':observation}]
                if not done and len(state['turns'])>=cfg['max_solver_turns']:done=True;reason='turn_limit'
                if not done and state['total_tokens']>=cfg['max_solver_total_tokens']:done=True;reason='total_token_limit'
                if not done and cfg.get('max_episode_seconds') and time.monotonic()-state['env'].started>=cfg['max_episode_seconds']:done=True;reason='episode_time_limit'
                # Full context is never compacted; leave unfinished episodes explicitly incomplete.
                if not done:
                    proof=proven_ascii_context_overflow(observation,token_byte_bound,cfg['max_context_tokens'])
                    if proof:
                        save(RUN/'episodes'/state['job']['id']/f'context-bound-{len(state["turns"])-1:03d}.json',proof)
                        done=True;reason='context_limit'
                    else:
                        prefix=processor.apply_chat_template(state['messages'],tokenize=True,add_generation_prompt=True,
                            enable_thinking=True,return_dict=True,return_tensors='pt',tools=TOOLS)
                        if prefix['input_ids'].shape[-1]>=cfg['max_context_tokens']:
                            done=True;reason='context_limit'
                if done:
                    complete(state,reason);active_jobs-=1
                    try:add_job(next(iterator));active_jobs+=1
                    except StopIteration:pass
                else:enqueue(state)
            if not live and tool_futures:time.sleep(.05)
        for f in grade_futures:f.result()
    finally:
        for state in list(live.values())+list(tool_futures.values()):
            if 'env' in state:state['env'].close()
        pool.shutdown(wait=True);llm.llm_engine.engine_core.shutdown()
    return [results[j['id']] for j in jobs]

def main():
    p=argparse.ArgumentParser();p.add_argument('jobs');p.add_argument('--checkpoint');p.add_argument('--grade',action='store_true')
    a=p.parse_args();generate(load(a.jobs),a.checkpoint,a.grade)
if __name__=='__main__':main()
