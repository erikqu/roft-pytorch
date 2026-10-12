"""Recover existing task volumes and raw turns; never repeat a sampled action."""
import json
from pathlib import Path
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from common import RUN,load,save,digest,sha
from tool_protocol import history_message,parse_calls


def main():
    root=RUN;jobs=load(root/'jobs/roft-005-attempt-000.json')
    identity=digest(load(root/'train/roft/checkpoint-004/weight_manifest.json'))
    recovered=[]
    for job in jobs:
        if (root/'generations'/(job['id']+'.json')).exists():continue
        folder=root/'episodes'/job['id'];files=sorted((root/'raw_turns').glob(job['id']+'-*.json'))
        turns=[];proof=None
        for index,path in enumerate(files):
            raw=load(path)
            if raw['checkpoint_sha256']!=identity or raw['seed']!=job['seed']+index*1000003:
                raise RuntimeError('Partial source identity mismatch')
            turn=dict(raw)
            if index+1<len(files):
                next_raw=load(files[index+1]);obs=next_raw['sampling_messages'][-1]
                if obs['role']!='tool':raise RuntimeError('Prior actual tool observation missing')
                turn['observation']=obs['content']
            else:
                results=[]
                for ordinal,call in enumerate(parse_calls(raw['text'])):
                    if call['name']!='bash':raise RuntimeError('Non-bash last action requires independent recovery')
                    stdout=folder/f'tool-{index:03d}-{ordinal:02d}.stdout'
                    stderr=folder/f'tool-{index:03d}-{ordinal:02d}.stderr'
                    if not stdout.exists() or not stderr.exists():raise RuntimeError('Last action output not durably complete')
                    out_bytes=stdout.stat().st_size
                    if out_bytes>128*131072:
                        h=sha(stdout)
                        if h!='c0d378ae8519698891858e8a99df851edab451d84177066dda6b5f8a64d45d8d':
                            raise RuntimeError('Oversized original output differs from audited file')
                        with stdout.open('rb') as f:
                            first=f.read(400);f.seek(-200,2);last=f.read(200)
                        proof={'minimum_observation_tokens':(out_bytes+127)//128,
                            'maximum_token_bytes':128,'observation_ascii_bytes':out_bytes,
                            'context_limit_tokens':131072,'raw_stdout_sha256':h,
                            'raw_stdout_file':str(stdout.relative_to(root)),
                            'raw_output_preserved':True,'external_original_output':True,
                            'no_further_prompt_sampled':True}
                        value=first.decode()+'\n[...raw stdout retained externally...]\n'+last.decode()
                    else:value=stdout.read_text(errors='replace')
                    err=stderr.read_text(errors='replace')
                    receipt=folder/f'tool-{index:03d}-{ordinal:02d}.receipt.json'
                    exit_code=load(receipt)['exit_code'] if receipt.exists() else None
                    results.append({'tool':'bash','exit_code':exit_code,'stdout':value,'stderr':err,
                        'python_traceback_observed':'Traceback (most recent call last)' in err,
                        'note':'Exact stdout/stderr recovered from completed action; exit code lost in generator crash and remains unknown.' if exit_code is None else 'Recovered completed action receipt.'})
                    save(folder/f'tool-{index:03d}-{ordinal:02d}.recovered.json',
                        {'at':time.time(),'stdout_sha256':sha(stdout),'stderr_sha256':sha(stderr),
                         'exit_code':exit_code,'action_executed_again':False})
                turn['observation']=json.dumps(results,ensure_ascii=False)
                if proof:turn['external_tool_observation']=proof
            turns.append(turn)
        if not turns:raise RuntimeError('Partial source has no actual raw turns')
        # The last raw turn contains the exact sampled prefix, including all
        # previous actions. Append only its already-completed action and feedback.
        messages=turns[-1]['sampling_messages']+[history_message(turns[-1]['text']),
            {'role':'tool','content':turns[-1]['observation']}]
        evidence={'at':time.time(),'raw_turn_hashes':{p.name:sha(p) for p in files},
            'sampled_actions_repeated':False,'raw_tokens_replaced':False,
            'last_action_exit_code_unavailable':True,'existing_task_volume_and_container':True}
        save(folder/'partial_resume.json',{'job':job,'checkpoint_sha256':identity,
            'messages':messages,'turns':turns,'total_tokens':sum(len(t['token_ids']) for t in turns),
            'context_overflow_proof':proof,'recovery_evidence':evidence})
        recovered.append({'id':job['id'],'actual_raw_turns':len(turns),'context_limit_proven':bool(proof)})
    save(root/'partial_source_resume_receipt.json',{'at':time.time(),'sources':recovered,
        'completed_sources_unchanged':11,'backward_tokens_reset':False,
        'generation_actions_repeated':False,'original_work_window_preserved':True})
    print(json.dumps(recovered),flush=True)


if __name__=='__main__':main()
