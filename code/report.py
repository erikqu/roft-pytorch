"""Only fully measured fresh arms populate candidate reward columns."""
import random
from common import *

def main():
    result={'protocol_sha256':sha(RUN/'protocol.json'),'model':config()['model'],
        'implementation_provenance':config()['implementation_provenance'],'arms':{},'deltas':{}}
    paired={}
    for arm in ['base']+config()['arms']:
        receipt=RUN/'evaluation'/f'{arm}-complete.json'
        if not receipt.exists():
            result['arms'][arm]={'provenance':'not measured'};continue
        jobs=load(receipt)['jobs'];rows=[]
        for identity in jobs:
            path=RUN/'episodes'/identity/'result.json'
            if not path.exists():raise RuntimeError('Full-set evaluation has missing actual reward: '+identity)
            rows.append(load(path))
        scores={}
        for split in ('transfer','within_task'):
            subset=[r for r in rows if r['split']==split]
            scores[split]={'mean_reward':sum(r['reward'] for r in subset)/len(subset),
                'n':len(subset),'actual_verifier_results':True}
        result['arms'][arm]={'provenance':'fresh inference','scores':scores,
            'episodes':[r['id'] for r in rows],
            'backward_tokens':0 if arm=='base' else load(RUN/'train'/arm/'complete.json')['backward_tokens']}
        paired[arm]={(r['split'],r['task'],r['seed']):r['reward'] for r in rows}
    if 'roft' in paired:
        for control in ('base','grpo'):
            if control not in paired:continue
            for split in ('transfer','within_task'):
                keys=sorted(k for k in paired['roft'] if k[0]==split)
                if set(keys)!={k for k in paired[control] if k[0]==split}:raise RuntimeError('Unpaired evaluation membership')
                differences=[paired['roft'][k]-paired[control][k] for k in keys]
                # Task-cluster bootstrap keeps repeats of the same task together.
                tasks=sorted({k[1] for k in keys});rng=random.Random(SEED);boot=[]
                for _ in range(10000):
                    sample=[rng.choice(tasks) for _ in tasks]
                    values=[paired['roft'][k]-paired[control][k] for t in sample for k in keys if k[1]==t]
                    boot.append(sum(values)/len(values))
                boot.sort();result['deltas'][f'roft_minus_{control}_{split}']={
                    'mean_reward_delta':sum(differences)/len(differences),'ci95_task_bootstrap':[boot[249],boot[9749]],
                    'paired_fixes':sum(x>0 for x in differences),'paired_regressions':sum(x<0 for x in differences)}
    save(RUN/'results.json',result)
    lines=['# Paper-derived Blender rerun','',
        'This is a local implementation of the ROFT paper recipe. No official ROFT trainer was found.',
        'Qwen3.5-4B full-parameter training on one H100; continuous Blender verifier rewards, not the paper coding success rate.','',
        '| Arm | Provenance | Backward tokens | Transfer reward | Within-task reward |',
        '|---|---|---:|---:|---:|']
    for arm,record in result['arms'].items():
        scores=record.get('scores',{})
        fmt=lambda split:f"{scores[split]['mean_reward']:.5f}" if split in scores else 'not measured'
        lines.append(f"| {arm} | {record['provenance']} | {record.get('backward_tokens','not measured')} | {fmt('transfer')} | {fmt('within_task')} |")
    lines+=['','All results require preserved raw multi-turn samples, real verifier rewards and audited numerical updates.',
        'Previous 27B one-shot results are separate historical experiments. The node remains running by user instruction.']
    (RUN/'RESULTS.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
