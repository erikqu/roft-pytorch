"""Read-only source evidence audit; has no authority to route or modify training."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

def read(path):return json.loads(path.read_text())
def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def audit(root,step):
    rows=[read(p) for p in sorted((root/'generations').glob(f'roft-{step:03d}-attempt-*.json')) if 'reflection' not in p.name]
    modes=collections.Counter();checkpoints=collections.Counter();rewards=[];missing=[];invalid=0;observations=0;executions=0;identity_mismatch=[];reward_mismatch=[]
    for row in rows:
        modes[row['termination_reason']]+=1;checkpoints[str(row.get('checkpoint_sha256'))]+=1
        for turn in row['turns']:
            if turn['checkpoint_sha256']!=row['checkpoint_sha256']:identity_mismatch.append(turn['id'])
            obs=turn.get('observation')
            if obs is None:continue
            observations+=1
            if obs.startswith('Invalid native tool call:'):invalid+=1
            else:executions+=1
        grade=root/'episodes'/row['id']/'result.json'
        if not grade.exists():missing.append(row['id']);continue
        result=read(grade)
        reward_file=root/'episodes'/row['id']/'grader/verifier/reward.json'
        if not reward_file.exists() or read(reward_file).get('reward')!=result['reward'] or hash_file(reward_file)!=result['reward_file_sha256']:
            reward_mismatch.append(row['id'])
        if result['response_sha256']!=row['response_sha256']:identity_mismatch.append(row['id'])
        rewards.append({'id':row['id'],'reward':result['reward'],'saved':result['execution']['saved'],
            'validation_status':result.get('validation_status'),'verifier_failure':result.get('verifier_failure'),
            'reward_file_sha256':result['reward_file_sha256']})
    return {'scope':'all currently complete original source attempts for step '+str(step),
        'complete_sources':len(rows),'attempt_terminations':dict(modes),'checkpoint_identities':dict(checkpoints),
        'generated_source_tokens':sum(row['output_tokens'] for row in rows),
        'generated_source_turns':sum(len(row['turns']) for row in rows),'actual_observations':observations,
        'invalid_native_tool_observations':invalid,'noninvalid_observations':executions,
        'identity_mismatches':identity_mismatch,'rewards':rewards,'missing_rewards':missing,
        'recorded_reward_file_mismatches':reward_mismatch,
        'mean_actual_verifier_reward':sum(r['reward'] for r in rewards)/len(rewards) if rewards else None,
        'artifact_sha256':{row['id']:hash_file(root/'generations'/(row['id']+'.json')) for row in rows},
        'read_only':True,'routing_or_training_changes':False}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--step',type=int,default=1)
    args=p.parse_args();print(json.dumps(audit(args.root,args.step),indent=2))
