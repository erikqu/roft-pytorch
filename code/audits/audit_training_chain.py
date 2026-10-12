"""Read-only completion evidence. Never substitutes a baseline or changes admission."""
import argparse
import hashlib
import json
from pathlib import Path
from raw_evidence_io import load_json

def load(path):return load_json(path)
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def audit(root,verify_large=False):
    cfg=load(root/'protocol.json');problems=[];steps=[];used=set();cumulative=0
    previous_identity=None
    for folder in sorted((root/'train/roft').glob('checkpoint-*')):
        ap=folder/'audit.json'
        if not ap.exists():problems.append({'checkpoint':folder.name,'reason':'update not saved/audited yet'});continue
        row=load(ap);step=row['step'];selection=load(root/'train/roft'/f'selection-{step:03d}.json')
        targets=load(root/'jobs'/f'roft-{step:03d}-targets.json')
        count=sum(len(t.get('supervised_positions',t['token_ids'])) for t in targets)
        if step!=len(steps)+1:problems.append('Noncontiguous updates')
        if row['optimizer_updates']!=1 or row['optimizer_step_counters']!=[step]:problems.append('Incorrect optimizer counters')
        if step>1 and row['resumed_optimizer_step_counters']!=[step-1]:problems.append('Optimizer did not resume prior step')
        if step>1:
            proof=row.get('native_resume_verification')
            if not proof or proof.get('native_fp32_parameters_exactly_restored')!=row['eligible_parameter_tensors']:
                problems.append('Native FP32 training parameters were not all verified against saved masters')
            elif not proof.get('cpu_cuda_rng_exactly_restored') or not proof.get('adam_moments_step_exactly_restored'):
                problems.append('Exact RNG or Adam moment/step restoration was not verified')
        if not row['numerical_reload_verified'] or row['actual_changed_tensors']<=0:problems.append('No verified numerical model update')
        if row.get('native_leaf_gradient_storage')!='FP32':problems.append('Native gradients were not FP32')
        if count!=row['supervised_tokens'] or count!=selection['selected_tokens']:problems.append('Backward-token count mismatch')
        identity=digest(load(folder/'weight_manifest.json'))
        checked_sources=[]
        for source_id in selection['attempt_ids']:
            source=load(root/'generations'/(source_id+'.json'))
            if source['checkpoint_sha256']!=previous_identity:problems.append('Source checkpoint identity mismatch: '+source_id)
            if source.get('policy_version')!=step-1:problems.append('Source was stale: '+source_id)
            checked_sources.append(source_id)
        partial=[]
        for target in targets:
            tid=target['id'];raw=load(root/'generations'/(tid+'.json'))
            if tid in used:problems.append('Repeated reflection target: '+tid)
            used.add(tid)
            if raw['checkpoint_sha256']!=previous_identity:problems.append('Reflection checkpoint identity mismatch: '+tid)
            if raw['attempt_id'] not in checked_sources:problems.append('Reflection source not admitted: '+tid)
            if raw['finish_reason']!='stop':problems.append('Truncated reflection admitted: '+tid)
            if target['raw_generation_token_ids']!=raw['token_ids']:problems.append('Raw reflection token identity changed: '+tid)
            if not target['preprocessed_target'].endswith('<|im_end|>\n'):problems.append('EOT/newline suffix omitted: '+tid)
            if 'supervised_positions' in target:
                if target['supervised_positions']!=list(range(len(target['supervised_positions']))):problems.append('Random target mask remains: '+tid)
                partial.append(tid)
            grade=load(root/'episodes'/raw['attempt_id']/'result.json')
            if grade.get('verifier_failure') and grade['validation_status']!='invalid_submission':problems.append('Invalid scene presented as verified')
        if len(partial)>1:problems.append('Multiple partially supervised reflections in one update')
        large_state='not inspected in metadata-only audit'
        if verify_large:
            if sha(folder/'resume.pt.gz')!=row['resume_sha256']:problems.append('Optimizer/RNG file hash mismatch')
            for name,expected in load(folder/'weight_manifest.json').items():
                if sha(folder/name)!=expected:problems.append('Model weight file hash mismatch')
            large_state='hash verified'
        cumulative+=count
        steps.append({'step':step,'backward_tokens':count,'cumulative_backward_tokens':cumulative,
            'actual_changed_tensors':row['actual_changed_tensors'],'fresh_source_attempts':len(checked_sources),
            'retained_target_count':len(targets),'boundary_target':partial,'checkpoint_identity':identity,
            'source_checkpoint_identity':previous_identity,'numerical_reload_verified':True,'full_state':large_state})
        previous_identity=identity
    complete=(root/'train/roft/complete.json').exists()
    if complete:
        receipt=load(root/'train/roft/complete.json')
        if cumulative!=cfg['total_supervised_tokens_per_arm'] or receipt['backward_tokens']!=cumulative:problems.append('Exact final budget not met')
    return {'scope':'full saved ROFT update chain currently present','steps':steps,'backward_tokens':cumulative,
        'target_backward_tokens':cfg['total_supervised_tokens_per_arm'],'training_complete_receipt':complete,
        'replayed_reflection_ids':len(used)!=sum(x['retained_target_count'] for x in steps),
        'problems':problems,'training_verified_complete':complete and not problems and bool(steps),
        'fresh_final_evaluation_complete':all((root/'evaluation'/f'{a}-complete.json').exists() for a in ['roft','base']),
        'note':'This audit alone does not prove model quality, complete final backup, or entire goal completion.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--verify-large',action='store_true')
    a=p.parse_args();print(json.dumps(audit(a.root,a.verify_large),indent=2))
