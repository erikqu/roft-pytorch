"""Recover only fully saved full-weight numerical updates; never reset consumed tokens."""
from common import load,sha

def recover(root,target,arm):
    audits=[];tokens=0;last=None;used=set()
    for folder in sorted((root/'train'/arm).glob('checkpoint-*')):
        if not (folder/'audit.json').exists():raise RuntimeError('Interrupted checkpoint requires inspection: '+str(folder))
        a=load(folder/'audit.json')
        if a['step']!=len(audits)+1 or a['optimizer_updates']!=1:raise RuntimeError('Noncontiguous online update chain')
        if not a['numerical_reload_verified'] or a['actual_changed_tensors']<=0:raise RuntimeError('Unverified full-weight update')
        if sha(folder/'resume.pt.gz')!=a['resume_sha256']:raise RuntimeError('Saved optimizer/RNG state changed')
        if sha(folder/'weight_manifest.json')!=a['weight_manifest_sha256']:raise RuntimeError('Saved weight identities changed')
        for name,expected in load(folder/'weight_manifest.json').items():
            if sha(folder/name)!=expected:raise RuntimeError('Saved full weights changed')
        if a['optimizer_step_counters']!=[a['step']]:raise RuntimeError('AdamW step counters inconsistent')
        ids=set(a['target_ids'])
        if len(ids)!=len(a['target_ids']) or ids&used:raise RuntimeError('Previously consumed target replay')
        used|=ids;tokens+=a['supervised_tokens']
        if tokens>target:raise RuntimeError('Backward target exceeded')
        audits.append(a);last=folder
    return audits,tokens,last,used
