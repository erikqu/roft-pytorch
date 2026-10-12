"""Paper Appendix C equations; task-dependent Blender rewards stay unchanged."""
import torch

def group_advantages(rewards, indices, size=8):
    rewards=torch.as_tensor(rewards,dtype=torch.float32)
    result=torch.zeros_like(rewards)
    for identity in sorted(set(indices)):
        positions=[i for i,x in enumerate(indices) if x==identity]
        if len(positions)!=size: raise ValueError('Incomplete original GRPO group')
        values=rewards[positions]
        if len(set(values.tolist()))==1: raise ValueError('Zero-advantage group must be replenished')
        result[positions]=(values-values.mean())/(values.std(correction=1)+1e-6)
    return result

def policy_tokens(logp, old, advantage, cfg):
    p=logp.exp();q=old.exp();threshold=cfg['grpo_binary_tv_threshold']
    mask=torch.where(advantage>0,p-q<=threshold,p-q>=-threshold).detach()
    return -(mask.to(logp.dtype)*torch.exp(logp-old)*advantage)

def response_logprobs(model, batch, labels, needs_grad):
    from torch.utils.checkpoint import checkpoint
    hidden=model.model(**batch,use_cache=False).last_hidden_state
    active=labels[:,1:]!=-100
    states=hidden[:,:-1][active];targets=labels[:,1:][active]
    def head(h,y):
        return -torch.nn.functional.cross_entropy(model.lm_head(h).float(),y,reduction='none')
    parts=[]
    for start in range(0,len(targets),32):
        args=(states[start:start+32],targets[start:start+32])
        parts.append(checkpoint(head,*args,use_reentrant=False) if needs_grad else head(*args))
    if not parts:raise ValueError('No supervised tokens')
    return torch.cat(parts)
