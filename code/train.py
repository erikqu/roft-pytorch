"""Full BF16 model, FP32 CPU gradient accumulation/master AdamW, one fresh pass."""
import argparse
import gc
import gzip
import hashlib
import subprocess
import tempfile
import time
from pathlib import Path
import torch
from common import *
from objectives import response_logprobs,policy_tokens
from tool_protocol import TOOLS


def base(checkpoint=None):
    from transformers import Qwen3_5ForConditionalGeneration
    model=Qwen3_5ForConditionalGeneration.from_pretrained(str(checkpoint or MODEL),
        dtype=torch.bfloat16,device_map={'':0},attn_implementation='sdpa')
    # FP32 parameter/gradient storage with BF16 forward autocast gives FP32
    # within-backward accumulation as well as between-microbatch accumulation.
    model.to(dtype=torch.float32)
    model.requires_grad_(True);model.config.use_cache=False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    return model


def process(processor,record):
    from generate import job_messages
    batch=processor.apply_chat_template(job_messages(record),tokenize=True,add_generation_prompt=True,
        return_dict=True,return_tensors='pt',enable_thinking=True,tools=[] if record.get('reflection') else TOOLS)
    if batch['input_ids'][0].tolist()!=record['prompt_token_ids']:
        raise RuntimeError('Training context differs from sampled multimodal prefix')
    if record.get('reflection_target'):
        opening=processor.tokenizer.encode('<think>\n',add_special_tokens=False)
        if batch['input_ids'][0,-len(opening):].tolist()!=opening:
            raise RuntimeError('Unexpected thinking generation prefix')
        for name in ('input_ids','attention_mask','mm_token_type_ids'):
            if name in batch:batch[name]=batch[name][:,:-len(opening)]
    n=batch['input_ids'].shape[-1];ids=record['token_ids']
    if n+len(ids)>config()['max_training_tokens']:raise ValueError('Sequence exceeds frozen bound')
    batch['input_ids']=torch.cat([batch['input_ids'],torch.tensor([ids],dtype=torch.long)],dim=-1)
    batch['attention_mask']=torch.ones_like(batch['input_ids'])
    if 'mm_token_type_ids' in batch:
        batch['mm_token_type_ids']=torch.cat([batch['mm_token_type_ids'],torch.zeros((1,len(ids)),dtype=batch['mm_token_type_ids'].dtype)],dim=-1)
    labels=torch.full_like(batch['input_ids'],-100)
    positions=record.get('supervised_positions',list(range(len(ids))))
    if len(set(positions))!=len(positions) or any(i<0 or i>=len(ids) for i in positions):raise RuntimeError('Invalid supervision mask')
    if positions:
        active=torch.tensor([n+i for i in positions],dtype=torch.long);labels[:,active]=batch['input_ids'][:,active]
    return batch,labels,len(positions)


def logprobs(model,batch,labels,grad):
    batch={k:v.to('cuda') for k,v in batch.items()};labels=labels.to('cuda')
    context=torch.autograd.graph.save_on_cpu(pin_memory=True) if grad else torch.no_grad()
    with context,torch.autocast('cuda',dtype=torch.bfloat16):
        return response_logprobs(model,batch,labels,grad)


class CPUAdam:
    """Accumulate leaf gradients in FP32 between microbatches; preserve masters/moments."""
    def __init__(self,model,cfg):
        self.named=list(model.named_parameters());self.masters=[];self.grads={};self.hooks=[]
        for name,p in self.named:
            master=torch.nn.Parameter(p.detach().cpu().float());self.masters.append(master)
        self.optimizer=torch.optim.AdamW(self.masters,lr=cfg['learning_rate'],betas=tuple(cfg['betas']),
            eps=cfg['adam_epsilon'],weight_decay=cfg['weight_decay'],foreach=False)
    def accumulate(self):
        # Capture after native backward completes, so tied embeddings/head accumulate
        # through the identical native path before FP32 microbatch accumulation.
        for index,(name,p) in enumerate(self.named):
            if p.grad is None:continue
            g=p.grad.detach().to(device='cpu',dtype=torch.float32)
            if index in self.grads:self.grads[index].add_(g)
            else:self.grads[index]=g
            p.grad=None
    def step(self,cfg):
        for index,g in self.grads.items():self.masters[index].grad=g
        norm=torch.nn.utils.clip_grad_norm_(self.masters,cfg['max_grad_norm'],foreach=False)
        if not torch.isfinite(norm) or norm<=0:raise RuntimeError('No finite nonzero complete-batch gradient')
        self.optimizer.step();changed=0;subsystems=set()
        for (name,p),master in zip(self.named,self.masters):
            new=master.detach().to(device=p.device,dtype=p.dtype)
            if not torch.equal(new,p.detach()):changed+=1;subsystems.add('vision' if '.visual.' in name else 'language')
            p.data.copy_(new)
        self.optimizer.zero_grad(set_to_none=True);count=len(self.grads);self.grads.clear()
        if not changed:raise RuntimeError('No numerical BF16 model update')
        return float(norm),changed,count,sorted(subsystems)
    def state(self):
        return {'masters':[p.detach() for p in self.masters],'optimizer':self.optimizer.state_dict(),
            'names':[n for n,p in self.named],'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()}
    def restore(self,state):
        if state['names']!=[n for n,p in self.named]:raise RuntimeError('Optimizer parameter identities changed')
        exact_parameters=0
        for ((name,model_parameter),p,value) in zip(self.named,self.masters,state['masters']):
            if value.dtype!=torch.float32 or model_parameter.dtype!=torch.float32:
                raise RuntimeError('Native/master resume must retain FP32 storage')
            p.data.copy_(value)
            # Serving checkpoints are BF16 exports. Native training resumes the
            # exact saved FP32 weights, not a newly rounded serving snapshot.
            model_parameter.data.copy_(value.to(device=model_parameter.device,dtype=model_parameter.dtype))
            if not torch.equal(model_parameter.detach().cpu(),value) or not torch.equal(p.detach(),value):
                raise RuntimeError('Native FP32 model or CPU master differs from saved state: '+name)
            exact_parameters+=1
        self.optimizer.load_state_dict(state['optimizer'])
        torch.set_rng_state(state['cpu_rng']);torch.cuda.set_rng_state(state['cuda_rng'])
        if not torch.equal(torch.get_rng_state(),state['cpu_rng']) or not torch.equal(torch.cuda.get_rng_state(),state['cuda_rng']):
            raise RuntimeError('CPU/CUDA RNG states did not resume exactly')
        for index,expected in state['optimizer']['state'].items():
            actual=self.optimizer.state[self.masters[index]]
            for name in ('step','exp_avg','exp_avg_sq'):
                if not torch.equal(actual[name],expected[name]):
                    raise RuntimeError('Restored Adam moment/step differs: '+str((index,name)))
        return {'native_fp32_parameters_exactly_restored':exact_parameters,
            'cpu_master_parameters_exactly_restored':exact_parameters,
            'cpu_cuda_rng_exactly_restored':True,'adam_moments_step_exactly_restored':True}


def load_resume(folder):
    audit=load(Path(folder)/'audit.json');path=Path(folder)/'resume.pt.gz'
    if sha(path)!=audit['resume_sha256']:raise RuntimeError('Resume state hash mismatch')
    # torch.load seeks through a zip archive. A gzip file object repeatedly
    # decompresses its 48GB stream on seek; decompress once into a temporary
    # file, then map the exact same serialized tensor storages.
    with tempfile.TemporaryDirectory(prefix='.verified-resume-',dir=folder) as temporary:
        native=Path(temporary)/'resume.pt'
        with native.open('wb') as output:
            result=subprocess.run(['pigz','-d','-c',str(path)],stdout=output,stderr=subprocess.PIPE)
        if result.returncode:raise RuntimeError('Verified resume decompression failed')
        return torch.load(native,map_location='cpu',weights_only=False,mmap=True)


def update(arm,step,records,previous=None,qualification=False):
    from transformers import AutoProcessor
    cfg=config();torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);torch.set_num_threads(12)
    started=time.monotonic();processor=AutoProcessor.from_pretrained(str(MODEL));model=base(previous)
    trainer=CPUAdam(model,cfg)
    resumed=[];resume_proof=None
    if previous:
        state=load_resume(previous);resume_proof=trainer.restore(state)
        resumed=sorted(set(int(v['step']) for v in trainer.optimizer.state.values()))
        if resumed!=[step-1]:raise RuntimeError('AdamW step counters did not resume')
        save(RUN/'train'/arm/f'resume-{step:03d}.json',dict(resume_proof,at=time.time(),
            source_checkpoint=previous,source_resume_sha256=load(Path(previous)/'audit.json')['resume_sha256'],
            optimizer_step_counters=resumed,before_any_backward_for_this_update=True))
        del state;gc.collect()
    total=sum(len(r.get('supervised_positions',r['token_ids'])) for r in records)
    if total<=0:raise RuntimeError('No actual targets')
    model.train();loss_sum=0.;completed=0;agreement=[]
    for record in records:
        batch,labels,n=process(processor,record)
        if not n:continue
        if arm=='grpo':
            model.eval();teacher=logprobs(model,batch,labels,False).cpu()
            pos=record.get('supervised_positions',list(range(len(record['token_ids']))))
            old=torch.tensor([record['sampled_raw_logprobs'][j] for j in pos],dtype=torch.float32)
            error=float((teacher-old).abs().mean());agreement.append(error)
            if error>.25:raise RuntimeError('Sampled/training probability disagreement: '+str(error))
            model.train()
        lp=logprobs(model,batch,labels,True)
        if arm=='roft':value=-lp.sum()/total
        else:value=policy_tokens(lp,old.to(lp.device),torch.tensor(record['advantage'],device=lp.device),cfg).sum()*record['trajectory_weight']
        if not torch.isfinite(value):raise RuntimeError('Nonfinite objective')
        (value*cfg['data_gradient_scale']).backward();trainer.accumulate();loss_sum+=float(value.detach());completed+=n
        append(RUN/'train'/arm/'microbatches.jsonl',{'step':step,'id':record['id'],'backward_tokens':n,
            'batch_completed_tokens':completed,'batch_target_tokens':total,'loss_contribution':float(value.detach())})
        del lp,value,batch,labels
    if completed!=total:raise RuntimeError('Backward token accounting mismatch')
    norm,changed,grad_count,subsystems=trainer.step(cfg)
    folder=RUN/'qualification'/arm if qualification else RUN/'train'/arm/f'checkpoint-{step:03d}'
    folder.mkdir(parents=True,exist_ok=True)
    save(folder/'optimizer_applied_unverified.json',{'at':time.time(),'step':step,
        'supervised_tokens':total,'actual_changed_tensors':changed,'gradient_parameter_tensors':grad_count,
        'loss':loss_sum,'gradient_norm_after_data_scale_before_clipping':norm,
        'state':'optimizer applied; checkpoint save/reload not verified yet',
        'completed_update_admitted':False})
    # Export BF16 serving weights while keeping optimizer FP32 masters intact.
    # Export learned parameters only. Module.to also rounds nonpersistent rotary
    # buffers, which are regenerated in FP32 by a fresh load and alters the probe.
    for parameter in model.parameters():
        parameter.data=parameter.data.to(dtype=torch.bfloat16)
    model.save_pretrained(folder,max_shard_size='5GB');processor.save_pretrained(folder)
    weights={p.name:sha(p) for p in folder.glob('*.safetensors')};save(folder/'weight_manifest.json',weights)
    state=trainer.state();steps=sorted(set(int(v['step']) for v in trainer.optimizer.state.values()))
    if steps!=[step]:raise RuntimeError('Optimizer counters differ from admitted update')
    temporary=folder/'resume.pt.gz.tmp'
    with temporary.open('wb') as output:
        compressor=subprocess.Popen(['pigz','-1','-p','12','-c'],stdin=subprocess.PIPE,stdout=output,stderr=subprocess.PIPE)
        try:
            torch.save(state,compressor.stdin);compressor.stdin.close()
            if compressor.wait()!=0:raise RuntimeError('Checkpoint gzip compression failed')
        except BaseException:
            compressor.kill();compressor.wait();raise
    temporary.replace(folder/'resume.pt.gz')
    # Actual image-conditioned reload probe, not just checkpoint existence.
    first=next(r for r in records if len(r.get('supervised_positions',r['token_ids'])))
    short=dict(first,token_ids=first['token_ids'][:16],supervised_positions=list(range(min(16,len(first['token_ids'])))))
    probe,labels,_=process(processor,short);model.to(dtype=torch.float32);model.eval();expected=logprobs(model,probe,labels,False).cpu()
    parameter_count=sum(p.numel() for p in model.parameters());eligible_tensors=len(list(model.parameters()))
    for hook in trainer.hooks:hook.remove()
    del model,trainer,state;gc.collect();torch.cuda.empty_cache()
    reloaded=base(folder);reloaded.eval();actual=logprobs(reloaded,probe,labels,False).cpu()
    torch.testing.assert_close(actual,expected,rtol=.02,atol=.04)
    audit={'arm':arm,'step':step,'qualification':qualification,'supervised_tokens':total,'optimizer_updates':1,
        'actual_changed_tensors':changed,'gradient_parameter_tensors':grad_count,'eligible_parameter_tensors':eligible_tensors,
        'trainable_parameter_count':parameter_count,'full_parameter_training':True,'changed_subsystems':subsystems,
        'loss':loss_sum,'gradient_norm_after_data_scale_before_clipping':norm,'data_gradient_scale':cfg['data_gradient_scale'],
        'optimizer_seconds':time.monotonic()-started,'optimizer_step_counters':steps,'resumed_optimizer_step_counters':resumed,
        'weight_manifest_sha256':sha(folder/'weight_manifest.json'),'resume_sha256':sha(folder/'resume.pt.gz'),
        'numerical_reload_verified':True,'reload_max_logprob_error':float((actual-expected).abs().max()),
        'rollout_logprob_mean_errors':agreement,'target_ids':[r['id'] for r in records],
        'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'fp32_gradient_accumulation':True,'native_leaf_gradient_storage':'FP32','compute_precision':'BF16 autocast',
        'fp32_master_weights_and_adam_state':True,'model_precision':'BF16','replayed_targets':0}
    audit['native_resume_verification']=resume_proof
    save(folder/'audit.json',audit);print(json.dumps(audit),flush=True)
    return folder


def qualification():
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(str(MODEL));row=prompt_rows()[config()['train'][0]]
    msg=messages(dict(row,text='Describe these shapes.'))
    prefix=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',
        enable_thinking=True,preserve_thinking=True)
    ids=processor.tokenizer.encode('The visible shape has a body and a narrow extension.</think> I should inspect the cameras.<|im_end|>\n',add_special_tokens=False)
    records=[{'id':'native-qualification','task':row['task'],'sampling_messages':msg,
        'prompt_token_ids':prefix['input_ids'][0].tolist(),'token_ids':ids,'finish_reason':'stop'}]
    update('roft',1,records,qualification=True)
    # Longest configured sequence must execute a real backward before launch.
    save(RUN/'qualification/native_update.json',{'passed':True,'receipt':'roft/audit.json',
        'note':'Real full-model multimodal optimizer and serialized reload qualification; maximum-length backward still required separately'})


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['update','qualify']);p.add_argument('--arm');p.add_argument('--step',type=int,default=1)
    p.add_argument('--records');p.add_argument('--previous');a=p.parse_args()
    if a.action=='qualify':qualification()
    else:update(a.arm,a.step,load(a.records),a.previous)
if __name__=='__main__':main()
