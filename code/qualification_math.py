"""Actual full Qwen multimodal microbatch gradients vs native autograd; no training admission."""
import gc
import torch
from common import *
from train import base,CPUAdam,logprobs,process
from tool_protocol import TOOLS
from transformers import AutoProcessor

cfg=config();torch.set_num_threads(12);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
model=base();model.train();processor=AutoProcessor.from_pretrained(str(MODEL));row=prompt_rows()[cfg['train'][0]]
msg=messages(dict(row,text='Describe the body and neck in the images.'))
prefix=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,return_tensors='pt',
    enable_thinking=True,tools=TOOLS)
records=[]
for text in ['The body is rounded.</think> Inspect camera orientation.<|im_end|>\n',
             'The neck is narrow.</think> Verify actual saved geometry.<|im_end|>\n']:
    records.append({'task':row['task'],'sampling_messages':msg,'prompt_token_ids':prefix['input_ids'][0].tolist(),
        'token_ids':processor.tokenizer.encode(text,add_special_tokens=False)})
total=sum(len(r['token_ids']) for r in records)
# Reference keeps each real microbatch gradient separately in FP32, exactly the reference recipe.
reference={}
initial_cpu_rng=torch.get_rng_state();initial_cuda_rng=torch.cuda.get_rng_state()
for r in records:
    b,l,n=process(processor,r);(-logprobs(model,b,l,True).sum()*cfg['data_gradient_scale']/total).backward()
    for name,p in model.named_parameters():
        if p.grad is not None:
            g=p.grad.detach().cpu().float()
            if name in reference:reference[name].add_(g)
            else:reference[name]=g
    model.zero_grad(set_to_none=True)
torch.set_rng_state(initial_cpu_rng);torch.cuda.set_rng_state(initial_cuda_rng)
trainer=CPUAdam(model,cfg)
for r in records:
    b,l,n=process(processor,r);(-logprobs(model,b,l,True).sum()*cfg['data_gradient_scale']/total).backward();trainer.accumulate()
difference=0.;magnitude=0.;max_error=0.
for i,(name,p) in enumerate(trainer.named):
    if name not in reference:
        if i in trainer.grads:raise RuntimeError('Gradient eligibility changed')
        continue
    candidate=trainer.grads[i];native=reference[name]
    difference+=float((candidate-native).square().sum());magnitude+=float(native.square().sum())
    max_error=max(max_error,float((candidate-native).abs().max()))
relative=(difference/max(magnitude,1e-30))**.5
# Independently measure native/native repeatability under exactly the same RNG.
torch.set_rng_state(initial_cpu_rng);torch.cuda.set_rng_state(initial_cuda_rng)
control={}
for r in records:
    b,l,n=process(processor,r);(-logprobs(model,b,l,True).sum()*cfg['data_gradient_scale']/total).backward()
    for name,p in model.named_parameters():
        if p.grad is not None:
            g=p.grad.detach().cpu().float()
            if name in control:control[name].add_(g)
            else:control[name]=g
    model.zero_grad(set_to_none=True)
native_difference=sum(float((control[name]-g).square().sum()) for name,g in reference.items())
native_relative=(native_difference/max(magnitude,1e-30))**.5
if relative>max(.01,native_relative*1.1):
    raise RuntimeError('Gradient mismatch exceeds measured native repeatability: '+str((relative,native_relative)))
save(RUN/'qualification/native_gradient_equivalence.json',{'passed':True,'relative_l2':relative,
    'max_abs_error':max_error,'actual_full_qwen_model':True,'actual_public_images':True,
    'microbatches':len(records),'diagnostic_backward_tokens':3*total,'optimizer_updates':0,'native_leaf_gradient_dtype':'float32',
    'parameter_gradients_compared':len(reference),'not_counted_toward_training_budget':True,
    'relative_l2_floor':.01,'measured_native_repeatability_relative_l2':native_relative,
    'criterion':'candidate/native discrepancy <= max(1%,1.1 times measured native/native discrepancy)',
    'precision_note':'Native backward before FP32 accumulation retains tied-parameter accumulation semantics'})
print('NATIVE_GRADIENT_QUALIFICATION',relative,len(reference),flush=True)
