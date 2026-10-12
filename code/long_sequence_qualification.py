"""Actual full-model longest admitted multimodal sequence forward/backward, no update."""
import torch,time
from common import *
from train import base,CPUAdam,process,logprobs
from tool_protocol import TOOLS
from transformers import AutoProcessor
cfg=config();torch.set_num_threads(12);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED)
processor=AutoProcessor.from_pretrained(str(MODEL));row=prompt_rows()[cfg['train'][0]]
msg=messages(dict(row,text='Inspect the public reference images and summarize dimensions.'))
prefix=processor.apply_chat_template(msg,tokenize=True,add_generation_prompt=True,return_dict=True,
    return_tensors='pt',enable_thinking=True,tools=TOOLS)
prompt=prefix['input_ids'][0].tolist();n=cfg['max_training_tokens']-len(prompt)
unit=processor.tokenizer.encode('We inspect the geometry and compare camera framing. ',add_special_tokens=False)
record={'id':'longest-native-probe','task':row['task'],'sampling_messages':msg,'prompt_token_ids':prompt,
    'token_ids':(unit*(n//len(unit)+1))[:n],'supervised_positions':list(range(n-16,n))}
model=base();model.train();trainer=CPUAdam(model,cfg);started=time.monotonic()
b,l,count=process(processor,record);(-logprobs(model,b,l,True).mean()).backward();trainer.accumulate()
norm=sum(float(g.square().sum()) for g in trainer.grads.values())**.5
if not (norm>0):raise RuntimeError('Longest-sequence gradient absent')
save(RUN/'qualification/max_sequence.json',{'passed':True,'sequence_tokens':b['input_ids'].shape[-1],
    'actual_full_model_backward':True,'diagnostic_backward_tokens':count,'optimizer_updates':0,'native_leaf_gradient_dtype':'float32',
    'gradient_norm':norm,'gradient_tensors':len(trainer.grads),'seconds':time.monotonic()-started,
    'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'not_counted_toward_training_budget':True})
print('MAX_SEQUENCE_QUALIFICATION',b['input_ids'].shape[-1],norm,flush=True)
