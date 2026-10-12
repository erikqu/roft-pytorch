"""Literal Appendix B bounds and target preprocessing; no semantic filtering."""
import re

def bounded(text,budget):
    text=str(text)
    if len(text)<=budget:return text
    head=budget*2//3
    return text[:head]+'\n[...truncated...]\n'+text[-(budget-head):]

def trajectory(attempt):
    parts=[]
    for turn in attempt.get('turns',[]):
        raw=turn['text'];split=raw.split('</think>',1)
        thought=split[0].removeprefix('<think>') if len(split)==2 else ''
        action=split[-1]
        parts.append('Thought: '+bounded(thought,600)+'\nAction: '+bounded(action,600)+
            '\nObservation: '+bounded(turn.get('observation',''),600))
    return bounded('\n\n'.join(parts),24000)

def accepted_target(record,tokenizer):
    if record['finish_reason']!='stop':return None,'unfinished'
    if not record['text'].strip():return None,'empty'
    text='<think>\n'+record['text'].strip()
    text=re.sub(r'(?:<\|im_end\|>|<\|endoftext\|>)\s*$','',text).strip()
    text=re.sub(r'^\s*<think>\s*</think>\s*','',text)
    if not text:return None,'empty'
    ids=tokenizer.encode(text+'<|im_end|>\n',add_special_tokens=False)
    if not ids:return None,'empty'
    return dict(record,token_ids=ids,preprocessed_target=text+'<|im_end|>\n',reflection_target=True,
        raw_generation_token_ids=record['token_ids'],target_preprocessing='Appendix B whitespace/empty-thinking/EOT-newline'),None
