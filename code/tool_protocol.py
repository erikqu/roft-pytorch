"""Qwen3.5 native function serialization, read from its pinned chat template."""
import re
from pathlib import PurePosixPath

TOOLS=[{'type':'function','function':{'name':'bash','description':'Execute a shell command in the persistent Blender task container. Working directory is /app. Read files, inspect errors, run Blender and validate the saved scene.',
    'parameters':{'type':'object','properties':{'command':{'type':'string','description':'Shell command to execute.'}},'required':['command']}}},
 {'type':'function','function':{'name':'write_file','description':'Write UTF-8 content to a file under /app. Use to create or revise Blender Python scripts without shell string escaping.',
    'parameters':{'type':'object','properties':{'path':{'type':'string','description':'Absolute file path under /app.'},'content':{'type':'string','description':'Exact file content, including newlines.'}},'required':['path','content']}}},
 {'type':'function','function':{'name':'finish','description':'Submit the current saved /app/result.blend and end the task. Save and validate the scene before finishing.',
    'parameters':{'type':'object','properties':{'summary':{'type':'string','description':'Brief summary of completed work and validation.'}},'required':['summary']}}}]

def parse_calls(text):
    final=text.split('</think>',1)[-1].strip()
    blocks=re.findall(r'<tool_call>\s*(.*?)\s*</tool_call>',final,re.S)
    if not blocks:raise ValueError('Expected native <tool_call><function=...>...</function></tool_call>')
    calls=[]
    for block in blocks:
        match=re.fullmatch(r'<function=([A-Za-z_][A-Za-z_0-9]*)>\s*(.*?)\s*</function>',block,re.S)
        if not match:raise ValueError('Invalid native function wrapper')
        name,body=match.groups();allowed=next((t['function'] for t in TOOLS if t['function']['name']==name),None)
        if allowed is None:raise ValueError('Unknown function: '+name)
        args={}
        for parameter in re.finditer(r'<parameter=([A-Za-z_][A-Za-z_0-9]*)>\n?(.*?)\n?</parameter>',body,re.S):
            key,value=parameter.groups()
            if key in args:raise ValueError('Duplicate function parameter: '+key)
            if key not in allowed['parameters']['properties']:raise ValueError('Unexpected function parameter: '+key)
            args[key]=value
        if any(key not in args for key in allowed['parameters']['required']):raise ValueError('Missing required function parameter')
        calls.append({'name':name,'arguments':args})
    if any(c['name']=='finish' for c in calls[:-1]):raise ValueError('Finish must be the final function call')
    return calls


def history_message(text):
    # Keep the exact raw model text. Its native XML calls are already serialized;
    # adding tool_calls would duplicate them in the pinned Qwen chat template.
    return {'role':'assistant','content':text}
