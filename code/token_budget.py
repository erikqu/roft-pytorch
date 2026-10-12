"""Whole fresh targets in completion order; one boundary target for exact token cap."""
def select_fresh(records,remaining,step):
    available=sum(len(r['token_ids']) for r in records)
    if available<=0:raise RuntimeError('No retained fresh reflections')
    budget=min(remaining,available);left=budget;selected=[];boundary=None
    for record in records:
        if left<=0:break
        n=min(left,len(record['token_ids']))
        if n<len(record['token_ids']):
            record=dict(record,supervised_positions=list(range(n)));boundary=record['id']
        selected.append(record);left-=n
    if left:raise RuntimeError('Token selection accounting error')
    return selected,{'selected_tokens':budget,'available_tokens':available,'partial_batch':budget<available,
        'reward_used_for_token_mask':False,'rule':'whole fresh targets in generation completion order; at most one partial boundary target',
        'boundary_target':boundary,'supervised_positions':{r['id']:r.get('supervised_positions',list(range(len(r['token_ids'])))) for r in selected}}
