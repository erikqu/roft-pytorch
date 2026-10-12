"""Prove the existing context cutoff for enormous ASCII tool responses."""
import json


def ascii_token_byte_bound(tokenizer_path):
    data = json.loads(tokenizer_path.read_text())
    if data['model']['type'] != 'BPE' or data['normalizer'] != {'type': 'NFC'}:
        return None
    if data['decoder']['type'] != 'ByteLevel':
        return None
    stages = data['pre_tokenizer'].get('pretokenizers', [])
    if [stage['type'] for stage in stages] != ['Split', 'ByteLevel']:
        return None
    if stages[0].get('behavior') != 'Isolated' or stages[1].get('add_prefix_space'):
        return None
    # ByteLevel vocabulary spelling uses one visible Unicode symbol per input
    # byte. NFC leaves ASCII unchanged. Added tokens consume their literal bytes.
    return max(max(map(len, data['model']['vocab'])),
        max((len(t['content'].encode('utf-8')) for t in data['added_tokens']), default=0))


def proven_ascii_context_overflow(observation, max_token_bytes, context_tokens):
    if not max_token_bytes or len(observation) < max_token_bytes * context_tokens:
        return None
    if not observation.isascii():
        return None
    lower_bound = (len(observation) + max_token_bytes - 1) // max_token_bytes
    return {'reason': 'ASCII tool-response byte count proves the existing context limit exceeded',
        'observation_ascii_bytes': len(observation), 'maximum_token_bytes': max_token_bytes,
        'minimum_observation_tokens': lower_bound, 'context_limit_tokens': context_tokens,
        'observation_compacted_or_removed': False, 'raw_output_preserved': True}
