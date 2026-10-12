"""Explicit training-only exclusion of a preserved provider-refused judgment."""
import hashlib


def validate_unavailable(row, receipt, request_bytes):
    expected = {'attempt_id': row['id'], 'response_sha256': row['response_sha256'],
                'checkpoint_sha256': row['checkpoint_sha256'], 'scope': 'training',
                'request_sha256': hashlib.sha256(request_bytes).hexdigest(),
                'provider': 'azure', 'http_status': 400,
                'error_code': 'content_policy_violation', 'reward': None,
                'retry_performed': False}
    if row.get('phase') != 'training' or any(receipt.get(k) != v for k, v in expected.items()):
        raise RuntimeError('Grading exclusion identity or training scope changed')
    if not receipt.get('transport_diagnostic_sha256'):
        raise RuntimeError('Grading exclusion lacks retained provider evidence')
    return receipt
