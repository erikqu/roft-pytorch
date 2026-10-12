"""Require full training, fresh paired outputs, saved states and verified backups."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_training_chain import audit as training_audit
from audit_evaluation import audit as evaluation_audit
from audit_backward_ledger import audit as backward_audit
from audit_report import audit as report_audit


def read(path):
    return json.loads(path.read_text()) if path.exists() else None


def audit(artifacts):
    root = artifacts / 'remote'
    requirements = []

    def check(name, passed, evidence):
        requirements.append({'requirement': name, 'verified': bool(passed), 'evidence': evidence})

    chain = training_audit(root)
    paired = evaluation_audit(root)
    backwards = backward_audit(root)
    native = read(root / 'qualification/native_gradient_equivalence.json')
    longest = read(root / 'qualification/max_sequence.json')
    end_to_end = read(root / 'qualification/complete.json')
    check('Actual full-model native FP32 gradient and131072sequence qualification',
        native and native['passed'] and native['actual_full_qwen_model'] and
        native['parameter_gradients_compared'] == 723 and native['native_leaf_gradient_dtype'] == 'float32' and
        longest and longest['passed'] and longest['sequence_tokens'] == 131072 and
        longest['actual_full_model_backward'] and longest['native_leaf_gradient_dtype'] == 'float32',
        {'native': native, 'maximum_sequence': longest})
    check('Native tools/readableBlender/freshreflection/matching sampled-prefix qualification',
        end_to_end and all(end_to_end.get(k) for k in ['passed', 'actual_native_tool_execution',
            'actual_readable_blender_scene', 'fresh_thinking_reflection', 'sampled_prefix_exact',
            'all_target_thinking_eot_supervised']), end_to_end)
    check('Paper mismatch table and explicit remaining adaptations exist',
        (artifacts.parent / 'METHODOLOGY_ALIGNMENT.md').exists() and
        (artifacts.parent.parent / 'roft_blender_paper_aligned_v4/METHODOLOGY_AUDIT_20261012.md').exists(),
        ['METHODOLOGY_ALIGNMENT.md', '../roft_blender_paper_aligned_v4/METHODOLOGY_AUDIT_20261012.md'])
    check('Exactly256721actualbackwardtokens in verified numerical updates',
        chain['training_verified_complete'] and chain['backward_tokens'] == 256721, chain)
    check('Every selected target has exactly one completed native backward in the saved chain',
        backwards['verified'] and backwards['saved_backward_tokens'] == 256721 and
        backwards['pending_backward_tokens'] == 0 and
        all(s['completed_backwards'] == s['selected_targets'] and s['saved_numerical_update']
            for s in backwards['steps']), backwards)
    states = []
    cycles = []
    for step in chain['steps']:
        ordinal = step['step']
        state = read(root / 'audits/optimizer_states' / f'checkpoint-{ordinal:03d}.json')
        if ordinal == 1 and state is None:
            state = read(artifacts / 'recovered_optimizer_state_audit.json')
        states.append({'step': ordinal, 'receipt': state})
        cycles.append({'step': ordinal, 'receipt': read(artifacts / 'cycle_audits' / f'cycle-{ordinal:03d}-complete.json')})
    check('All currently saved full-parameter masters/Adam/RNG and serving weights audited',
        states and all(x['receipt'] and x['receipt']['verified'] and not x['receipt']['problems'] for x in states), states)
    check('All currently saved update cycles source/reflection/token evidence audited and backed up',
        cycles and all(x['receipt'] and x['receipt']['verified'] and
            x['receipt']['complete_raw_backup_verified'] and x['receipt']['fresh_reflections'] == 64 and
            x['receipt']['original_sources'] == 16 for x in cycles), cycles)
    check('Fresh34episodeROFT and34episodebase evaluation with frozen actualverifiers',
        paired['paired_full_set_verified'], paired)
    final_state = read(artifacts / 'roft-final-state-backup.json')
    current_state = read(artifacts / 'final_state_current_hash_audit.json')
    check('Complete final model/optimizer/RNG backed up off-node and hashverified',
        final_state and final_state['verified'] and final_state['full_weights_optimizer_rng'] and
        current_state and current_state['verified'] and current_state['full_weights_optimizer_rng'] and
        current_state['all_model_shards_hashed'] and current_state['complete_state_bytes_hashed'] and
        current_state['state_parse_proof_matched_by_full_bytes_sha256'] and
        current_state['at'] >= final_state['at'],
        {'backup_receipt': final_state, 'current_bytes_audit': current_state})
    raw = read(artifacts / 'raw_backup_audit.json')
    storage = read(artifacts / 'local_raw_storage_audit.json')
    check('Full run raw evidence backup verified',
        raw and raw['verified'] and not raw['missing_or_mismatched'] and raw['files'] > 0,
        {k: raw[k] for k in ('at', 'verified', 'files', 'missing_or_mismatched')} if raw else None)
    check('Actual original raw bytes verified in current direct or reversible compressed local storage',
        storage and storage['verified'] and storage['entire_run_complete_claimed'] and
        raw and storage['files'] == raw['files'] and storage['at'] >= raw['at'], storage)
    completed = read(root / 'worker_complete.json')
    check('Worker completed exact training and all fresh evaluation arms',
        completed and completed['token_equal_training'] and completed['fresh_evaluation_all_arms'], completed)
    report = report_audit(root)
    check('Reported rewards/counts/paired deltas/intervals reconcile with complete fresh actual verifiers',
        report['verified'] and report['fresh_full_set_verified'], report)
    keep = read(artifacts / 'node_kept_running.json')
    check('Node retention receipt says active and no automatic shutdown',
        keep and keep['state'] == 'active' and keep['automatic_shutdown'] is False, keep)
    check('Original bounded workwindow preserved across repairs',
        read(artifacts / 'work_window.json') == read(root / 'work_window.json'),
        read(artifacts / 'work_window.json'))
    window = read(artifacts / 'work_window.json')
    if window and window.get('extension_approved_at'):
        original = read(artifacts / 'work_window_original.json')
        original_remote = read(root / 'work_window_original.json')
        approval = read(artifacts / 'work_window_extension_approval.json')
        check('Explicit approved extension retains original work/allocation clocks',
            original and original == original_remote and approval and
            approval['user_response'] == 'yes' and
            approval['approved_deadline_unix'] == window['deadline_unix'] and
            original['started_unix'] == window['started_unix'] and
            original['allocation_launch_unix'] == window['allocation_launch_unix'] and
            original['deadline_unix'] == window['original_deadline_unix'] and
            approval['original_deadline_unix'] == original['deadline_unix'],
            {'original': original, 'approved_extension': approval, 'current': window})
    check('Original failed-update artifacts retained alongside repair evidence',
        (root / 'failure-before-reload-recovery.json').exists() and
        (root / 'train/roft/checkpoint-001/recovery_reload_receipt.json').exists(),
        ['failure-before-reload-recovery.json', 'train/roft/checkpoint-001/recovery_reload_receipt.json'])
    return {'requirements': requirements, 'all_requirements_verified': all(x['verified'] for x in requirements),
        'node_must_be_rechecked_live_before_goal_completion': True,
        'note': 'A positive receipt requires fresh live node verification and independent final artifact hash checks before marking the goal complete.'}


if __name__ == '__main__':
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
