"""Continue only the existing frozen base evaluation after explicit extension."""
import fcntl
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import RUN, load, save, validate_prepared
from worker import evaluate, command


def main():
    approval = load(RUN / 'work_window_extension_approval.json')
    window = load(RUN / 'work_window.json')
    if approval['approved_deadline_unix'] != window['deadline_unix'] or time.time() >= window['deadline_unix']:
        raise RuntimeError('Explicit additional window is absent or expired')
    def stop(signum, frame):
        raise InterruptedError('Run supervisor requested stop; node retained')
    signal.signal(signal.SIGTERM, stop)
    with (RUN / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validate_prepared()
        if load(RUN / 'train/roft/complete.json')['backward_tokens'] != 256721:
            raise RuntimeError('Original complete training state differs')
        try:
            evaluate('base', None)
            command('report.py', training=False)
            save(RUN / 'worker_complete.json', {'at': time.time(),
                'token_equal_training': True, 'fresh_evaluation_all_arms': True,
                'keep_node_running': True, 'automatic_shutdown': False,
                'base_evaluation_resumed': True, 'training_repeated': False})
        except BaseException as error:
            import traceback
            save(RUN / 'failure.json', {'at': time.time(), 'type': type(error).__name__,
                'error': str(error), 'traceback': traceback.format_exc()[-10000:],
                'keep_node_running': True, 'training_repeated': False})
            raise


if __name__ == '__main__':
    main()
