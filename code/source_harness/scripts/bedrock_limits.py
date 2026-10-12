"""Cross-process admission limits and shared cooldown for Bedrock judging."""
import base64
from contextlib import contextmanager
import fcntl
import json
import math
from pathlib import Path
import struct
import time


def token_reservation(payload):
    total = len(payload.get("instructions", "").encode()) + 2048
    total += len(json.dumps(payload.get("text", {})).encode())
    for message in payload.get("input", []):
        for part in message.get("content", []):
            if part.get("type") == "input_text":
                total += len(part["text"].encode())
            elif part.get("type") == "input_image":
                prefix, encoded = part["image_url"].split(",", 1)
                if prefix != "data:image/png;base64":
                    raise ValueError("Unqualified image format for judge quota estimation")
                raw = base64.b64decode(encoded, validate=True)
                if len(raw) < 24 or raw[:8] != b"\x89PNG\r\n\x1a\n":
                    raise ValueError("Invalid PNG for judge quota estimation")
                width, height = struct.unpack(">II", raw[16:24])
                total += math.ceil(width / 32) * math.ceil(height / 32) * 4
    return total + payload.get("max_output_tokens", 0)


class BedrockLimits:
    def __init__(self, directory, settings):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.concurrent = settings.get("max_concurrent", 4)
        self.requests_per_minute = settings.get("requests_per_minute", 30)
        self.tokens_per_minute = settings.get("tokens_per_minute", 1_000_000)
        for value in (self.concurrent, self.requests_per_minute, self.tokens_per_minute):
            if type(value) is not int or value <= 0:
                raise ValueError("Bedrock admission limits must be positive integers")

    @contextmanager
    def state(self):
        with (self.directory / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / "state.json"
            state = json.loads(path.read_text()) if path.exists() else {"requests": [], "cooldown_until": 0}
            yield state
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(state))
            temporary.replace(path)

    def cooldown(self, seconds):
        with self.state() as state:
            state["cooldown_until"] = max(state["cooldown_until"], time.time() + seconds)

    def try_reserve(self, tokens):
        if tokens > self.tokens_per_minute:
            raise ValueError("One judge request exceeds the configured token admission limit")
        with self.state() as state:
            now = time.time()
            state["requests"] = [entry for entry in state["requests"] if entry[0] > now - 60]
            if state["cooldown_until"] > now:
                return state["cooldown_until"] - now
            entries = state["requests"]
            if (len(entries) >= self.requests_per_minute
                    or sum(entry[1] for entry in entries) + tokens > self.tokens_per_minute):
                return max(0.01, entries[0][0] + 60 - now)
            entries.append([now, tokens])
            return 0

    @contextmanager
    def admit(self, tokens, deadline):
        while time.monotonic() < deadline:
            delay = 0.1
            for index in range(self.concurrent):
                slot = (self.directory / f"slot-{index}.lock").open("a")
                try:
                    try:
                        fcntl.flock(slot, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    delay = self.try_reserve(tokens)
                    if delay == 0:
                        yield
                        return
                    break
                finally:
                    slot.close()
            remaining = deadline - time.monotonic()
            if remaining > 0:
                time.sleep(min(max(delay, 0.01), 1.0, remaining))
        raise TimeoutError("Bedrock admission deadline exceeded before sending a request")
