"""Azure Responses transport with host-only renewable Entra credentials."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import stat
import time
import urllib.error
import urllib.parse
import urllib.request

from bedrock_judge import retry_after
from bedrock_limits import BedrockLimits, token_reservation


ENDPOINT = os.environ.get('ROFT_AZURE_ENDPOINT', '').rstrip('/')


def private_json(path):
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW)) as stream:
        info = os.fstat(stream.fileno())
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or not stat.S_ISREG(info.st_mode):
            raise RuntimeError("Azure credentials/cache must be owned by this user and mode 0600")
        return json.load(stream)


def atomic_private(path, value):
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class AzureToken:
    def __init__(self, credentials, directory):
        self.credentials = Path(credentials).expanduser()
        self.directory = Path(directory)
        self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise RuntimeError("Azure token directory must be private to the host user")

    def __call__(self):
        import fcntl
        with (self.directory / "token.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            path = self.directory / "token.json"
            cached = private_json(path) if path.exists() else {}
            if cached.get("expires_at", 0) > time.time() + 300 and cached.get("access_token"):
                return cached["access_token"]
            values = private_json(self.credentials)
            tenant = urllib.parse.quote(values["AZURE_TENANT_ID"], safe="")
            body = urllib.parse.urlencode({
                "client_id": values["AZURE_CLIENT_ID"], "client_secret": values["AZURE_CLIENT_SECRET"],
                "grant_type": "client_credentials", "scope": "https://cognitiveservices.azure.com/.default",
            }).encode()
            request = urllib.request.Request(
                f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token", data=body)
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.load(response)
            except Exception as error:
                raise RuntimeError(f"Azure token acquisition failed ({type(error).__name__})") from None
            token, ttl = data.get("access_token"), int(data.get("expires_in", 0))
            if not isinstance(token, str) or not token or ttl <= 300:
                raise RuntimeError("Azure returned an unusable access token")
            atomic_private(path, {"access_token": token, "expires_at": time.time() + ttl})
            return token


def azure_retry_after(headers):
    headers = {key.lower(): value for key, value in headers.items()}
    seconds = retry_after(headers, time.time())
    try:
        milliseconds = float(headers.get("retry-after-ms", "0"))
        if math.isfinite(milliseconds):
            seconds = max(seconds, milliseconds / 1000)
    except ValueError:
        pass
    return max(0, seconds)


def transport_for(config, diagnostics, *, reserve_retry=None, token_provider=None):
    endpoint = config["judge_azure_endpoint"].rstrip("/")
    if not ENDPOINT or endpoint != ENDPOINT or config["judge_model"] != "gpt-6-luna":
        raise ValueError("Unqualified Azure judge endpoint or model")
    diagnostics = Path(diagnostics)
    diagnostics.mkdir(parents=True, exist_ok=True, mode=0o700)
    token_provider = token_provider or AzureToken(
        config["judge_azure_credentials_file"], diagnostics.parent / "azure-auth")
    settings = config["judge_azure_limits"]
    # The file-lock admission implementation is provider independent. Azure has
    # its own directory, concurrency, rolling token budget and shared cooldown.
    limits = BedrockLimits(diagnostics.parent / "azure-admission", settings)
    attempts_limit = settings.get("max_attempts", 6)
    elapsed_limit = settings.get("max_elapsed_seconds", 600)
    base_delay = settings.get("backoff_base_seconds", 1)
    max_delay = settings.get("backoff_max_seconds", 60)
    if (type(attempts_limit) is not int or not 1 <= attempts_limit <= 10
            or not 0 < elapsed_limit <= 900 or not 0 < base_delay <= max_delay <= 120):
        raise ValueError("Invalid bounded Azure retry policy")

    def transport(payload, unused_api_key):
        if payload["model"] != config["judge_model"]:
            raise ValueError("Payload differs from the qualified Azure judge model")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        identity = hashlib.sha256(body).hexdigest()
        receipt = {"provider": "azure", "endpoint": endpoint, "deployment": payload["model"],
                   "payload_sha256": identity, "attempts": []}
        receipt_path = diagnostics / f"{identity}-{time.time_ns()}.json"
        deadline = time.monotonic() + elapsed_limit
        for attempt in range(1, attempts_limit + 1):
            entry = {"attempt": attempt, "started": time.time()}
            delay = None
            try:
                with limits.admit(token_reservation(payload), deadline):
                    token = token_provider()
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Azure deadline exceeded before sending")
                    http = urllib.request.Request(endpoint + "/responses", data=body, headers={
                        "Content-Type": "application/json", "Authorization": "Bearer " + token})
                    # Reasoning may legitimately outlast three minutes. Bound
                    # the read by this pass's existing overall deadline.
                    with urllib.request.urlopen(http, timeout=remaining) as response:
                        data = json.load(response)
                        headers = {key.lower(): value for key, value in response.headers.items()}
                        request_id = headers.get("x-request-id") or headers.get("apim-request-id")
                        entry.update(http_status=response.status, request_id=request_id,
                                     response_id=data.get("id"), response_model=data.get("model"),
                                     response_status=data.get("status"), usage=data.get("usage"),
                                     rate_limits={k: v for k, v in headers.items() if k.startswith("x-ratelimit-")})
            except urllib.error.HTTPError as error:
                raw = error.read(8192)
                try:
                    detail = json.loads(raw)
                except ValueError:
                    detail = {}
                detail = detail if isinstance(detail, dict) else {}
                nested = detail.get("error") if isinstance(detail.get("error"), dict) else {}
                code = nested.get("code") or detail.get("code")
                entry.update(http_status=error.code, request_id=error.headers.get("apim-request-id"),
                             error_code=str(code)[:120] if code else None,
                             error_body_sha256=hashlib.sha256(raw).hexdigest())
                retryable = (error.code in (429, 500, 502, 503, 504)
                             and code not in {"insufficient_quota", "quota_exceeded", "BillingHardLimitReached"}
                             and not any(d.get("usage") or d.get("id") for d in (detail, nested)))
                delay = max(azure_retry_after(error.headers),
                            random.uniform(0, min(max_delay, base_delay * 2 ** (attempt - 1))))
                if not retryable or attempt == attempts_limit or time.monotonic() + delay >= deadline:
                    raise RuntimeError(f"Azure judge HTTP {error.code}; bounded retries unavailable or exhausted") from None
                if error.code != 429:
                    if reserve_retry is None:
                        raise RuntimeError("Uncertain Azure retry requires a conservative charge reservation") from None
                    reserve_retry(payload, attempt)
                limits.cooldown(delay)
                entry["retry_delay_seconds"] = delay
            except Exception as error:
                entry.update(error_type=type(error).__name__, automatic_retry=False)
                raise
            finally:
                entry["ended"] = time.time()
                receipt["attempts"].append(entry)
                atomic_private(receipt_path, receipt)
            if delay is None:
                break
            time.sleep(delay)
        if data.get("model") != config["judge_model"] or not request_id:
            raise RuntimeError("Azure response omitted or changed the qualified model/request identity")
        data["azure_provenance"] = receipt
        return data, request_id

    return transport
