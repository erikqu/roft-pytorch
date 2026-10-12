"""AWS-signed Responses transport for the unchanged Luna judge rubric."""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from email.utils import parsedate_to_datetime
import urllib.error
import urllib.request

from bedrock_limits import BedrockLimits, token_reservation


def retry_after(headers, now):
    value = headers.get("retry-after") or headers.get("Retry-After")
    if value is None:
        return 0.0
    try:
        seconds = float(value)
        return max(0.0, seconds) if math.isfinite(seconds) else 0.0
    except ValueError:
        try:
            return max(0.0, parsedate_to_datetime(value).timestamp() - now)
        except (ValueError, TypeError, OverflowError):
            return 0.0


def transport_for(config, diagnostics, *, reserve_retry=None):
    import boto3
    from botocore.auth import SigV4Auth
    from botocore.awsrequest import AWSRequest
    from botocore.credentials import Credentials

    region = config["judge_aws_region"]
    if region != "us-east-1" or config["judge_bedrock_model"] != "us.openai.gpt-6-luna":
        raise ValueError("Unqualified Bedrock endpoint or model")
    credential_path = config.get("judge_aws_credentials_file")
    if credential_path:
        path = Path(credential_path).expanduser()
        if path.stat().st_mode & 0o077:
            raise RuntimeError("AWS judge credentials must be private to the host user")
        values = json.loads(path.read_text())
        credentials = Credentials(values["AccessKeyId"], values["SecretAccessKey"], values.get("SessionToken"))
    else:
        credentials = boto3.Session(region_name=region).get_credentials()
    if credentials is None:
        raise RuntimeError("AWS judge credentials are unavailable")
    endpoint = f"https://bedrock-runtime.{region}.amazonaws.com/openai/v1/responses"
    diagnostics = Path(diagnostics)
    diagnostics.mkdir(parents=True, exist_ok=True, mode=0o700)
    settings = config.get("judge_bedrock_limits", {})
    limits = BedrockLimits(diagnostics.parent / "bedrock-admission", settings)
    attempts_limit = settings.get("max_attempts", 6)
    elapsed_limit = settings.get("max_elapsed_seconds", 300)
    base_delay = settings.get("backoff_base_seconds", 1)
    maximum_delay = settings.get("backoff_max_seconds", 60)
    if (type(attempts_limit) is not int or not 1 <= attempts_limit <= 10
            or not 0 < elapsed_limit <= 900 or not 0 < base_delay <= maximum_delay <= 120):
        raise ValueError("Invalid bounded Bedrock retry policy")

    def transport(payload, unused_api_key):
        if payload["model"] != config["judge_bedrock_model"]:
            raise ValueError("Payload does not name the exact Bedrock inference profile")
        body = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        identity = hashlib.sha256(body).hexdigest()
        receipt = {"provider": "aws-bedrock", "region": region,
                   "inference_profile": payload["model"], "payload_sha256": identity, "attempts": []}
        deadline = time.monotonic() + elapsed_limit
        for attempt in range(1, attempts_limit + 1):
            started = time.time()
            entry = {"attempt": attempt, "started": started}
            delay = None
            try:
                with limits.admit(token_reservation(payload), deadline):
                    request = AWSRequest(method="POST", url=endpoint, data=body,
                                         headers={"Content-Type": "application/json"})
                    SigV4Auth(credentials.get_frozen_credentials(), "bedrock", region).add_auth(request)
                    http = urllib.request.Request(endpoint, data=body, headers=dict(request.headers), method="POST")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("Bedrock retry deadline exceeded before sending")
                    with urllib.request.urlopen(http, timeout=min(180, remaining)) as response:
                        data = json.load(response)
                        request_id = response.headers.get("x-amzn-requestid") or response.headers.get("x-request-id")
                        entry.update(http_status=response.status, request_id=request_id,
                                     response_id=data.get("id"), response_model=data.get("model"),
                                     response_status=data.get("status"), usage=data.get("usage"))
            except urllib.error.HTTPError as error:
                raw_error = error.read(8192).decode(errors="replace")
                entry.update(http_status=error.code, request_id=error.headers.get("x-amzn-requestid"),
                             error_body=raw_error)
                try:
                    detail = json.loads(raw_error)
                except ValueError:
                    detail = {}
                detail = detail if isinstance(detail, dict) else {}
                nested = detail.get("error") if isinstance(detail.get("error"), dict) else {}
                code = detail.get("code") or nested.get("code")
                throttled = error.code == 429 and code not in {
                    "ServiceQuotaExceededException", "insufficient_quota", "quota_exceeded"}
                retryable = ((throttled or error.code == 503)
                             and not detail.get("usage") and not detail.get("id")
                             and not nested.get("usage") and not nested.get("id"))
                if retryable:
                    delay = max(retry_after(error.headers, time.time()),
                                random.uniform(0, min(maximum_delay, base_delay * 2 ** (attempt - 1))))
                    limits.cooldown(delay)
                if not retryable or attempt == attempts_limit or time.monotonic() + delay >= deadline:
                    raise RuntimeError(f"Bedrock judge HTTP {error.code}; bounded retries unavailable or exhausted") from error
                if error.code == 503:
                    if reserve_retry is None:
                        raise RuntimeError("Cannot retry an uncertain 503 without a conservative charge reservation") from error
                    reserve_retry(payload, attempt)
                entry["retry_delay_seconds"] = delay
            except Exception as error:
                entry["error_type"] = type(error).__name__
                entry["automatic_retry"] = False
                raise
            finally:
                entry["ended"] = time.time()
                for key in ("error_body", "error_type", "automatic_retry", "retry_delay_seconds"):
                    receipt.pop(key, None)
                receipt.update(entry)
                receipt["attempts"].append(entry)
                path = diagnostics / f"{identity}.json"
                descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(descriptor, "w") as stream:
                    json.dump(receipt, stream, indent=2)
            if delay is None:
                break
            time.sleep(delay)
        allowed = {config["judge_model"], "openai." + config["judge_model"], config["judge_bedrock_model"]}
        if data.get("model") not in allowed or not request_id:
            raise RuntimeError("Bedrock response omitted or changed the qualified model/request identity")
        data["bedrock_provenance"] = receipt
        data["model"] = config["judge_model"]
        return data, request_id

    return transport
