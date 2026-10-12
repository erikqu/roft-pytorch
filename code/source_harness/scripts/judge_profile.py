"""Run-local host profile; published verifier files remain unchanged."""
import hashlib
import importlib.util
from pathlib import Path


def configured_judge(config):
    from refresh_blender_verifiers import terra

    effort = config.get("judge_reasoning_effort", "low")
    if effort == "low":
        return terra
    if effort != "xhigh" or config.get("judge_model") != "gpt-6-luna":
        raise ValueError("Unsupported run-local judge profile")
    source = Path(terra.__file__)
    if hashlib.sha256(source.read_bytes()).hexdigest() != config.get("judge_host_source_sha256"):
        raise RuntimeError("Trusted host judge source changed; requalify the run-local profile")
    spec = importlib.util.spec_from_file_location(
        "refresh_blender_verifiers._qwen_luna_xhigh", source)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the trusted host judge")
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    if config.get("judge_provider", "openai") == "aws-bedrock":
        if config.get("judge_bedrock_model") != "us.openai.gpt-6-luna":
            raise ValueError("Unqualified Bedrock inference profile")
        original_cost_metadata = judge._profile.cost_metadata

        def cost_metadata(model):
            metadata = original_cost_metadata(model)
            return {**metadata, "provider": "aws-bedrock", "region": config["judge_aws_region"],
                    "inference_profile": config["judge_bedrock_model"],
                    "rates": {key: value * 1.1 for key, value in metadata["rates"].items()},
                    "pricing_source": "AWS GPT-6 Luna model card, US Geo CRIS, 2026-09-29"}

        judge._profile.cost_metadata = cost_metadata
    elif config.get("judge_provider", "openai") == "azure":
        from azure_judge import ENDPOINT
        if config.get("judge_azure_endpoint", "").rstrip("/") != ENDPOINT:
            raise ValueError("Unqualified Azure endpoint")
        original_cost_metadata = judge._profile.cost_metadata

        def cost_metadata(model):
            metadata = original_cost_metadata(model)
            return {**metadata, "provider": "azure", "endpoint": ENDPOINT,
                    "deployment": config["judge_model"],
                    "pricing_source": "Conservative accounting at prior Bedrock rates; Azure invoice not yet reconciled",
                    "rates": {key: value * 1.1 for key, value in metadata["rates"].items()}}

        judge._profile.cost_metadata = cost_metadata
    elif config.get("judge_provider", "openai") != "openai":
        raise ValueError("Unsupported judge provider")
    original_request_for = judge.request_for

    def request_for(request, evidence, ordinal):
        if request.get("model") != config["judge_model"]:
            raise ValueError("Judge evidence model differs from the run profile")
        payload, maximum = original_request_for(request, evidence, ordinal)
        if payload.get("reasoning") != {"effort": "low"}:
            raise RuntimeError("Unexpected original judge reasoning profile")
        output_limit = config.get("judge_max_output_tokens", payload["max_output_tokens"])
        if type(output_limit) is not int or not payload["max_output_tokens"] <= output_limit <= 32768:
            raise ValueError("Invalid run-local judge output bound")
        output_rate = judge._profile.cost_metadata(request["model"])["rates"]["output"]
        maximum += (output_limit - payload["max_output_tokens"]) * output_rate / 1_000_000
        return {**payload, "reasoning": {"effort": effort},
                "max_output_tokens": output_limit,
                "model": config["judge_bedrock_model"] if config.get("judge_provider") == "aws-bedrock"
                else payload["model"]}, maximum

    judge.request_for = request_for
    return judge
