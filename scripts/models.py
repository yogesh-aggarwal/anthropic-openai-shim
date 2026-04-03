#!/usr/bin/env python3
"""Discover free models using OpenRouter as source of truth and convert to LiteLLM models.yaml.

Pipeline:
1. Read providers.yaml
2. Fetch free models from OpenRouter (hardcoded) as the canonical list
3. For each other provider, fuzzy-match their models against OpenRouter's free list
4. Write models.config.yaml
5. Convert models.config.yaml to models.yaml with fallbacks, pricing, and OpenRouter metadata
"""

import httpx
import itertools
import yaml
from collections import defaultdict
from pathlib import Path

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"


class CostValue:
    """Custom YAML representer to output cost values without scientific notation."""

    def __init__(self, value):
        self.value = value

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:float", f"{data.value:.6f}")


def model_slug(model_id: str) -> str:
    """Return the model slug without the provider prefix."""
    return model_id.split("/", 1)[-1] if "/" in model_id else model_id


def openrouter_model_id(model_id: str) -> str:
    """Return the base OpenRouter model id from a provider model slug.

    For example, `openai/x-ai/grok-code-fast-1:optimized:free` becomes
    `x-ai/grok-code-fast-1`.
    """
    slug = model_slug(model_id)
    return slug.split(":", 1)[0]


def final_model_name(model_id: str) -> str:
    """Return model_name as text between the last '/' and first ':'."""
    slug = model_slug(model_id)
    after_last_slash = slug.rsplit("/", 1)[-1]
    name = after_last_slash.split(":", 1)[0]
    if name.endswith("-free"):
        return name[: -len("-free")]
    return name


def fetch_openrouter_models(api_key: str) -> list[dict]:
    """Fetch all models from OpenRouter and return parsed model data."""
    resp = httpx.get(
        OPENROUTER_MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


def fetch_provider_models(base: str, api_key: str) -> list[str]:
    """Fetch model IDs from an OpenAI-compatible /models endpoint."""
    url = f"{base.rstrip('/')}/models"
    resp = httpx.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [m["id"] for m in data.get("data", [])]


def fetch_provider_model_data(base: str, api_key: str) -> list[dict]:
    """Fetch full model objects from an OpenAI-compatible /models endpoint."""
    url = f"{base.rstrip('/')}/models"
    resp = httpx.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def fuzzy_match(provider_model: str, openrouter_free_ids: list[str]) -> str | None:
    """Fuzzy match a provider model ID against OpenRouter free model IDs.

    Strips common prefixes/suffixes and does substring matching.
    Returns the matched OpenRouter model ID or None.
    """
    normalized = provider_model.lower().strip()

    for or_id in openrouter_free_ids:
        or_normalized = or_id.lower().strip()

        if normalized == or_normalized:
            return or_id

        provider_base = normalized.split(":")[0]
        or_base = or_normalized.split(":")[0]

        if provider_base == or_base:
            return or_id

        if normalized in or_normalized or or_normalized in normalized:
            return or_id

        provider_parts = set(provider_base.replace("/", " ").replace("-", " ").split())
        or_parts = set(or_base.replace("/", " ").replace("-", " ").split())

        if provider_parts & or_parts and (
            provider_parts.issubset(or_parts) or or_parts.issubset(provider_parts)
        ):
            return or_id

    return None


def build_openrouter_lookup(openrouter_models: list[dict]) -> dict[str, dict]:
    """Build a lookup dict from model ID to full OpenRouter model metadata."""
    lookup = {}
    for model in openrouter_models:
        lookup[model["id"]] = model
    return lookup


def discover_free_models(providers: dict) -> tuple[dict, dict]:
    """Use OpenRouter as source of truth for free models, check other providers.

    Returns (config dict, openrouter metadata lookup dict).
    """
    openrouter_info = providers.get("openrouter", {})
    openrouter_keys = openrouter_info.get("keys", [])
    openrouter_prefix = openrouter_info.get("prefix", "openrouter")

    print("Fetching models from OpenRouter ...")
    openrouter_models_by_id: dict[str, dict] = {}
    for key in openrouter_keys:
        try:
            for model in fetch_openrouter_models(key):
                openrouter_models_by_id[model["id"]] = model
        except Exception as e:
            print(f"  Key failed: {e}")
            continue

    all_openrouter_models = list(openrouter_models_by_id.values())
    print(f"  Found {len(all_openrouter_models)} total models")

    free_models = [m for m in all_openrouter_models if "free" in m["id"].lower()]
    free_model_ids = [m["id"] for m in free_models]
    or_lookup = build_openrouter_lookup(all_openrouter_models)
    print(f"  {len(free_model_ids)} free models")

    results: dict = {}

    for provider_name, info in providers.items():
        base = info.get("base", "")
        keys = info.get("keys", [])
        explicit_models = info.get("models")

        if explicit_models:
            results[provider_name] = info
            continue

        if not base or not keys:
            print(f"Skipping {provider_name}: no base or keys")
            continue

        prefix = info.get("prefix", "openai")

        if provider_name == "openrouter":
            results[provider_name] = {
                "base": base,
                "models": sorted(f"{openrouter_prefix}/{m}" for m in free_model_ids),
                "keys": keys,
            }
            continue

        print(f"Checking {provider_name} against OpenRouter free models ...")
        provider_models_by_id: dict[str, dict] = {}
        for key in keys:
            try:
                for model in fetch_provider_model_data(base, key):
                    provider_models_by_id[model["id"]] = model
            except Exception as e:
                print(f"  Key failed: {e}")
                continue

        provider_models = list(provider_models_by_id.values())

        matched = []
        for model in provider_models:
            provider_model_id = model["id"]
            if "free" in provider_model_id.lower():
                matched.append(f"{prefix}/{provider_model_id}")
                continue

            match = fuzzy_match(provider_model_id, free_model_ids)
            if match:
                matched.append(f"{prefix}/{match}")

        if matched:
            results[provider_name] = {
                "base": base,
                "models": sorted(set(matched)),
                "keys": keys,
            }
            print(f"  {len(matched)} matching free models")
        else:
            print("  No matching free models")

    return results, or_lookup


def convert_models(config: dict, or_lookup: dict, output_path: str) -> None:
    """Convert models.config dict to models.yaml format with OpenRouter metadata."""
    grouped = defaultdict(list)

    for provider, info in config.items():
        models = list(dict.fromkeys(info.get("models", [])))
        api_base = info.get("base")
        api_keys = list(dict.fromkeys(info.get("keys", [])))

        for model, api_key in itertools.product(models, api_keys):
            unique_id = final_model_name(model)
            grouped[unique_id].append((model, api_base, api_key))

    model_list = []

    for unique_id, items in grouped.items():
        primary_model, primary_base, primary_key = items[0]
        fallbacks = items[1:]

        litellm_params = {
            "model": primary_model,
            "api_base": primary_base,
            "api_key": primary_key,
        }

        if fallbacks:
            litellm_params["fallbacks"] = [
                {
                    "model": m,
                    "api_base": b,
                    "api_key": k,
                }
                for m, b, k in fallbacks
            ]

        model_info: dict = {}

        or_model_id = openrouter_model_id(primary_model)
        if or_model_id in or_lookup:
            or_data = or_lookup[or_model_id]
            skip_keys = {"id", "pricing"}
            for key, value in or_data.items():
                if key in skip_keys or value is None:
                    continue
                if key == "architecture":
                    arch = value
                    for ak, av in arch.items():
                        if av is not None:
                            model_info[f"arch_{ak}"] = av
                elif key == "top_provider":
                    tp = value
                    for tk, tv in tp.items():
                        if tv is not None:
                            model_info[f"top_provider_{tk}"] = tv
                elif key == "default_parameters":
                    dp = value
                    for dk, dv in dp.items():
                        if dv is not None:
                            model_info[f"default_param_{dk}"] = dv
                elif key == "per_request_limits":
                    if value:
                        model_info["per_request_limits"] = value
                else:
                    model_info[key] = value

            model_info["mode"] = "chat"
            model_info["supports_vision"] = "image" in or_data.get(
                "architecture", {}
            ).get("input_modalities", [])

        model_info["input_cost_per_token"] = CostValue(0.000003)
        model_info["output_cost_per_token"] = CostValue(0.000015)

        model_entry = {
            "model_name": unique_id,
            "litellm_params": litellm_params,
            "model_info": model_info,
        }
        model_list.append(model_entry)

    output = {"model_list": model_list}

    yaml.add_representer(CostValue, CostValue.representer)

    with open(output_path, "w") as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


def main() -> None:
    script_dir = Path(__file__).parent.parent
    providers_file = script_dir / "providers.yaml"
    config_file = script_dir / "models.config.yaml"
    output_file = script_dir / "models.yaml"

    with open(providers_file) as f:
        providers = yaml.safe_load(f)

    config, or_lookup = discover_free_models(providers)

    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote models.config.yaml with {len(config)} providers")

    convert_models(config, or_lookup, str(output_file))
    print("Converted models ✌️")


if __name__ == "__main__":
    main()
