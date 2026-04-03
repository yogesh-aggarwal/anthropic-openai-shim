#!/usr/bin/env python3
"""Discover free models and convert to LiteLLM models.yaml format.

Pipeline:
1. Read providers.yaml
2. For providers without explicit models, fetch {base}/models and filter for "free"
3. Write models.config.yaml
4. Convert models.config.yaml to models.yaml with fallbacks and pricing
"""

import httpx
import itertools
import yaml
from collections import defaultdict
from pathlib import Path


class CostValue:
    """Custom YAML representer to output cost values without scientific notation."""

    def __init__(self, value):
        self.value = value

    @staticmethod
    def representer(dumper, data):
        return dumper.represent_scalar("tag:yaml.org,2002:float", f"{data.value:.6f}")


def fetch_models(base: str, api_key: str) -> list[str]:
    """Fetch available models from an OpenAI-compatible /models endpoint."""
    url = f"{base.rstrip('/')}/models"
    resp = httpx.get(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return [m["id"] for m in data.get("data", [])]


def discover_free_models(providers: dict) -> dict:
    """For providers without a models key, discover free models via API."""
    results = {}

    for provider_name, info in providers.items():
        base = info.get("base", "")
        keys = info.get("keys", [])
        models = info.get("models")

        if models:
            results[provider_name] = info
            continue

        if not base or not keys:
            print(f"Skipping {provider_name}: no base or keys")
            continue

        print(f"Discovering models for {provider_name} at {base}/models ...")

        free_models = set()
        for key in keys:
            try:
                all_models = fetch_models(base, key)
                for model_id in all_models:
                    if "free" in model_id.lower():
                        prefix = info.get("prefix", "openai")
                        free_models.add(f"{prefix}/{model_id}")
                print(
                    f"  Found {len(free_models)} free models (using first working key)"
                )
                break
            except Exception as e:
                print(f"  Key failed: {e}")
                continue

        if free_models:
            results[provider_name] = {
                "base": base,
                "models": sorted(free_models),
                "keys": keys,
            }
        else:
            print(f"  No free models found for {provider_name}")

    return results


def convert_models(config: dict, output_path: str) -> None:
    """Convert models.config dict to models.yaml format."""
    grouped = defaultdict(list)

    for provider, info in config.items():
        models = list(dict.fromkeys(info.get("models", [])))
        api_base = info.get("base")
        api_keys = list(dict.fromkeys(info.get("keys", [])))

        for model, api_key in itertools.product(models, api_keys):
            unique_id = model.split("/")[-1].split(":")[0]
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

        model_entry = {
            "model_name": unique_id,
            "litellm_params": litellm_params,
            "model_info": {
                "input_cost_per_token": CostValue(0.000003),
                "output_cost_per_token": CostValue(0.000015),
            },
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

    config = discover_free_models(providers)

    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"\nWrote models.config.yaml with {len(config)} providers")

    convert_models(config, str(output_file))
    print("Converted models ✌️")


if __name__ == "__main__":
    main()
