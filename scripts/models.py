#!/usr/bin/env python3
"""Convert models.config.yaml to models.yaml format."""

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


def convert_models(input_path: str, output_path: str) -> None:
    """Convert models.config.yaml to models.yaml format.

    Model names use unique model ID: model.split("/")[-1].split(":")[0]
    Each model is backed by all available providers as fallbacks.
    """
    with open(input_path, "r") as f:
        data = yaml.safe_load(f)

    grouped = defaultdict(list)

    # Process each provider and group by unique model ID
    for provider, info in data.items():
        # Deduplicate models while preserving order
        models = list(dict.fromkeys(info.get("models", [])))
        api_base = info.get("base")
        # Deduplicate keys while preserving order
        api_keys = list(dict.fromkeys(info.get("keys", [])))

        # Generate all combinations of models and keys (Cartesian product)
        for model, api_key in itertools.product(models, api_keys):
            unique_id = model.split("/")[-1].split(":")[0]
            grouped[unique_id].append((model, api_base, api_key))

    model_list = []

    # Create model entries with fallbacks
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


if __name__ == "__main__":
    script_dir = Path(__file__).parent.parent
    input_file = script_dir / "models.config.yaml"
    output_file = script_dir / "models.yaml"

    convert_models(str(input_file), str(output_file))
    print("Converted models ✌️")
