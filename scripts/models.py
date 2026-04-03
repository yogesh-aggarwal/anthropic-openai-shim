#!/usr/bin/env python3
"""Convert models.config.yaml to models.yaml format."""

import itertools
import yaml
from pathlib import Path


def convert_models(input_path: str, output_path: str) -> None:
    """Convert models.config.yaml to models.yaml format.

    Model names use format: provider:models.items.split("/")[-1]
    e.g., kilo.openai/x-ai/grok-code-fast-1:optimized:free -> kilo:grok-code-fast-1:optimized:free
    """
    with open(input_path, 'r') as f:
        data = yaml.safe_load(f)

    model_list = []

    # Process each provider
    for provider, info in data.items():
        # Deduplicate models while preserving order
        models = list(dict.fromkeys(info.get('models', [])))
        api_base = info.get('base')
        # Deduplicate keys while preserving order
        api_keys = list(dict.fromkeys(info.get('keys', [])))

        # Generate all combinations of models and keys (Cartesian product)
        for model, api_key in itertools.product(models, api_keys):
            # Split on first '/' to separate provider prefix from rest
            # e.g., "openai/x-ai/grok-code-fast-1:optimized:free"
            parts = model.split('/', 1)
            if len(parts) == 2:
                # Take the last component after splitting by '/'
                model_suffix = parts[1].split('/')[-1]
            else:
                model_suffix = model

            # Create model_name in format: provider:model_suffix
            model_name = f"{provider}:{model_suffix}"

            model_entry = {
                'model_name': model_name,
                'litellm_params': {
                    'model': model,
                    'api_base': api_base,
                    'api_key': api_key,
                }
            }
            model_list.append(model_entry)

    output = {'model_list': model_list}

    with open(output_path, 'w') as f:
        yaml.dump(output, f, default_flow_style=False, sort_keys=False)


if __name__ == '__main__':
    script_dir = Path(__file__).parent.parent
    input_file = script_dir / 'models.config.yaml'
    output_file = script_dir / 'models.yaml'

    convert_models(str(input_file), str(output_file))
    print("Converted models ✌️")
