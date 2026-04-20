#!/usr/bin/env python3
"""Read models.yaml and produce opencode.json with model names."""

import json
import yaml
from pathlib import Path


def main() -> None:
    script_dir = Path(__file__).parent.parent
    input_file = script_dir / "models.yaml"
    output_file = script_dir / "opencode.json"

    with open(input_file) as f:
        data = yaml.safe_load(f)

    models_dict = {}

    for model_entry in data.get("model_list", []):
        model_id = model_entry.get("model_name")
        model_info = model_entry.get("model_info", {})
        name = model_info.get("name")

        if model_id and name:
            models_dict[model_id] = {"name": name}

    output = {"models": models_dict}

    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote {len(models_dict)} models to opencode.json")


if __name__ == "__main__":
    main()