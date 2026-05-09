#!/usr/bin/env python3
"""Discover free models per-provider and convert to LiteLLM models.yaml.

Pipeline:
1. Read providers.yaml
2. For each provider, fetch their models and identify free ones locally
3. Standardize model names (strip provider prefixes, colons, :free suffixes)
4. Write models.config.yaml with standardized model IDs
5. Convert models.config.yaml to models.yaml with OpenRouter metadata enrichment
"""

import httpx
import itertools
import yaml
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from rich.console import Console
from rich.live import Live
from rich.text import Text

OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

console = Console(force_terminal=True)


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


def standardize_model_id(provider_prefix: str, model_id: str) -> str:
    """Standardize a model ID by stripping provider prefix, colons, and :free suffixes.

    This ensures models from different providers with similar names can be
    grouped together in LiteLLM deployment.

    Example: With provider_prefix="openai", model_id="openai/gpt-4o:free"
    returns "openai/gpt-4o".
    """
    # Start with the full model ID
    result = model_id

    # Remove provider prefix if present
    if result.startswith(f"{provider_prefix}/"):
        result = result[len(provider_prefix) + 1:]

    # Extract the slug part before any colons
    result = result.split(":", 1)[0]

    # Remove -free suffix
    if result.endswith("-free"):
        result = result[: -len("-free")]

    return result


def final_model_name(model_id: str) -> str:
    """Return model_name as text between the last '/' and first ':'."""
    slug = model_slug(model_id)
    after_last_slash = slug.rsplit("/", 1)[-1]
    name = after_last_slash.split(":", 1)[0]
    if name.endswith("-free"):
        return name[: -len("-free")]
    return name


def generate_model_display_name(model_id: str) -> str:
    """Generate display name from model ID using naming rules.

    1. Replace all - with spaces
    2. For each word: if alphabet count <= 2, uppercase the whole word;
       otherwise capitalize just the first letter
    """
    slug = final_model_name(model_id)
    words = slug.replace("-", " ").split()

    result_words = []
    for word in words:
        alpha_count = sum(1 for c in word if c.isalpha())
        if alpha_count <= 2:
            result_words.append(word.upper())
        else:
            result_words.append(word.capitalize())

    final =  " ".join(result_words)
    if final.endswith(" IT"):
        final = final.rpartition(" IT")[0]
    return final


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
    """Build a lookup dict from standardized model name (final_model_name) to OpenRouter metadata."""
    lookup = {}
    for model in openrouter_models:
        key = final_model_name(model["id"])
        lookup[key] = model
    return lookup


def is_model_free(model: dict, model_id: str) -> bool:
    """Check if a model is free based on ID or pricing."""
    model_id_lower = model_id.lower()
    pricing = model.get("pricing", {})

    # Check if ID contains "free"
    if "free" in model_id_lower:
        return True

    # Check if pricing is explicitly 0 for both prompt and completion
    if (
        pricing.get("prompt") == "0"
        and pricing.get("completion") == "0"
    ):
        return True

    return False


def discover_free_models(providers: dict) -> tuple[dict, dict]:
    """Fetch models from each provider independently and identify free ones.

    Returns (config dict, openrouter metadata lookup dict).
    """
    openrouter_info = providers.get("openrouter", {})
    openrouter_keys = openrouter_info.get("keys", [])
    openrouter_prefix = openrouter_info.get("prefix", "openrouter")

    console.print("[bold blue]Fetching OpenRouter models for metadata...[/]")
    openrouter_models_by_id: dict[str, dict] = {}
    for key in openrouter_keys:
        try:
            for model in fetch_openrouter_models(key):
                openrouter_models_by_id[model["id"]] = model
        except Exception as e:
            console.print(f"  [red]Key failed: {e}[/]")
            continue

    all_openrouter_models = list(openrouter_models_by_id.values())
    console.print(f"  [green]Found {len(all_openrouter_models)} total OpenRouter models[/]")

    # Build OpenRouter lookup using standardized names
    or_lookup = build_openrouter_lookup(all_openrouter_models)

    results: dict = {}

    # Identify free OpenRouter models and add them to results
    openrouter_matched = []
    for model in all_openrouter_models:
        model_id = model["id"]
        if is_model_free(model, model_id):
            # Prefix with openrouter/ for litellm model ID
            openrouter_matched.append(f"openrouter/{model_id}")

    if openrouter_matched:
        results["openrouter"] = {
            "base": providers["openrouter"]["base"],
            "models": sorted(set(openrouter_matched)),
            "keys": providers["openrouter"]["keys"],
        }
        console.print(f"  [cyan]{len(openrouter_matched)} free OpenRouter models[/]")

    def check_provider(provider_name: str, info: dict) -> tuple[str, dict | None, int]:
        base = info.get("base", "")
        keys = info.get("keys", [])
        explicit_models = info.get("models")
        prefix = info.get("prefix", provider_name)

        if explicit_models:
            return provider_name, info, len(explicit_models) if isinstance(explicit_models, list) else 0

        if not base or not keys:
            console.print(f"[yellow]Skipping {provider_name}: no base or keys[/]")
            return provider_name, None, 0

        # Fetch all models for this provider
        provider_models_by_id: dict[str, dict] = {}
        for key in keys:
            try:
                for model in fetch_provider_model_data(base, key):
                    provider_models_by_id[model["id"]] = model
            except Exception:
                continue

        provider_models = list(provider_models_by_id.values())
        matched = []

        for model in provider_models:
            model_id = model["id"]
            if is_model_free(model, model_id):
                # Prefix with provider prefix for litellm model ID
                matched.append(f"{prefix}/{model_id}")

        if matched:
            result = {
                "base": base,
                "models": sorted(set(matched)),
                "keys": keys,
            }
            return provider_name, result, len(matched)
        else:
            return provider_name, None, 0

    providers_to_check = [
        (name, info)
        for name, info in providers.items()
        if name != "openrouter" and not info.get("models")
    ]

    status_updates = {}

    def render_todo():
        parts = []
        for name, _ in providers_to_check:
            status_text, count_val = status_updates.get(name, (None, 0))
            if status_text is None:
                parts.append(f"[yellow]○[/] {name}\t[yellow dim]Fetching...[/]")
            elif count_val > 0:
                parts.append(f"[green]✓[/] {name}\t[green]{count_val} models[/]")
            else:
                parts.append(f"[dim]✗[/] {name}\t[dim]No free models[/]")

        output = Text()
        for i, part in enumerate(parts):
            if i > 0:
                output.append("\n")
            output.append(Text.from_markup(part))
        return output

    console.print()
    with Live(render_todo(), console=console, refresh_per_second=4) as live:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(check_provider, name, info): name
                for name, info in providers_to_check
            }

            for future in as_completed(futures):
                provider_name, result, count = future.result()
                if result is not None:
                    results[provider_name] = result
                    status = f"[green]✓ {count} models[/]"
                else:
                    status = "[dim]No free models[/]"
                status_updates[provider_name] = (status, count)

                live.update(render_todo())

    for name, info in providers.items():
        if info.get("models"):
            results[name] = info

    return results, or_lookup


def convert_models(config: dict, or_lookup: dict, output_path: str, name_overrides: dict | None = None) -> None:
    """Convert models.config dict to models.yaml format with OpenRouter metadata."""
    if name_overrides is None:
        name_overrides = {}

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

        # Enrich with OpenRouter metadata if available
        if unique_id in or_lookup:
            or_data = or_lookup[unique_id]
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

        if unique_id in name_overrides:
            model_info["name"] = name_overrides[unique_id]
        else:
            model_info["name"] = generate_model_display_name(unique_id)

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
    names_file = script_dir / "models.names.yaml"

    with open(providers_file) as f:
        providers = yaml.safe_load(f)

    name_overrides = {}
    if names_file.exists():
        with open(names_file) as f:
            data = yaml.safe_load(f)
            if data:
                for key, value in data.items():
                    name_overrides[key] = value

    config, or_lookup = discover_free_models(providers)

    with open(config_file, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    console.print(f"\n[bold green]Wrote models.config.yaml with {len(config)} providers[/]")

    convert_models(config, or_lookup, str(output_file), name_overrides)
    console.print("[bold green]Converted models ✌️[/]")


if __name__ == "__main__":
    main()
