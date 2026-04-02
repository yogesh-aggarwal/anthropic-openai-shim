# Anthropic-Compatible Skin over LiteLLM

This setup exposes an Anthropic-compatible `/v1/messages` API for Anthropic-only tools, backed by LiteLLM.

## Architecture

- `anthropic-adapter` on `http://localhost:4000`
- `litellm` management/API on `http://localhost:8080`
- `postgres` for LiteLLM state

## Single Source of Truth

Base gateway settings live in `litellm_config.yaml`.
Model entries live in `models.yaml` (gitignored).

To deploy a new inference:
1. Copy the example model file once:
   ```bash
   cp models.example.yaml models.yaml
   ```
2. Edit `models.yaml` (`model_list`, `litellm_params.model`, `api_base`, `api_key`).
3. If needed, edit gateway-wide settings in `litellm_config.yaml`.
4. Run:

```bash
make reload
```

No adapter-side model mapping is used.

## 1) Configure environment

```bash
cp .env.example .env
```

Set in `.env`:
- `LITELLM_MASTER_KEY`
- `LITELLM_SALT_KEY`

## 2) Start services

```bash
docker compose up -d --build
```

## 3) Point your Anthropic-only tool

- Base URL: `http://localhost:4000`
- API key: `LITELLM_MASTER_KEY`
- Endpoint: `/v1/messages`

Use a model name that exists in `models.yaml` `model_list`.

## Quick test

```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: $LITELLM_MASTER_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "max_tokens": 120,
    "messages": [{"role": "user", "content": "Say hello in one sentence."}]
  }'
```

## Useful commands

```bash
make start
make reload
make clear-db
make status
make logs
```
