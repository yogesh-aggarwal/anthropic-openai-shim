# Anthropic API Adapter for LiteLLM

A powerful adapter that bridges Anthropic's API with LiteLLM's flexible model inference, enabling seamless integration of any LLM provider with Anthropic-compatible tools and applications.

## ✨ Features

- **🔄 Full Anthropic API Compatibility**: Drop-in replacement for Anthropic's `/v1/messages` endpoint
- **🚀 LiteLLM Backend**: Access 100+ LLM providers through a single, unified interface
- **🛠️ Tool Calling**: Complete support for function calling and tool use
- **⚡ Streaming Responses**: Real-time streaming with proper event handling
- **🔐 Secure Authentication**: API key-based authentication with encryption
- **🧠 Reasoning Support**: Handles thinking blocks and complex reasoning content
- **🐳 Docker Ready**: Containerized deployment with Docker Compose
- **📊 Monitoring**: Built-in logging and health checks

## 📋 Prerequisites

- Docker and Docker Compose (or Podman)
- API keys for your preferred LLM providers (OpenAI, Anthropic, etc.)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd anthropic-openai-shim
```

### 2. Environment Configuration

```bash
# Copy and configure environment variables
cp .env.example .env

# Edit .env with your API keys and settings
# LITELLM_MASTER_KEY=your_master_key_here
# LITELLM_SALT_KEY=your_salt_key_here
```

### 3. Model Configuration

```bash
# Copy and configure model settings
cp models.example.yaml models.yaml

# Edit models.yaml with your provider configurations
# Add your API keys and model preferences
```

### 4. Launch Services

```bash
# Start all services
docker compose up -d --build

# Check status
docker compose ps
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `LITELLM_MASTER_KEY` | Master API key for authentication | Yes |
| `LITELLM_SALT_KEY` | Salt key for encryption | Yes |
| `DATABASE_URL` | PostgreSQL connection string | No (auto-configured) |

### Model Configuration

Configure your models in `models.yaml`:

```yaml
- model_name: claude-3-5-sonnet-latest
  litellm_params:
    model: anthropic/claude-3-5-sonnet-20241022
    api_key: your_anthropic_key
  model_info:
    mode: chat
    input_cost_per_token: 0.000003
    output_cost_per_token: 0.000015

- model_name: gpt-4
  litellm_params:
    model: openai/gpt-4
    api_key: your_openai_key
```

## 📖 Usage

### Basic API Usage

Once running, your adapter is available at `http://localhost:4000`. Use it exactly like Anthropic's API:

```bash
curl http://localhost:4000/v1/messages \
  -H "x-api-key: YOUR_MASTER_KEY" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "claude-3-5-sonnet-latest",
    "max_tokens": 1000,
    "messages": [
      {
        "role": "user",
        "content": "Hello! How can I help you today?"
      }
    ]
  }'
```

### With Python Client

```python
import anthropic

client = anthropic.Anthropic(
    api_key="YOUR_MASTER_KEY",
    base_url="http://localhost:4000"
)

response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "Explain quantum computing simply"}
    ]
)

print(response.content[0].text)
```

### Tool Calling Example

```python
response = client.messages.create(
    model="claude-3-5-sonnet-latest",
    max_tokens=1000,
    messages=[
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    tools=[
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    ]
)
```

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Client Apps   │────│ Anthropic       │────│     LiteLLM      │
│ (Claude Desktop │    │   Adapter       │    │   Proxy/Gateway  │
│  Cursor, etc.)  │    │   (Port 4000)   │    │   (Port 8080)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │                        │
                                └────────────────────────┘
                                         │
                                ┌─────────────────┐
                                │   PostgreSQL    │
                                │   Database      │
                                └─────────────────┘
```

### Components

- **Anthropic Adapter** (`:4000`): FastAPI service that translates Anthropic API requests to OpenAI-compatible format
- **LiteLLM Gateway** (`:8080`): Core proxy service managing model routing, load balancing, and provider integrations
- **PostgreSQL Database**: Persistent storage for LiteLLM configuration, usage metrics, and caching

## 🛠️ Development

### Available Commands

```bash
# Service management
make start          # Start all services
make stop           # Stop all services
make restart        # Restart services
make status         # Show service status
make logs           # View all service logs

# Configuration
make reload         # Reload LiteLLM config without restart
make clear-db       # Reset database (⚠️  destructive)

# Development
make build          # Rebuild containers
make shell          # Access adapter container shell
make test           # Run tests
```

### Adding New Models

1. **Edit `models.yaml`**:
   ```yaml
   - model_name: your-custom-model
     litellm_params:
       model: provider/model-name
       api_key: your_api_key
       api_base: https://custom-endpoint.com
     model_info:
       mode: chat
       input_cost_per_token: 0.000001
       output_cost_per_token: 0.000002
   ```

2. **Reload configuration**:
   ```bash
   make reload
   ```

3. **Test the new model**:
   ```bash
   curl -X POST http://localhost:4000/v1/messages \
     -H "x-api-key: YOUR_MASTER_KEY" \
     -H "anthropic-version: 2023-06-01" \
     -d '{"model": "your-custom-model", "messages": [...]}'
   ```

### Supported Providers

LiteLLM supports 100+ providers including:
- OpenAI (GPT-4, GPT-3.5)
- Anthropic (Claude)
- Google (Gemini, PaLM)
- Meta (Llama)
- Mistral, Cohere, Hugging Face
- And many more...

## 🔍 Monitoring & Troubleshooting

### Health Checks

```bash
# Check service health
curl http://localhost:4000/health

# View LiteLLM status
curl http://localhost:8080/health
```

### Common Issues

**Connection Refused**: Ensure services are running with `docker compose ps`

**Authentication Failed**: Verify `LITELLM_MASTER_KEY` in your `.env` file

**Model Not Found**: Check `models.yaml` configuration and run `make reload`

**Rate Limiting**: Configure provider-specific rate limits in `models.yaml`

### Logs

```bash
# View all service logs
docker compose logs -f

# View specific service logs
docker compose logs -f anthropic-adapter
docker compose logs -f litellm
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch: `git checkout -b feature/amazing-feature`
3. **Make** your changes with clear commit messages
4. **Test** thoroughly - ensure all existing functionality works
5. **Submit** a pull request with a detailed description

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
make test

# Format code
make format
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

Built with ❤️ using:
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [LiteLLM](https://litellm.ai/) - Unified LLM API interface
- [Docker](https://www.docker.com/) - Containerization platform

---

**Need help?** Check the [issues](https://github.com/your-repo/issues) or start a discussion!