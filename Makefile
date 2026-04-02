.PHONY: start stop restart restart-proxy status logs

start:
	@docker compose up -d --build

stop:
	@docker compose down

restart:
	@docker compose down
	@docker compose up -d --build

status:
	@docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'

logs:
	@docker compose logs -f anthropic-adapter litellm
