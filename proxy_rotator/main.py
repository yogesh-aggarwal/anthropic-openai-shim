import asyncio
import hashlib
import logging
import time

import aiohttp
from aiohttp import web

# Configuration
PROXY_LIST_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text&protocol=http&timeout=150"
REFRESH_INTERVAL = 300
LISTEN_PORT = 8080

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proxy-rotator")

PROVIDERS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "kilo": "https://api.kilo.ai/api/gateway",
}


class ProxyManager:
    def __init__(self):
        self.proxies = []
        self.last_fetch = 0

    async def fetch_proxies(self):
        try:
            logger.info("Fetching fresh proxy list...")
            async with aiohttp.ClientSession() as session:
                async with session.get(PROXY_LIST_URL, timeout=30) as resp:
                    if resp.status != 200:
                        logger.error(f"Failed to fetch proxies: {resp.status}")
                        return
                    text = await resp.text()
                    new_proxies = []
                    for line in text.splitlines():
                        p = line.strip()
                        if p:
                            if not p.startswith("http"):
                                p = f"http://{p}"
                            new_proxies.append(p)

                    if new_proxies:
                        self.proxies = new_proxies
                        self.last_fetch = time.time()
                        logger.info(f"Successfully loaded {len(self.proxies)} proxies")
                    else:
                        logger.warning("Fetched proxy list is empty")
        except Exception as e:
            logger.error(f"Error fetching proxies: {e}")

    def get_proxy(self, auth_header: str) -> str:
        if not self.proxies:
            return None
        h = hashlib.sha256(
            auth_header.encode() if auth_header else b"anonymous"
        ).hexdigest()
        idx = int(h, 16) % len(self.proxies)
        return self.proxies[idx]


proxy_manager = ProxyManager()


async def handle_request(request):
    provider_key = request.match_info.get("provider")
    if provider_key not in PROVIDERS:
        # Fallback to OpenRouter if no provider specified
        upstream_base = PROVIDERS["openrouter"]
        path = f"{provider_key}/{request.match_info.get('tail', '')}".strip("/")
    else:
        upstream_base = PROVIDERS[provider_key]
        path = request.match_info.get("tail", "").strip("/")

    target_url = f"{upstream_base}/{path}"
    if request.query_string:
        target_url += f"?{request.query_string}"

    auth_header = request.headers.get("Authorization", "")
    proxy_url = proxy_manager.get_proxy(auth_header)

    logger.info(f"Forwarding to {target_url} via {proxy_url} [Auth: {auth_header}]")

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length")
    }

    try:
        # Use a long timeout for LLM calls
        timeout = aiohttp.ClientTimeout(total=300)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            # Read body
            body = await request.read()

            async with session.request(
                method=request.method,
                url=target_url,
                headers=headers,
                data=body,
                proxy=proxy_url,
                allow_redirects=True,
            ) as resp:
                response_body = await resp.read()
                return web.Response(
                    body=response_body,
                    status=resp.status,
                    headers={
                        k: v
                        for k, v in resp.headers.items()
                        if k.lower() not in ("transfer-encoding", "content-encoding")
                    },
                )
    except Exception as e:
        logger.error(f"Gateway error: {e}")
        return web.Response(text=f"Gateway Error: {str(e)}", status=502)


async def refresh_loop():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        await proxy_manager.fetch_proxies()


async def main():
    await proxy_manager.fetch_proxies()
    asyncio.create_task(refresh_loop())

    app = web.Application()
    app.add_routes(
        [
            web.route("*", "/{provider}/{tail:.*}", handle_request),
            web.route("*", "/{provider}", handle_request),
        ]
    )

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    logger.info("Reverse Proxy Gateway running on port 8080")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
