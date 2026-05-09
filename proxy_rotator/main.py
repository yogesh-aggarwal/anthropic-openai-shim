import asyncio
import hashlib
import logging
import time

import aiohttp
from aiohttp import web

# Configuration
PROXY_LIST_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text&protocol=http&timeout=150"
REFRESH_INTERVAL = 60  # 5 minutes
LISTEN_PORT = 8080

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proxy-rotator")


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

        # Hash-based load balancing on Authorization header
        h = hashlib.sha256(
            auth_header.encode() if auth_header else b"anonymous"
        ).hexdigest()
        idx = int(h, 16) % len(self.proxies)
        return self.proxies[idx]

    async def refresh_loop(self):
        while True:
            await asyncio.sleep(REFRESH_INTERVAL)
            await self.fetch_proxies()


proxy_manager = ProxyManager()


async def handle_proxy(request: web.Request):
    auth_header = request.headers.get("Authorization", "")
    proxy_url = proxy_manager.get_proxy(auth_header)

    if not proxy_url:
        return web.Response(
            text="Service Unavailable: No proxies available", status=503
        )

    method = request.method
    url = request.url

    # Skip standard headers that should be regenerated or ignored
    exclude_headers = {
        "host",
        "content-length",
        "connection",
        "proxy-connection",
        "transfer-encoding",
    }
    headers = {
        k: v for k, v in request.headers.items() if k.lower() not in exclude_headers
    }

    try:
        body = await request.read()
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method,
                url,
                headers=headers,
                data=body,
                proxy=proxy_url,
                timeout=aiohttp.ClientTimeout(total=30),
                allow_redirects=False,
            ) as resp:
                logger.info(
                    f"Request: {method} {url} -> Proxy: {proxy_url} [Auth: {auth_header[:8]}...]"
                )
                resp_body = await resp.read()
                resp_headers = {
                    k: v
                    for k, v in resp.headers.items()
                    if k.lower() not in exclude_headers
                }
                logger.info(f"Response: {resp.status} for {url} via {proxy_url}")
                return web.Response(
                    body=resp_body, status=resp.status, headers=resp_headers
                )
    except Exception as e:
        logger.error(f"Proxy failure [{proxy_url}] for {url}: {e}")
        return web.Response(text=f"Bad Gateway: {str(e)}", status=502)


async def start_background_tasks(app):
    await proxy_manager.fetch_proxies()
    app["proxy_fetcher"] = asyncio.create_task(proxy_manager.refresh_loop())


async def cleanup_background_tasks(app):
    app["proxy_fetcher"].cancel()
    try:
        await app["proxy_fetcher"]
    except asyncio.CancelledError:
        pass


if __name__ == "__main__":
    app = web.Application()
    # Capture everything
    app.router.add_route("*", "/{tail:.*}", handle_proxy)
    app.on_startup.append(start_background_tasks)
    app.on_cleanup.append(cleanup_background_tasks)

    web.run_app(app, host="0.0.0.0", port=LISTEN_PORT, access_log=None)
