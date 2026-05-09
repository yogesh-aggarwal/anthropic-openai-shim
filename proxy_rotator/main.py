import asyncio
import logging
import random
import time
from typing import Dict, List

import aiohttp
from aiohttp import web

# Configuration
PROXY_LIST_URL = "https://api.proxyscrape.com/v4/free-proxy-list/get?request=display_proxies&proxy_format=protocolipport&format=text&protocol=http&timeout=150"
REFRESH_INTERVAL = 300
MAX_RETRIES = 10
INITIAL_SCORE = 100
FAILURE_PENALTY = 50
SUCCESS_REWARD = 10

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("proxy-rotator")

PROVIDERS = {
    "openrouter": "https://openrouter.ai/api/v1",
    "kilo": "https://api.kilo.ai/api/gateway",
}


class ProxyStats:
    def __init__(self, url: str):
        self.url = url
        self.score = INITIAL_SCORE
        self.avg_latency = 0.0
        self.total_calls = 0
        self.failures = 0

    def record_success(self, latency: float):
        self.total_calls += 1
        self.score = min(200, self.score + SUCCESS_REWARD)
        # EMA for latency (alpha=0.3)
        if self.avg_latency == 0:
            self.avg_latency = latency
        else:
            self.avg_latency = (self.avg_latency * 0.7) + (latency * 0.3)

    def record_failure(self):
        self.total_calls += 1
        self.failures += 1
        self.score = max(0, self.score - FAILURE_PENALTY)


class ProxyManager:
    def __init__(self):
        self.proxies: Dict[str, ProxyStats] = {}
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
                    new_urls = []
                    for line in text.splitlines():
                        p = line.strip()
                        if p:
                            if not p.startswith("http"):
                                p = f"http://{p}"
                            new_urls.append(p)

                    if new_urls:
                        # Keep stats for existing proxies, add new ones
                        current_urls = set(self.proxies.keys())
                        for url in new_urls:
                            if url not in self.proxies:
                                self.proxies[url] = ProxyStats(url)

                        # Cleanup dead proxies (not in newest list)
                        for url in list(self.proxies.keys()):
                            if url not in new_urls:
                                del self.proxies[url]

                        self.last_fetch = time.time()
                        logger.info(
                            f"Successfully loaded {len(self.proxies)} proxies. Stats preserved."
                        )
                    else:
                        logger.warning("Fetched proxy list is empty")
        except Exception as e:
            logger.error(f"Error fetching proxies: {e}")

    def get_proxy(self, auth_header: str, attempt: int = 0) -> str:
        if not self.proxies:
            return None

        # Sort proxies by score (desc) and then latency (asc)
        # Proxies with score 0 are relegated to the very end
        sorted_proxies = sorted(
            self.proxies.values(),
            key=lambda p: (p.score, -p.avg_latency if p.avg_latency > 0 else -999),
            reverse=True,
        )

        # On first few attempts, try the cream of the crop
        # On later attempts, add some randomness to explore other potential proxies
        if attempt < 3:
            return sorted_proxies[attempt % len(sorted_proxies)].url
        else:
            # Pick from the top 50% randomly to avoid hammering the same "good" but slow proxies
            top_half = sorted_proxies[: max(1, len(sorted_proxies) // 2)]
            return random.choice(top_half).url

    def report_outcome(self, url: str, success: bool, latency: float = 0):
        if url in self.proxies:
            if success:
                self.proxies[url].record_success(latency)
            else:
                self.proxies[url].record_failure()


proxy_manager = ProxyManager()


async def handle_request(request):
    provider_key = request.match_info.get("provider")
    if provider_key not in PROVIDERS:
        upstream_base = PROVIDERS["openrouter"]
        path = f"{provider_key}/{request.match_info.get('tail', '')}".strip("/")
    else:
        upstream_base = PROVIDERS[provider_key]
        path = request.match_info.get("tail", "").strip("/")

    target_url = f"{upstream_base}/{path}"
    if request.query_string:
        target_url += f"?{request.query_string}"

    auth_header = request.headers.get("Authorization", "")
    body = await request.read()

    headers = {
        k: v
        for k, v in request.headers.items()
        if k.lower() not in ("host", "content-length", "expect")
    }

    last_error = None
    for attempt in range(MAX_RETRIES):
        proxy_url = proxy_manager.get_proxy(auth_header, attempt)
        if not proxy_url:
            return web.Response(text="No proxies available", status=503)

        logger.info(
            f"Attempt {attempt + 1}/{MAX_RETRIES}: Using {proxy_url} (Score: {proxy_manager.proxies[proxy_url].score}) for {target_url}"
        )

        start_time = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=60, connect=10)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    proxy=proxy_url,
                    allow_redirects=True,
                ) as resp:
                    response_body = await resp.read()
                    latency = time.time() - start_time

                    # If we got a response from the actual provider, it's a success for the proxy
                    # 401, 404, etc. from OpenRouter mean the proxy WORKED.
                    # 502, 503, 504 usually mean the PROXY failed.
                    if resp.status < 500 and resp.status != 407:
                        proxy_manager.report_outcome(proxy_url, True, latency)
                        logger.info(
                            f"SUCCESS! {resp.status} via {proxy_url} in {latency:.2f}s"
                        )
                    else:
                        proxy_manager.report_outcome(proxy_url, False)
                        logger.warning(f"Upstream error {resp.status} via {proxy_url}")

                    return web.Response(
                        body=response_body,
                        status=resp.status,
                        headers={
                            k: v
                            for k, v in resp.headers.items()
                            if k.lower()
                            not in ("transfer-encoding", "content-encoding")
                        },
                    )
        except Exception as e:
            latency = time.time() - start_time
            last_error = e
            proxy_manager.report_outcome(proxy_url, False)
            logger.warning(f"Attempt {attempt + 1} FAILED via {proxy_url}: {str(e)}")
            continue

    logger.error(
        f"FATAL: All {MAX_RETRIES} attempts failed for {target_url}. Last error: {last_error}"
    )
    return web.Response(
        text=f"Gateway Error after {MAX_RETRIES} retries: {str(last_error)}", status=502
    )


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
    logger.info("Intelligent Reputation-Based Gateway running on port 8080")

    while True:
        await asyncio.sleep(3600)


if __name__ == "__main__":
    asyncio.run(main())
