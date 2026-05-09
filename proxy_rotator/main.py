import asyncio
import hashlib
import time
import logging
import aiohttp
import os
from aiohttp import web
from typing import Dict, List
import random

# Configuration
# Check both /app (local dev) and / (Docker mount)
PROXY_FILES = ["PROXIES.TXT", "/PROXIES.TXT"]
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
    def __init__(self, url: str, source: str = "url"):
        self.url = url
        self.source = source # "file" or "url"
        self.score = INITIAL_SCORE
        self.avg_latency = 0.0
        self.total_calls = 0
        self.failures = 0

    def record_success(self, latency: float):
        self.total_calls += 1
        self.score = min(200, self.score + SUCCESS_REWARD)
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
            new_file_urls = []
            new_url_urls = []
            
            # 1. Read from local files (Priority Source)
            for file_path in PROXY_FILES:
                if os.path.exists(file_path):
                    logger.info(f"Loading priority proxies from {file_path}...")
                    with open(file_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line: continue
                            parts = line.split(':')
                            if len(parts) == 4:
                                url = f"http://{parts[2]}:{parts[3]}@{parts[0]}:{parts[1]}"
                                new_file_urls.append(url)
                            elif len(parts) == 2:
                                url = f"http://{parts[0]}:{parts[1]}"
                                new_file_urls.append(url)
            
            # 2. Fetch from URL (Fallback Source)
            logger.info("Fetching secondary proxies from URL...")
            async with aiohttp.ClientSession() as session:
                async with session.get(PROXY_LIST_URL, timeout=30) as resp:
                    if resp.status == 200:
                        text = await resp.text()
                        for line in text.splitlines():
                            p = line.strip()
                            if p:
                                if not p.startswith("http"):
                                    p = f"http://{p}"
                                new_url_urls.append(p)

            # Update master list
            incoming_urls = {}
            for u in new_file_urls: incoming_urls[u] = "file"
            for u in new_url_urls: 
                if u not in incoming_urls: incoming_urls[u] = "url"

            if incoming_urls:
                # Preservation logic
                for url, source in incoming_urls.items():
                    if url not in self.proxies:
                        self.proxies[url] = ProxyStats(url, source)
                    else:
                        # Update source in case it moved from URL to File
                        self.proxies[url].source = source
                
                # Cleanup dead ones
                for url in list(self.proxies.keys()):
                    if url not in incoming_urls:
                        del self.proxies[url]
                
                self.last_fetch = time.time()
                file_count = sum(1 for p in self.proxies.values() if p.source == "file")
                url_count = sum(1 for p in self.proxies.values() if p.source == "url")
                logger.info(f"Proxy Pool Updated: {file_count} Priority (File), {url_count} Secondary (URL)")
            else:
                logger.warning("No proxies found from any source!")
        except Exception as e:
            logger.error(f"Error fetching proxies: {e}")

    def get_proxy(self, auth_header: str, attempt: int = 0) -> str:
        if not self.proxies:
            return None
            
        # Priority Logic:
        # 1. Source (File > URL)
        # 2. Score (High > Low)
        # 3. Latency (Low > High)
        sorted_proxies = sorted(
            self.proxies.values(),
            key=lambda p: (
                p.source == "file", # True (1) > False (0)
                p.score, 
                -p.avg_latency if p.avg_latency > 0 else -999
            ),
            reverse=True
        )
        
        # On early attempts, respect the sorting strictly
        if attempt < 3:
            return sorted_proxies[attempt % len(sorted_proxies)].url
        else:
            # On retries, pick from the top 30% randomly to ensure coverage
            sample_size = max(1, int(len(sorted_proxies) * 0.3))
            return random.choice(sorted_proxies[:sample_size]).url

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

        logger.info(f"Attempt {attempt+1}/{MAX_RETRIES}: Forwarding to {target_url} via {proxy_url} ({proxy_manager.proxies[proxy_url].source})")

        start_time = time.time()
        try:
            timeout = aiohttp.ClientTimeout(total=300, connect=15)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(
                    method=request.method,
                    url=target_url,
                    headers=headers,
                    data=body,
                    proxy=proxy_url,
                    allow_redirects=True,
                ) as resp:
                    if (resp.status < 400) or (resp.status in (401, 404)):
                        proxy_manager.report_outcome(proxy_url, True, time.time() - start_time)
                        stream_resp = web.StreamResponse(
                            status=resp.status,
                            headers={
                                k: v
                                for k, v in resp.headers.items()
                                if k.lower() not in ("transfer-encoding", "content-encoding", "content-length")
                            }
                        )
                        await stream_resp.prepare(request)
                        async for chunk in resp.content.iter_chunked(16384):
                            await stream_resp.write(chunk)
                        await stream_resp.write_eof()
                        return stream_resp
                    
                    proxy_manager.report_outcome(proxy_url, False)
                    logger.warning(f"Retriable error {resp.status} via {proxy_url}. Trying next...")
                    continue
        except Exception as e:
            last_error = e
            proxy_manager.report_outcome(proxy_url, False)
            logger.warning(f"Attempt {attempt+1} FAILED via {proxy_url}: {str(e)}")
            continue

    logger.error(f"FATAL: All {MAX_RETRIES} attempts failed. Last error: {last_error}")
    return web.Response(text=f"Gateway Error after {MAX_RETRIES} retries: {str(last_error)}", status=502)

async def refresh_loop():
    while True:
        await asyncio.sleep(REFRESH_INTERVAL)
        await proxy_manager.fetch_proxies()

async def main():
    await proxy_manager.fetch_proxies()
    asyncio.create_task(refresh_loop())

    app = web.Application()
    app.add_routes([
        web.route("*", "/{provider}/{tail:.*}", handle_request),
        web.route("*", "/{provider}", handle_request),
    ])

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "0.0.0.0", 8080)
    await site.start()
    logger.info("Priority-Aware Gateway running on port 8080")
    while True: await asyncio.sleep(3600)

if __name__ == "__main__":
    asyncio.run(main())
