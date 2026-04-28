import base64
import ipaddress
import logging
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

from proxy.common.constants import (
    ANY_INTERFACE_HOSTNAMES,
    COLON,
    LOCAL_INTERFACE_HOSTNAMES,
)
from proxy.common.flag import flags
from proxy.common.utils import bytes_, text_
from proxy.core.base import TcpUpstreamConnectionHandler
from proxy.http import Url, httpHeaders, httpMethods
from proxy.http.exception import HttpProtocolException
from proxy.http.parser import HttpParser
from proxy.http.proxy import HttpProxyBasePlugin

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if not any('--proxy-pool' in action.option_strings for action in flags.parser._actions):
    flags.add_argument(
        '--proxy-pool',
        action='append',
        nargs=1,
        default=[],
        help='List of upstream proxies to use in the pool',
    )


class RotatingProxyPlugin(TcpUpstreamConnectionHandler, HttpProxyBasePlugin):
    """Proxy rotator plugin for proxy.py.

    Routes client requests through a pool of upstream HTTP(S) proxies, retrying
    until one works.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._endpoint: Optional[Url] = None
        self._metadata: List[Any] = [None, None, None, None]

    def _proxy_file(self) -> List[str]:
        proxy_file = Path('PROXIES.txt')
        if not proxy_file.is_file():
            return []

        proxies: List[str] = []
        try:
            for line in proxy_file.read_text(encoding='utf-8').splitlines():
                proxy = line.strip()
                if not proxy or proxy.startswith('#'):
                    continue
                proxies.append(proxy)
        except Exception as exc:
            logger.warning('Unable to read PROXIES.txt: %s', exc)
        return proxies

    def _proxy_pool(self) -> List[str]:
        pool: List[str] = []
        pool.extend(self._proxy_file())

        for entry in getattr(self.flags, 'proxy_pool', []) or []:
            value = entry[0] if isinstance(entry, list) and entry else entry
            if isinstance(value, str) and value.strip():
                pool.append(value.strip())

        env_value = os.getenv('PROXY_POOL', '')
        for item in env_value.split(','):
            if item and item.strip():
                pool.append(item.strip())

        return pool

    def _select_proxy(self) -> Url:
        pool = [url for url in self._proxy_pool() if url]
        if not pool:
            raise HttpProtocolException(
                'No upstream proxies configured. Set PROXY_POOL or --proxy-pool.',
            )
        return Url.from_bytes(bytes_(random.choice(pool)))

    def before_upstream_connection(
        self,
        request: HttpParser,
    ) -> Optional[HttpParser]:
        try:
            if ipaddress.ip_address(text_(request.host)).is_private:
                return request
        except ValueError:
            pass

        proxy_urls = self._proxy_pool()
        if not proxy_urls:
            raise HttpProtocolException(
                'No upstream proxies configured. Set PROXY_POOL, --proxy-pool, or add PROXIES.txt.',
            )

        random.shuffle(proxy_urls)
        last_error: Optional[Exception] = None

        for proxy_url in proxy_urls:
            endpoint = Url.from_bytes(bytes_(proxy_url))
            assert endpoint.hostname and endpoint.port

            if endpoint.port == self.flags.port and endpoint.hostname in (
                LOCAL_INTERFACE_HOSTNAMES + ANY_INTERFACE_HOSTNAMES
            ):
                return request

            if self.upstream and not self.upstream.closed:
                self.upstream.close()
                self.upstream = None

            endpoint_tuple = (text_(endpoint.hostname), endpoint.port)
            logger.info('Trying upstream proxy %s:%s', *endpoint_tuple)

            self.initialize_upstream(*endpoint_tuple)
            assert self.upstream
            try:
                self.upstream.connect()
            except (TimeoutError, ConnectionRefusedError) as exc:
                logger.warning(
                    'Upstream proxy failed: %s:%s (%s)',
                    *endpoint_tuple,
                    exc,
                )
                if self.upstream and not self.upstream.closed:
                    self.upstream.close()
                    self.upstream = None
                last_error = exc
                continue

            self._endpoint = endpoint
            logger.info('Connected to upstream proxy %s:%s', *endpoint_tuple)
            return None

        raise HttpProtocolException(
            'Could not connect to any upstream proxy. Last error: %s' %
            (last_error or 'unknown',),
        )

    def handle_client_request(
        self,
        request: HttpParser,
    ) -> Optional[HttpParser]:
        if not self.upstream:
            return request
        assert self.upstream

        host, port = None, None
        if request.has_header(b'host'):
            url = Url.from_bytes(request.header(b'host'))
            assert url.hostname
            host, port = url.hostname.decode('utf-8'), url.port
            port = port if port else (443 if request.is_https_tunnel else 80)

        path = None if not request.path else request.path.decode()
        self._metadata = [host, port, path, request.method]

        if self._endpoint and self._endpoint.has_credentials:
            assert self._endpoint.username and self._endpoint.password
            request.add_header(
                httpHeaders.PROXY_AUTHORIZATION,
                b'Basic ' +
                base64.b64encode(
                    self._endpoint.username + COLON + self._endpoint.password,
                ),
            )

        self.upstream.queue(memoryview(request.build(for_proxy=True)))
        return request

    def handle_client_data(self, raw: memoryview) -> Optional[memoryview]:
        assert self.upstream
        self.upstream.queue(raw)
        return raw

    def handle_upstream_chunk(self, chunk: memoryview) -> Optional[memoryview]:
        if not self.upstream:
            return chunk
        raise Exception('Unexpected upstream chunk handling')

    def on_upstream_connection_close(self) -> None:
        if self.upstream and not self.upstream.closed:
            logger.debug('Closing upstream proxy connection')
            self.upstream.close()
            self.upstream = None

    def on_access_log(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not self.upstream:
            return context
        addr, port = (self.upstream.addr[0], self.upstream.addr[1]) if self.upstream else (None, None)
        context.update({
            'upstream_proxy_host': addr,
            'upstream_proxy_port': port,
            'server_host': self._metadata[0],
            'server_port': self._metadata[1],
            'request_path': self._metadata[2],
            'response_bytes': self.total_size,
        })
        self.access_log(context)
        return None

    def access_log(self, log_attrs: Dict[str, Any]) -> None:
        access_log_format = '{client_ip}:{client_port} - {request_method} {server_host}:{server_port}{request_path} -> {upstream_proxy_host}:{upstream_proxy_port} - {response_code} {response_reason} - {response_bytes} bytes - {connection_time_ms} ms'
        request_method = self._metadata[3]
        if request_method and request_method == httpMethods.CONNECT:
            access_log_format = '{client_ip}:{client_port} - {request_method} {server_host}:{server_port} -> {upstream_proxy_host}:{upstream_proxy_port} - {response_bytes} bytes - {connection_time_ms} ms'
        logger.info(access_log_format.format_map(log_attrs))
