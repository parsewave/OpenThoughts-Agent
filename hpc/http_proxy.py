#!/usr/bin/env python3
"""
Simple HTTP proxy that forwards CONNECT requests through a SOCKS5 proxy.

This allows HTTP clients (like aiohttp with trust_env=True) to use a SOCKS proxy
by setting HTTP_PROXY/HTTPS_PROXY to this local HTTP proxy.

Usage:
    python http_proxy.py --socks-host 10.128.24.25 --socks-port 21004 &
    export HTTP_PROXY=http://127.0.0.1:8080
    export HTTPS_PROXY=http://127.0.0.1:8080

Requirements:
    pip install python-socks[asyncio]
"""

import argparse
import asyncio
import logging
import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [http-proxy] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


class SocksForwardingProxy:
    """HTTP proxy that forwards CONNECT requests through SOCKS5."""

    def __init__(self, socks_host: str, socks_port: int, listen_port: int = 8080):
        self.socks_host = socks_host
        self.socks_port = socks_port
        self.listen_port = listen_port
        self.server = None

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle an incoming proxy request."""
        try:
            # Read the request line
            request_line = await asyncio.wait_for(reader.readline(), timeout=30)
            if not request_line:
                return

            request_str = request_line.decode("utf-8", errors="ignore").strip()
            parts = request_str.split()
            if len(parts) < 2:
                return

            method = parts[0]
            target = parts[1]

            # Read and discard headers
            while True:
                line = await asyncio.wait_for(reader.readline(), timeout=30)
                if line in (b"\r\n", b"\n", b""):
                    break

            if method == "CONNECT":
                await self._handle_connect(target, reader, writer)
            else:
                writer.write(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
                await writer.drain()

        except asyncio.TimeoutError:
            logger.debug("Client timeout")
        except Exception as e:
            logger.warning(f"Client error: {e}")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _handle_connect(self, target: str, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """Handle CONNECT for HTTPS tunneling via SOCKS5."""
        remote_writer = None
        try:
            # Parse host:port
            if ":" in target:
                host, port_str = target.rsplit(":", 1)
                port = int(port_str)
            else:
                host, port = target, 443

            logger.debug(f"CONNECT {host}:{port}")

            # Connect through SOCKS5
            from python_socks.async_.asyncio.v2 import Proxy

            proxy = Proxy.from_url(f"socks5://{self.socks_host}:{self.socks_port}")
            sock = await proxy.connect(dest_host=host, dest_port=port, timeout=30)

            # Wrap socket in asyncio streams
            remote_reader, remote_writer = await asyncio.open_connection(sock=sock)

            # Send success response
            writer.write(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            await writer.drain()

            # Tunnel bidirectionally
            await self._tunnel(reader, writer, remote_reader, remote_writer)

        except Exception as e:
            logger.warning(f"CONNECT {target} failed: {e}")
            try:
                writer.write(f"HTTP/1.1 502 Bad Gateway\r\n\r\n{e}\r\n".encode())
                await writer.drain()
            except Exception:
                pass
        finally:
            if remote_writer:
                try:
                    remote_writer.close()
                    await remote_writer.wait_closed()
                except Exception:
                    pass

    async def _tunnel(self, client_reader, client_writer, remote_reader, remote_writer):
        """Tunnel data between client and remote."""
        async def pipe(src, dst, name):
            try:
                while True:
                    data = await asyncio.wait_for(src.read(8192), timeout=300)
                    if not data:
                        break
                    dst.write(data)
                    await dst.drain()
            except (asyncio.TimeoutError, asyncio.CancelledError):
                pass
            except (ConnectionResetError, BrokenPipeError):
                pass
            except Exception as e:
                logger.debug(f"Pipe {name} error: {e}")

        await asyncio.gather(
            pipe(client_reader, remote_writer, "c->r"),
            pipe(remote_reader, client_writer, "r->c"),
        )

    async def start(self):
        """Start the proxy server."""
        self.server = await asyncio.start_server(
            self.handle_client, "0.0.0.0", self.listen_port
        )
        logger.info(f"Listening on 0.0.0.0:{self.listen_port}")
        logger.info(f"Forwarding via SOCKS5 {self.socks_host}:{self.socks_port}")

    async def serve_forever(self):
        """Serve until stopped."""
        if self.server:
            await self.server.serve_forever()

    async def stop(self):
        """Stop the proxy server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()


async def main():
    parser = argparse.ArgumentParser(description="HTTP-to-SOCKS5 proxy")
    parser.add_argument("--socks-host", required=True, help="SOCKS5 proxy host")
    parser.add_argument("--socks-port", type=int, required=True, help="SOCKS5 proxy port")
    parser.add_argument("--listen-port", type=int, default=8080, help="Local listen port")
    args = parser.parse_args()

    proxy = SocksForwardingProxy(args.socks_host, args.socks_port, args.listen_port)
    await proxy.start()

    # Handle shutdown
    stop_event = asyncio.Event()
    loop = asyncio.get_event_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop_event.set)

    serve_task = asyncio.create_task(proxy.serve_forever())
    await stop_event.wait()
    serve_task.cancel()
    await proxy.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
