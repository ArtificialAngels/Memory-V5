# For details, see docs/scripts/core/v5/v5/hermes_client.md
from __future__ import annotations

import json
import logging
import os
import queue
import random
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("ikaros.memory.v5.hermes")

# ── Proxy bypass: strip all proxy env-vars + force urllib to ignore system proxy.
#    The broken socks://127.0.0.1:8086 from Windows registry otherwise crashes
#    _scrape_token() (urllib.request.urlopen) and httpx/websockets.
for _var in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY',
             'http_proxy', 'https_proxy', 'all_proxy'):
    os.environ.pop(_var, None)
os.environ['NO_PROXY'] = '*'
os.environ['no_proxy'] = '*'
# urllib on Windows reads proxy from registry even without env vars → force none.
urllib.request.install_opener(
    urllib.request.build_opener(urllib.request.ProxyHandler({}))
)

# Environment-configurable URLs
_HERMES_DASHBOARD_URL = os.environ.get(
    "HERMES_DASHBOARD_URL", "http://127.0.0.1:9119"
)
_HERMES_WS_URL = os.environ.get(
    "HERMES_WS_URL", "ws://127.0.0.1:9119/api/ws"
)

# Preload websockets C extension in the main thread.
# In portable-python environments, importing websockets for the first time
# in a daemon thread can trigger 0xC0000005. Preloading here ensures the
# cached module is reused by the daemon thread via sys.modules.
try:
    import websockets as _preload_ws  # noqa: F401
except Exception:
    pass


# ═══════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════


@dataclass
class RetryConfig:
    """Exponential backoff reconnection settings with jitter.

    Attributes:
        initial_delay: Seconds before the first retry.
        max_delay: Upper bound on the computed backoff (seconds).
        multiplier: Factor by which each successive delay grows.
        jitter_factor: Fraction (0-1) of the delay added as random jitter.
        max_retries: How many times a single request can be retried
                     after a WebSocket disconnection.
    """
    initial_delay: float = 1.0
    max_delay: float = 30.0
    multiplier: float = 2.0
    jitter_factor: float = 0.1
    max_retries: int = 3


@dataclass
class ClientConfig:
    """Global configuration for the Hermes client.

    Attributes:
        retry: Retry/backoff policy.
        request_timeout: Default per-request timeout in seconds.
        ws_connect_timeout: Timeout for the WebSocket handshake.
        ws_recv_timeout: Timeout for the initial ``gateway.ready`` message.
    """
    retry: RetryConfig = field(default_factory=RetryConfig)
    request_timeout: float = 120.0
    ws_connect_timeout: float = 10.0
    ws_recv_timeout: float = 5.0


# ═══════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════


class _RetryDelay:
    """Manages exponential backoff delays with jitter for a run."""

    def __init__(self, config: RetryConfig) -> None:
        self.config = config
        self._attempt = 0

    def next_delay(self) -> float:
        """Return the next backoff duration (seconds) and advance the counter."""
        delay = min(
            self.config.initial_delay * (self.config.multiplier ** self._attempt),
            self.config.max_delay,
        )
        jitter = random.uniform(0, self.config.jitter_factor * delay)
        self._attempt += 1
        return delay + jitter

    def reset(self) -> None:
        """Reset the attempt counter (e.g. after a successful connection)."""
        self._attempt = 0


class _HermesRequest:
    """A single RPC request enqueued for the background worker."""

    def __init__(
        self, session_name: str, prompt: str, timeout: float
    ) -> None:
        self.session_name = session_name
        self.prompt = prompt
        self.timeout = timeout
        self.result: "queue.Queue[str]" = queue.Queue()
        self.retry_count = 0


# ═══════════════════════════════════════════════════════════════
# Global state
# ═══════════════════════════════════════════════════════════════

_sessions: dict[str, str] = {}  # {session_name: session_id}
_sessions_lock = threading.Lock()
_requests: "queue.Queue[_HermesRequest] | None" = None
_started = False
_shutdown_event = threading.Event()
_worker_thread: Optional[threading.Thread] = None
_ws_connected = threading.Event()
_config = ClientConfig()


# ═══════════════════════════════════════════════════════════════
# Token scraping
# ═══════════════════════════════════════════════════════════════


def _scrape_token() -> str:
    """Scrape the WebSocket auth token from the dashboard HTML (HTTP GET)."""
    resp = urllib.request.urlopen(f"{_HERMES_DASHBOARD_URL}/", timeout=5)
    html = resp.read().decode("utf-8")
    m = re.search(r'__HERMES_SESSION_TOKEN__="([^"]+)"', html)
    if not m:
        raise RuntimeError("Hermes token not found in dashboard HTML")
    return m.group(1)


# ═══════════════════════════════════════════════════════════════
# Background worker
# ═══════════════════════════════════════════════════════════════


def _worker() -> None:
    """Background thread: maintain one WebSocket connection and process requests."""
    import asyncio
    import websockets

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    async def _run() -> None:
        ws = None
        retry_delay = _RetryDelay(_config.retry)

        while not _shutdown_event.is_set():
            # Poll with a short timeout so shutdown checks are prompt.
            req: Optional[_HermesRequest] = None
            try:
                req = _requests.get(timeout=1)  # type: ignore[union-attr]
            except queue.Empty:
                continue

            try:
                # ── Establish WebSocket if needed ────────────
                if ws is None:
                    token = _scrape_token()
                    ws = await asyncio.wait_for(
                        websockets.connect(
                            f"{_HERMES_WS_URL}?token={token}",
                            max_size=2 ** 20,
                        ),
                        timeout=_config.ws_connect_timeout,
                    )
                    await asyncio.wait_for(
                        ws.recv(), timeout=_config.ws_recv_timeout
                    )  # gateway.ready
                    _ws_connected.set()
                    retry_delay.reset()

                # ── Execute the RPC ──────────────────────────
                reply = await _chat_one(ws, req)
                req.result.put(reply)
                retry_delay.reset()

            except Exception as e:
                # Tear down the broken connection.
                if ws:
                    try:
                        await ws.close()
                    except Exception:
                        pass
                    ws = None
                _ws_connected.clear()

                # Retry the request if the retry budget remains.
                req.retry_count += 1
                if req.retry_count <= _config.retry.max_retries:
                    delay = retry_delay.next_delay()
                    logger.debug(
                        "hermes_client: %s, retrying in %.1fs (attempt %d/%d)",
                        e, delay, req.retry_count, _config.retry.max_retries,
                    )
                    await asyncio.sleep(delay)
                    _requests.put(req)  # type: ignore[union-attr]
                else:
                    try:
                        req.result.put_nowait(f"(Hermes unavailable: {e})")
                    except Exception:
                        pass
                    retry_delay.reset()

        # ── Shutdown: close WebSocket, drain remaining requests ──
        if ws:
            try:
                await ws.close()
            except Exception:
                pass

        while True:
            try:
                req = _requests.get_nowait()  # type: ignore[union-attr]
                try:
                    req.result.put_nowait("(Hermes shutting down)")
                except Exception:
                    pass
            except queue.Empty:
                break

    loop.run_until_complete(_run())


# ═══════════════════════════════════════════════════════════════
# Hermes RPC helpers (called from the event loop)
# ═══════════════════════════════════════════════════════════════


async def _chat_one(ws, req: _HermesRequest) -> str:
    """Single Hermes RPC: find/create session, submit prompt, collect reply.

    On ``session not found`` (stale cached session_id), invalidates the cache
    and retries once with a fresh session.
    """
    sid = await _find_or_create_session(ws, req.session_name)
    try:
        return await _submit_prompt(ws, sid, req.prompt, req.timeout)
    except RuntimeError as e:
        err_str = str(e)
        if "session not found" in err_str:
            # Invalidate stale cached session and retry once
            logger.info("_chat_one: session '%s' stale, recreating...", req.session_name)
            with _sessions_lock:
                _sessions.pop(req.session_name, None)
            sid = await _find_or_create_session(ws, req.session_name)
            if sid:
                return await _submit_prompt(ws, sid, req.prompt, req.timeout)
        raise  # Re-raise original error if retry also fails or not relevant


async def _find_or_create_session(ws, name: str) -> str:
    """Look up an existing session by title, or create a new one."""
    with _sessions_lock:
        if name in _sessions:
            return _sessions[name]

    # List sessions and look for a match.
    rid = f"find_{int(time.time() * 1000)}"
    await ws.send(json.dumps({
        "jsonrpc": "2.0", "id": rid, "method": "session.list", "params": {},
    }))
    for _ in range(10):
        raw = await ws.recv()
        d = json.loads(raw)
        if d.get("id") == rid:
            for s in d.get("result", {}).get("sessions", []):
                if s.get("title") == name:
                    sid = s.get("id", "")
                    if sid:
                        with _sessions_lock:
                            _sessions[name] = sid
                        return sid
            break

    # Not found — create a new session.
    rid2 = f"new_{int(time.time() * 1000)}"
    await ws.send(json.dumps({
        "jsonrpc": "2.0", "id": rid2, "method": "session.create",
        "params": {"title": name},
    }))
    for _ in range(10):
        raw = await ws.recv()
        d = json.loads(raw)
        if d.get("id") == rid2:
            sid = (d.get("result", {}).get("session_id")
                    or d.get("result", {}).get("id", ""))
            if sid:
                with _sessions_lock:
                    _sessions[name] = sid
                return sid
            break

    raise RuntimeError(
        f"Cannot create session '{name}' (Hermes backend may be unavailable)"
    )


async def _submit_prompt(ws, sid: str, text: str, timeout: float) -> str:
    """Submit ``prompt.submit`` and collect the full reply from the event stream."""
    import asyncio as _asyncio

    rid = f"chat_{int(time.time() * 1000)}"
    await ws.send(json.dumps({
        "jsonrpc": "2.0", "id": rid, "method": "prompt.submit",
        "params": {"session_id": sid, "text": text},
    }))
    reply = ""
    deadline = _asyncio.get_event_loop().time() + timeout
    while _asyncio.get_event_loop().time() < deadline:
        remaining = deadline - _asyncio.get_event_loop().time()
        if remaining <= 0:
            break
        try:
            raw = await _asyncio.wait_for(ws.recv(), timeout=min(remaining, 30))
        except _asyncio.TimeoutError:
            continue  # No event yet — recheck deadline and keep waiting.
        d = json.loads(raw)
        if d.get("id") == rid:
            # RPC acknowledgment (e.g. {"status": "streaming"}) — skip,
            # keep waiting for streaming events (message.delta, reasoning.delta, completion).
            if "error" in d:
                raise RuntimeError(
                    f"Hermes error: {d['error'].get('message', 'unknown')}"
                )
            continue
        if d.get("method") == "event":
            p = d.get("params", {})
            t = p.get("type", "")
            if t == "message.delta":
                reply += p.get("delta", "") or ""
            elif t in ("reasoning.delta", "thinking.delta"):
                reply += (p.get("payload", {}) if isinstance(p.get("payload"), dict) else {}).get("text", "") or ""
            elif t == "completion":
                final = p.get("text", "") or ""
                reply = final if final else reply
                break
            elif t == "error":
                raise RuntimeError(
                    f"Hermes error: {p.get('message', 'unknown')}"
                )
    return reply.strip()


# ═══════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════


def start() -> None:
    """Start the background asyncio worker thread (idempotent)."""
    global _started, _requests, _worker_thread
    if _started:
        return
    _shutdown_event.clear()
    _ws_connected.clear()
    _requests = queue.Queue()
    _started = True
    _worker_thread = threading.Thread(
        target=_worker, daemon=True, name="hermes-client"
    )
    _worker_thread.start()
    logger.info("hermes_client: background worker started")


def stop(timeout: float = 30.0) -> None:
    """Graceful shutdown: signal the worker, drain pending requests, close WS.

    Blocks until the worker thread exits or *timeout* seconds elapse.
    After calling ``stop()``, ``start()`` can be called again to restart
    the client.
    """
    global _started, _requests, _worker_thread
    if not _started:
        return

    _shutdown_event.set()

    if _worker_thread is not None and _worker_thread.is_alive():
        _worker_thread.join(timeout=timeout)

    _started = False
    _worker_thread = None
    _ws_connected.clear()
    _requests = None
    with _sessions_lock:
        _sessions.clear()
    logger.info("hermes_client: worker shut down")


def is_connected() -> bool:
    """Return ``True`` if the WebSocket is currently established."""
    return _ws_connected.is_set()


def is_healthy() -> bool:
    """Return ``True`` if the client is started and the WebSocket is connected."""
    return _started and _ws_connected.is_set()


def chat(
    session_name: str, prompt: str, timeout: Optional[float] = None
) -> str:
    """Synchronous blocking: send a prompt via Hermes :9119, return LLM reply.

    If Hermes is unavailable the result is an error string like
    ``"(Hermes unavailable: ...)"``; callers can fall back to local :8080.

    Args:
        session_name: Hermes session title to route the prompt to.
        prompt: The text prompt to submit.
        timeout: Per-request override in seconds.  Falls back to
            ``ClientConfig.request_timeout`` when not provided.
    """
    if _shutdown_event.is_set():
        return "(Hermes unavailable: client is shutting down)"
    if not _started:
        start()
    effective_timeout = (
        timeout if timeout is not None else _config.request_timeout
    )
    req = _HermesRequest(session_name, prompt, effective_timeout)
    _requests.put(req)  # type: ignore[union-attr]
    try:
        return req.result.get(timeout=effective_timeout + 10)
    except queue.Empty:
        return "(Hermes timeout)"


# ═══════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════


def reflect(prompt: str, timeout: Optional[float] = None) -> str:
    """Reflection session (``Ikaros-reflect``)."""
    return chat("Ikaros-reflect", prompt, timeout)


def whisper(prompt: str, timeout: Optional[float] = None) -> str:
    """Internal monologue session (``Ikaros-monologue``)."""
    return chat("Ikaros-monologue", prompt, timeout)
