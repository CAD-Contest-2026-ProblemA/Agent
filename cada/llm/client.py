"""Minimal dual-provider LLM client (OpenAI / Anthropic).

The LLM is only ever asked to translate one natural-language line into one
``{"intent": ..., "params": {...}}`` JSON object; it never reasons about
circuits.  Missing libraries or API keys degrade gracefully to ``None`` so the
deterministic engine keeps working offline.
"""

from __future__ import annotations

import time
from typing import Optional

from ..io_.config import Config

# Error signatures that never recover on retry (bad key, bad request).
_FATAL_MARKERS = ("api_key", "authentication", "invalid_request", "not_found")
# Rate limits recover, but only after the window refills — worth waiting out.
_RATE_MARKERS = ("rate_limit", "429", "tokens per min", "overloaded", "too many requests")
_ATTEMPTS = 3
_RATE_ATTEMPTS = 7


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self._client = None
        self._kind = None
        self._init()

    def _init(self):
        cfg = self.config
        if cfg.provider == "anthropic" and cfg.anthropic_api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
                self._kind = "anthropic"
            except Exception:
                self._client = None
        elif cfg.provider == "openai" and cfg.openai_api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=cfg.openai_api_key)
                self._kind = "openai"
            except Exception:
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def _openai_complete(self, model, max_tokens, temperature, messages):
        """Call OpenAI, falling back for models that reject legacy params.

        Newer models (gpt-5, o-series) require max_completion_tokens and
        do not accept temperature; retry once with both fixes applied.
        """
        kwargs = dict(model=model, messages=messages, temperature=temperature,
                      max_tokens=max_tokens)
        try:
            return self._client.chat.completions.create(**kwargs)
        except Exception as e:
            err = str(e)
            if "max_tokens" not in err and "temperature" not in err:
                raise
            # Apply all known fixes at once before the single retry
            if "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            kwargs.pop("temperature", None)
            return self._client.chat.completions.create(**kwargs)

    def complete(self, system: str, user: str,
                 dynamic: Optional[str] = None) -> Optional[str]:
        """``system`` is the fixed catalog; ``dynamic`` varies per request.

        Everything up to the cache breakpoint is reused across calls, so the
        per-request half must come strictly after it — on both providers that
        means appending, never interleaving.
        """
        if self._client is None:
            return None
        cfg = self.config
        # Transient failures (rate limits, timeouts, 5xx) must not silently
        # turn one request line into a no-op — retry with backoff first.
        for attempt in range(max(_ATTEMPTS, _RATE_ATTEMPTS)):
            try:
                if self._kind == "anthropic":
                    # The intent catalog is byte-identical on every call and
                    # dwarfs the one request line that follows it, so cache it:
                    # reads bill at ~0.1x.  Caching is a prefix match — anything
                    # per-request must stay in the user message, or the prefix
                    # stops matching and every call pays in full.
                    blocks = [{"type": "text", "text": system,
                               "cache_control": {"type": "ephemeral"}}]
                    if dynamic:
                        blocks.append({"type": "text", "text": dynamic})
                    msg = self._client.messages.create(
                        model=cfg.anthropic_model,
                        max_tokens=cfg.max_output_tokens,
                        temperature=cfg.temperature,
                        system=blocks,
                        messages=[{"role": "user", "content": user}],
                    )
                    return "".join(
                        b.text for b in msg.content if getattr(b, "type", "") == "text")
                else:
                    # OpenAI caches long prefixes automatically; keeping the
                    # variable half last is what preserves the hit.
                    sys_text = system if not dynamic else system + "\n\n" + dynamic
                    msgs = [{"role": "system", "content": sys_text},
                            {"role": "user", "content": user}]
                    resp = self._openai_complete(cfg.openai_model,
                                                 cfg.max_output_tokens,
                                                 cfg.temperature, msgs)
                    return resp.choices[0].message.content
            except Exception as e:
                low = str(e).lower()
                if any(m in low for m in _FATAL_MARKERS):
                    import sys
                    sys.stderr.write(f"[llm] fatal API error, degrading to no-op: {e}\n")
                    return None
                # A rate limit is not a failed request, it is a request that has
                # not happened yet: three quick retries can expire inside one
                # refill window and turn the line into a silent no-op that reads
                # as a completed run.  Wait the window out instead.
                rate = any(m in low for m in _RATE_MARKERS)
                budget = _RATE_ATTEMPTS if rate else _ATTEMPTS
                if attempt + 1 >= budget:
                    import sys
                    sys.stderr.write(f"[llm] API call failed, degrading to no-op: {e}\n")
                    return None
                # 2, 4, 8, ... capped at 60s
                time.sleep(min(60, 2 ** (attempt + 1)) if rate else 2 * (attempt + 1))
        return None
