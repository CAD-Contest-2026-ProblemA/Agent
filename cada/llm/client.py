"""Minimal dual-provider LLM client (OpenAI / Anthropic).

The LLM is only ever asked to translate one natural-language line into one
``{"intent": ..., "params": {...}}`` JSON object; it never reasons about
circuits.  Missing libraries or API keys degrade gracefully to ``None`` so the
deterministic engine keeps working offline.
"""

from __future__ import annotations

import sys
from typing import Optional

from ..io_.config import Config


def _log_err(msg: str) -> None:
    """Surface an LLM failure on stderr (stdout carries the #RESPONSE protocol,
    so diagnostics must never go there). Keeps the graceful-degradation return
    value intact — this only makes the swallowed error visible."""
    print(f"[LLM] {msg}", file=sys.stderr, flush=True)


class LLMClient:
    def __init__(self, config: Config):
        self.config = config
        self._client = None
        self._kind = None
        # Cumulative token usage across all complete() calls this session.
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.total_calls: int = 0
        self.total_errors: int = 0
        self.last_error: Optional[str] = None
        self._init()

    def _init(self):
        cfg = self.config
        if cfg.provider == "anthropic" and cfg.anthropic_api_key:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
                self._kind = "anthropic"
            except Exception as e:
                self._client = None
                _log_err(f"anthropic client init failed ({type(e).__name__}: {e}); "
                         "falling back to rules-only (unmatched lines become no-ops)")
        elif cfg.provider == "openai" and cfg.openai_api_key:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=cfg.openai_api_key)
                self._kind = "openai"
            except Exception as e:
                self._client = None
                _log_err(f"openai client init failed ({type(e).__name__}: {e}); "
                         "falling back to rules-only (unmatched lines become no-ops)")

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

    def complete(self, system: str, user: str) -> Optional[str]:
        if self._client is None:
            return None
        cfg = self.config
        self.total_calls += 1
        try:
            if self._kind == "anthropic":
                msg = self._client.messages.create(
                    model=cfg.anthropic_model,
                    max_tokens=cfg.max_output_tokens,
                    temperature=cfg.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                if hasattr(msg, "usage") and msg.usage:
                    self.total_input_tokens += getattr(msg.usage, "input_tokens", 0)
                    self.total_output_tokens += getattr(msg.usage, "output_tokens", 0)
                return "".join(
                    b.text for b in msg.content if getattr(b, "type", "") == "text")
            else:
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": user}]
                resp = self._openai_complete(cfg.openai_model,
                                             cfg.max_output_tokens,
                                             cfg.temperature, msgs)
                if hasattr(resp, "usage") and resp.usage:
                    self.total_input_tokens += getattr(resp.usage, "prompt_tokens", 0)
                    self.total_output_tokens += getattr(resp.usage, "completion_tokens", 0)
                return resp.choices[0].message.content
        except Exception as e:
            # Graceful degradation (return None so the REPL keeps running), but
            # make the failure VISIBLE — a swallowed billing/auth/model/rate-limit
            # error otherwise looks identical to "the model couldn't route this
            # line", which silently no-ops an entire run.
            self.total_errors += 1
            self.last_error = f"{type(e).__name__}: {e}"
            _log_err(f"{self._kind} call failed ({self.last_error})")
            return None
