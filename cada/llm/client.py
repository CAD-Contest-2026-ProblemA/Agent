"""Minimal dual-provider LLM client (OpenAI / Anthropic).

The LLM is only ever asked to translate one natural-language line into one
``{"intent": ..., "params": {...}}`` JSON object; it never reasons about
circuits.  Missing libraries or API keys degrade gracefully to ``None`` so the
deterministic engine keeps working offline.
"""

from __future__ import annotations

from typing import Optional

from ..io_.config import Config


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

    def complete(self, system: str, user: str) -> Optional[str]:
        if self._client is None:
            return None
        cfg = self.config
        try:
            if self._kind == "anthropic":
                msg = self._client.messages.create(
                    model=cfg.anthropic_model,
                    max_tokens=cfg.max_output_tokens,
                    temperature=cfg.temperature,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
                return "".join(
                    b.text for b in msg.content if getattr(b, "type", "") == "text")
            else:
                resp = self._client.chat.completions.create(
                    model=cfg.openai_model,
                    max_tokens=cfg.max_output_tokens,
                    temperature=cfg.temperature,
                    messages=[{"role": "system", "content": system},
                              {"role": "user", "content": user}],
                )
                return resp.choices[0].message.content
        except Exception:
            return None
