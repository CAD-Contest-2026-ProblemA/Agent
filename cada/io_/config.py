"""Parse the ``-config`` YAML described in the problem statement.

Expected shape (Figure 6 of the problem PDF)::

    provider: "openai"            # or "anthropic"
    openai:
      api_key: <YOUR_API_KEY>
      model: "gpt-4o-mini"
    anthropic:
      api_key: <YOUR_API_KEY>
      model: "claude-haiku-4-5"
    generation:
      temperature: 0.2
      max_output_tokens: 4096
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    provider: str = "openai"
    openai_api_key: Optional[str] = None
    openai_model: str = "gpt-4o-mini"
    anthropic_api_key: Optional[str] = None
    anthropic_model: str = "claude-haiku-4-5"
    temperature: float = 0.2
    max_output_tokens: int = 4096
    raw: dict = field(default_factory=dict)

    @property
    def model(self) -> str:
        return self.openai_model if self.provider == "openai" else self.anthropic_model

    @property
    def api_key(self) -> Optional[str]:
        return (self.openai_api_key if self.provider == "openai"
                else self.anthropic_api_key)


def _mini_yaml(text: str) -> dict:
    """A tiny parser for the flat, two-level config in Figure 6.

    Avoids a hard PyYAML dependency so the core agent runs on pure stdlib
    (handy on locked-down contest machines).  Handles ``key: value`` at the top
    level and one level of ``  key: value`` nesting under a section header.
    """
    data: dict = {}
    section = None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indented = line[:1] in (" ", "\t")
        key, _, val = line.strip().partition(":")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if not indented:
            if val == "":
                section = {}
                data[key] = section
            else:
                data[key] = val
                section = None
        elif section is not None:
            section[key] = val
    return data


def load_config(path: Optional[str]) -> Config:
    cfg = Config()
    if not path:
        return cfg
    try:
        with open(path, "r") as fh:
            text = fh.read()
    except FileNotFoundError:
        return cfg
    except Exception:
        return cfg
    data = None
    try:
        import yaml
        data = yaml.safe_load(text) or {}
    except Exception:
        data = None
    if not isinstance(data, dict):
        try:
            data = _mini_yaml(text)
        except Exception:
            return cfg

    cfg.raw = data
    cfg.provider = str(data.get("provider", cfg.provider)).strip().lower()

    oa = data.get("openai") or {}
    cfg.openai_api_key = oa.get("api_key") or cfg.openai_api_key
    cfg.openai_model = oa.get("model", cfg.openai_model)

    an = data.get("anthropic") or {}
    cfg.anthropic_api_key = an.get("api_key") or cfg.anthropic_api_key
    cfg.anthropic_model = an.get("model", cfg.anthropic_model)

    gen = data.get("generation") or {}
    cfg.temperature = float(gen.get("temperature", cfg.temperature))
    cfg.max_output_tokens = int(gen.get("max_output_tokens", cfg.max_output_tokens))

    # Treat unfilled placeholders as "no key".
    for attr in ("openai_api_key", "anthropic_api_key"):
        v = getattr(cfg, attr)
        if isinstance(v, str) and (v.strip() == "" or v.strip().startswith("<")):
            setattr(cfg, attr, None)

    return cfg
