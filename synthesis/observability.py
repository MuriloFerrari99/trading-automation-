"""Observabilidade Langfuse para o reasoner via LLM — ENV-GATED e a prova de falhas.

Mesma filosofia da Mesa Alumbra: liga sozinho quando LANGFUSE_PUBLIC_KEY +
LANGFUSE_SECRET_KEY estao no ambiente; sem chaves, vira no-op (zero overhead,
nem importa langfuse). Qualquer erro de tracing e engolido — o loop de trading
nunca quebra por causa de observabilidade.

So tem efeito quando um LLMReasoner com call_fn real e usado (o caminho padrao do
trading e o TemplateReasoner deterministico, que nao chama LLM).

Ligar (requer `uv sync --extra obs` para instalar o SDK):
    export LANGFUSE_PUBLIC_KEY=pk-lf-...
    export LANGFUSE_SECRET_KEY=sk-lf-...
    export LANGFUSE_HOST=http://localhost:3000
"""

from __future__ import annotations

import functools
import logging
import os

logger = logging.getLogger("synthesis.obs")


def langfuse_enabled() -> bool:
    return bool(
        os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY")
    )


def observe_generation(name: str):
    """Decora uma chamada de LLM como 'generation' no Langfuse (ou no-op sem chaves).

    Decide na PRIMEIRA chamada (nao no import) para nao depender da ordem de
    carga do .env; o wrapper real e cacheado por funcao.
    """

    def deco(func):
        cache: dict[str, object] = {}

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not langfuse_enabled():
                return func(*args, **kwargs)
            wrapped = cache.get("fn")
            if wrapped is None:
                try:
                    from langfuse import observe as _observe

                    wrapped = _observe(name=name, as_type="generation")(func)
                except Exception:  # noqa: BLE001
                    logger.exception("Langfuse indisponivel; sem tracing em %s", name)
                    wrapped = func
                cache["fn"] = wrapped
            return wrapped(*args, **kwargs)

        return wrapper

    return deco
