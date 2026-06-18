"""Saida ESTRUTURADA e VALIDADA de LLM (Pydantic) para o trading.

Espelha o padrao da Mesa Alumbra (brain.think_structured) adaptado ao contrato do
trading: o LLM e um `call_fn(prompt) -> str` INJETADO (sem acoplar a nenhuma API).
Aqui a saida do LLM e forcada a JSON, parseada e VALIDADA contra um schema pydantic
antes de virar decisao/narrativa — com reparo e fallback seguro (nunca derruba o loop).

Usa apenas `pydantic` (ja dependencia do projeto) — nada novo a instalar.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Callable, TypeVar

from pydantic import BaseModel, ValidationError

logger = logging.getLogger("synthesis.structured")

T = TypeVar("T", bound=BaseModel)


def _json_instruction(schema: type[BaseModel]) -> str:
    try:
        spec = json.dumps(schema.model_json_schema(), ensure_ascii=False)
    except Exception:  # noqa: BLE001
        spec = schema.__name__
    return (
        "\n\nFORMATO OBRIGATORIO: responda APENAS com um objeto JSON valido (sem "
        f"texto fora dele, sem markdown) que satisfaca este JSON Schema:\n{spec}"
    )


def extract_json(text: str):
    """Extrai o primeiro objeto/array JSON de um texto (tolera ```json e prosa)."""
    if not text:
        return None
    stripped = text.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", stripped, re.DOTALL)
    candidates = [fence.group(1).strip()] if fence else []
    candidates.append(stripped)
    for chunk in candidates:
        try:
            return json.loads(chunk)
        except Exception:  # noqa: BLE001
            pass
        for opener, closer in (("{", "}"), ("[", "]")):
            start, end = chunk.find(opener), chunk.rfind(closer)
            if start != -1 and end > start:
                try:
                    return json.loads(chunk[start : end + 1])
                except Exception:  # noqa: BLE001
                    continue
    return None


def structured_call(
    call_fn: Callable[[str], str],
    prompt: str,
    schema: type[T],
    *,
    default: T,
    repair_attempts: int = 1,
) -> T:
    """Chama o LLM pedindo JSON e devolve uma instancia validada de `schema`.

    `call_fn(prompt) -> str` e o LLM injetado. Em ate `repair_attempts` tentativas
    de reparo, devolve o erro para o modelo corrigir. Se nada validar, retorna
    `default` (obrigatorio — garante que o loop nunca quebra).
    """
    attempt_prompt = prompt + _json_instruction(schema)
    last_err: str | None = None
    for _ in range(repair_attempts + 1):
        try:
            raw = call_fn(attempt_prompt)
        except Exception:  # noqa: BLE001
            logger.exception("call_fn do LLM falhou; usando default.")
            return default
        blob = extract_json(raw)
        if blob is not None:
            try:
                return schema.model_validate(blob)
            except ValidationError as exc:
                last_err = str(exc)[:400]
        else:
            last_err = "nenhum JSON encontrado"
        attempt_prompt = (
            f"{prompt}\n\n--- CORRIGIR ---\nResposta anterior invalida: {str(raw)[:400]}\n"
            f"Erro: {last_err}\nResponda APENAS com JSON valido para o schema."
            + _json_instruction(schema)
        )
    logger.warning("structured_call(%s): sem saida valida — usando default (%s)", schema.__name__, last_err)
    return default
