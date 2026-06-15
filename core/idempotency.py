"""Idempotencia de ordens via client_order_id determinístico.

Regra do doc 02 (§7.4): toda ordem carrega um `client_order_id` unico e
DETERMINISTICO, gerado a partir do EVENTO DE DECISAO (estrategia + simbolo +
lado + timestamp da decisao). O mesmo evento gera o mesmo id, entao um retry
ou reconexao reenvia o mesmo id e o broker rejeita o duplicado — nunca duplica
posicao. NAO usar now() aqui (mudaria a cada tentativa).
"""

from __future__ import annotations

import hashlib

PREFIX = "bot-"
_MAX_LEN = 24  # comprimento do hash truncado (Alpaca aceita ate 128 chars)


def make_client_order_id(strategy: str, symbol: str, side: str, decision_ts: str) -> str:
    """ID deterministico de uma decisao de ordem.

    decision_ts deve ser o timestamp ESTAVEL da decisao (ex: OrderIntent.created_at
    em ISO-8601), nao o instante do envio.
    """
    raw = f"{strategy}|{symbol}|{side}|{decision_ts}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:_MAX_LEN]
    return PREFIX + digest
