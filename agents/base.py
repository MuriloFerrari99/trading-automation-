"""Contrato de agente e base comum (doc 05 §4).

Um agente consome de um topico (inbox), processa e publica em outro(s) — sem
conhecer os outros agentes nem a orquestracao concreta. `step()` executa um
ciclo; subclasses implementam `handle(msg)`. Agentes sem inbox (ex: ingestao,
disparada por schedule) sobrescrevem `step()` diretamente.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from orchestration.bus import Message, MessageBus


@runtime_checkable
class Agent(Protocol):
    name: str

    def step(self, bus: MessageBus) -> None: ...


class BaseAgent:
    """Agente base: nome, logging e o esqueleto do ciclo consumir->processar.

    Agentes COM inbox implementam `handle`. Agentes sem inbox (ex: ingestao)
    sobrescrevem `step()` e nao precisam de `handle`.
    """

    def __init__(self, name: str, inbox: str | None = None) -> None:
        self.name = name
        self.inbox = inbox
        self.log = logging.getLogger(f"agent.{name}")

    def step(self, bus: MessageBus) -> None:
        if self.inbox is None:
            return  # agentes sem inbox (ex: ingestao) sobrescrevem step()
        msg = bus.consume(self.inbox, timeout=None)
        while msg is not None:
            try:
                self.handle(msg, bus)
            except Exception:
                self.log.exception("Falha ao processar mensagem em '%s'", self.inbox)
            msg = bus.consume(self.inbox, timeout=None)

    def handle(self, msg: Message, bus: MessageBus) -> None:
        """Processa uma mensagem do inbox e publica resultados no bus.

        Subclasses com inbox devem sobrescrever. O default sinaliza uso indevido.
        """
        raise NotImplementedError(f"{self.name}: handle() nao implementado")
