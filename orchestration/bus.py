"""Message bus para comunicacao entre agentes (doc 05 §3).

No MVP, uma fila in-memory por topico (produtor/consumidor num processo). A
interface (`MessageBus`) e identica a de um bus Redis — trocar `InMemoryBus`
por Redis Streams depois e mudar a implementacao, nao os agentes.

Topicos do sistema (convencao): market_data, signals, intents, approved,
fills, alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Empty, Queue
from typing import Any, Protocol, runtime_checkable


@dataclass
class Message:
    topic: str
    payload: Any
    meta: dict = field(default_factory=dict)


@runtime_checkable
class MessageBus(Protocol):
    def publish(self, topic: str, payload: Any, **meta: Any) -> None: ...
    def consume(self, topic: str, timeout: float | None = None) -> Message | None: ...


class InMemoryBus:
    """Implementacao MVP: um Queue por topico. Mesma interface de um bus Redis."""

    def __init__(self) -> None:
        self._topics: dict[str, Queue[Message]] = {}

    def _q(self, topic: str) -> Queue[Message]:
        return self._topics.setdefault(topic, Queue())

    def publish(self, topic: str, payload: Any, **meta: Any) -> None:
        self._q(topic).put(Message(topic=topic, payload=payload, meta=meta))

    def consume(self, topic: str, timeout: float | None = None) -> Message | None:
        try:
            block = timeout is not None
            return self._q(topic).get(block=block, timeout=timeout)
        except Empty:
            return None

    def drain(self, topic: str) -> list[Message]:
        """Consome tudo que estiver disponivel no topico agora (nao bloqueia)."""
        out: list[Message] = []
        while True:
            msg = self.consume(topic, timeout=None)
            if msg is None:
                return out
            out.append(msg)

    def pending(self, topic: str) -> int:
        return self._q(topic).qsize()
