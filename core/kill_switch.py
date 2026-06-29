"""Kill switch central.

Mecanismo unico que pausa TODAS as ordens. E checado pelo Executor antes de
qualquer submissao na corretora. Combina duas fontes para ser dificil de
ignorar por acidente:

1. Variavel de ambiente `KILL_SWITCH` (truthy => engajado).
2. Presenca de um arquivo sentinela em disco (default: `KILL_SWITCH` no cwd).

Qualquer uma das duas engaja o kill switch. O arquivo permite parar o sistema
de fora (ex: `touch KILL_SWITCH`) sem reiniciar o processo.
"""

from __future__ import annotations

import os
from pathlib import Path

_TRUTHY = {"1", "true", "yes", "on", "engaged"}

DEFAULT_SENTINEL = Path(__file__).resolve().parent.parent / "KILL_SWITCH"


class KillSwitchEngagedError(RuntimeError):
    """Disparado quando uma ordem e tentada com o kill switch engajado."""


class KillSwitch:
    """Controla e consulta o estado do kill switch."""

    def __init__(self, sentinel_path: Path | str = DEFAULT_SENTINEL) -> None:
        self._sentinel = Path(sentinel_path)

    @property
    def sentinel_path(self) -> Path:
        return self._sentinel

    def is_engaged(self) -> bool:
        """True se o kill switch estiver ativo por env var ou arquivo."""
        env_val = os.environ.get("KILL_SWITCH", "").strip().lower()
        if env_val in _TRUTHY:
            return True
        return self._sentinel.exists()

    def reason(self) -> str | None:
        """Descreve por que o kill switch esta engajado, ou None."""
        env_val = os.environ.get("KILL_SWITCH", "").strip().lower()
        if env_val in _TRUTHY:
            return "variavel de ambiente KILL_SWITCH ativa"
        if self._sentinel.exists():
            return f"arquivo sentinela presente: {self._sentinel}"
        return None

    def engage(self, note: str = "") -> None:
        """Engaja o kill switch criando o arquivo sentinela."""
        self._sentinel.write_text(note or "kill switch engaged\n", encoding="utf-8")

    def release(self) -> None:
        """Desengaja removendo o arquivo sentinela (nao mexe na env var)."""
        self._sentinel.unlink(missing_ok=True)

    def ensure_clear(self) -> None:
        """Levanta KillSwitchEngagedError se o kill switch estiver ativo.

        Chamado pelo Executor imediatamente antes de submeter qualquer ordem.
        """
        if self.is_engaged():
            raise KillSwitchEngagedError(
                f"Kill switch engajado ({self.reason()}). Nenhuma ordem sera enviada."
            )
