"""Configuracao central da aplicacao.

Carrega variaveis de ambiente do `.env` (via pydantic-settings) e expoe um
objeto `Settings` tipado e validado. Aqui vive o **guard de seguranca** que
impede qualquer execucao real enquanto o sistema estiver na fase de paper.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Endpoint oficial de paper trading da Alpaca. Mantido como constante para que
# o guard possa comparar contra ele e recusar qualquer outro host nesta fase.
PAPER_ENDPOINT = "https://paper-api.alpaca.markets"


class LiveTradingBlockedError(RuntimeError):
    """Disparado quando o sistema tenta operar fora do modo paper.

    Enquanto o projeto estiver na fase atual, qualquer tentativa de habilitar
    trading real (LIVE_TRADING=true) ou de apontar para um endpoint que nao
    seja o de paper deve abortar a inicializacao.
    """


class Settings(BaseSettings):
    """Configuracao validada da aplicacao, carregada de `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    alpaca_api_key: str = Field(..., alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(..., alias="ALPACA_SECRET_KEY")
    alpaca_endpoint: str = Field(PAPER_ENDPOINT, alias="ALPACA_ENDPOINT")

    # Guard explicito. Deve permanecer False nesta fase. Mesmo que setado para
    # True, o guard em `enforce_paper_only()` aborta a execucao.
    live_trading: bool = Field(False, alias="LIVE_TRADING")

    # Implementacao de orquestracao: "local" (padrao) ou "opensquad".
    # "opensquad" exige um OrchestratorBridge concreto (aguardando doc/SDK).
    orchestrator: str = Field("local", alias="ORCHESTRATOR")

    @field_validator("alpaca_api_key", "alpaca_secret_key")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError(
                "Credenciais da Alpaca ausentes. Preencha ALPACA_API_KEY e "
                "ALPACA_SECRET_KEY no arquivo .env (veja .env.example)."
            )
        return v.strip()

    @model_validator(mode="after")
    def _enforce_paper_only(self) -> "Settings":
        """GUARD DE SEGURANCA: so permite paper trading nesta fase."""
        if self.live_trading:
            raise LiveTradingBlockedError(
                "LIVE_TRADING=true esta bloqueado nesta fase do projeto. "
                "O sistema opera exclusivamente em paper trading. Para habilitar "
                "trading real sera necessario remover este guard conscientemente."
            )
        if self.alpaca_endpoint.rstrip("/") != PAPER_ENDPOINT:
            raise LiveTradingBlockedError(
                f"Endpoint '{self.alpaca_endpoint}' nao permitido. Nesta fase o "
                f"sistema so pode apontar para o endpoint de paper: {PAPER_ENDPOINT}"
            )
        return self

    @property
    def is_paper(self) -> bool:
        return True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a configuracao validada (singleton em cache).

    A primeira chamada carrega o `.env`, valida as credenciais e executa o
    guard de paper-only. Levanta `LiveTradingBlockedError` se o guard falhar.
    """
    return Settings()  # type: ignore[call-arg]
