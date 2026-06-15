"""Configuracao da camada de risco (defaults conservadores, configuraveis).

Valores em FRACAO do equity (0.01 = 1%). Defaults seguem o doc 02:
- risk_per_trade: 1% do equity por trade (fixed fractional).
- max_per_symbol: <= 20% do equity em um unico ativo.
- max_portfolio_heat: risco somado em aberto <= 10% do equity.
- daily_loss_limit: para de abrir risco novo ao perder 3% no dia.
- max_drawdown: halt ao atingir 20% de drawdown pico-a-vale.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    risk_per_trade_pct: Decimal = Field(Decimal("0.01"), alias="RISK_PER_TRADE_PCT")
    max_per_symbol_pct: Decimal = Field(Decimal("0.20"), alias="MAX_PER_SYMBOL_PCT")
    max_portfolio_heat_pct: Decimal = Field(Decimal("0.10"), alias="MAX_PORTFOLIO_HEAT_PCT")
    daily_loss_limit_pct: Decimal = Field(Decimal("0.03"), alias="DAILY_LOSS_LIMIT_PCT")
    max_drawdown_pct: Decimal = Field(Decimal("0.20"), alias="MAX_DRAWDOWN_PCT")
    # Distancia ate o stop assumida no calculo de "portfolio heat" (risco-ate-o
    # -stop) quando nao ha stop especifico por posicao. Default 10% (= trailing).
    assumed_stop_pct: Decimal = Field(Decimal("0.10"), alias="ASSUMED_STOP_PCT")


def get_risk_settings() -> RiskSettings:
    return RiskSettings()  # type: ignore[call-arg]
