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
    # Trailing stop protetor padrao aplicado a QUALQUER long sem protecao e sem
    # config propria de trailing na watchlist (defesa universal de saida).
    default_trailing_stop_pct: Decimal = Field(
        Decimal("0.10"), alias="DEFAULT_TRAILING_STOP_PCT"
    )

    # --- Sizing por conviccao (DynamicSizer) -------------------------------
    # Quando ligado, a fracao de risco-por-trade deixa de ser fixa
    # (risk_per_trade_pct) e passa a ser funcao da CONFIANCA da decisao
    # (regime + sinais + ML), via Kelly fracionario. Sem edge -> 0 (nao opera);
    # mais conviccao -> mais risco, ate `conviction_max_risk_pct`. A conviccao
    # so modula DENTRO do envelope do RiskManager: os caps por simbolo, heat,
    # halt e buying power continuam valendo. Desligado => comportamento legado.
    conviction_sizing: bool = Field(True, alias="CONVICTION_SIZING")
    # Escala do Kelly. Calibrado p/ que a confianca NEUTRA (0.5, = sem edge
    # liquido) mapeie para ~1% de risco — o MESMO baseline fixo de hoje — e a
    # alta conviccao escale ate o teto de 2%. Assim ligar o cerebro nao aumenta
    # o risco por si so: sem sinais/ML, opera como antes (1%); so arrisca mais
    # quando ha conviccao real. (kelly(0.5, b=2)=0.25; 0.25*0.04=0.01=1%.)
    conviction_kelly_cap: float = Field(0.04, alias="CONVICTION_KELLY_CAP")
    # Teto de risco-por-trade com alta conviccao. Decisao do CFO: 2% (moderado).
    conviction_max_risk_pct: float = Field(0.02, alias="CONVICTION_MAX_RISK_PCT")
    # Piso de risco quando HA edge (evita posicoes-po sem sentido operacional).
    conviction_min_risk_pct: float = Field(0.0025, alias="CONVICTION_MIN_RISK_PCT")
    # Abaixo deste nivel de confianca, NAO opera (qty 0).
    conviction_confidence_floor: float = Field(0.5, alias="CONVICTION_CONFIDENCE_FLOOR")
    # Razao ganho/perda (R:R) assumida no Kelly quando a decisao nao informa uma.
    conviction_win_loss_ratio: float = Field(2.0, alias="CONVICTION_WIN_LOSS_RATIO")


def get_risk_settings() -> RiskSettings:
    return RiskSettings()  # type: ignore[call-arg]


def build_dynamic_sizer(settings: RiskSettings | None = None):
    """Constroi o DynamicSizer a partir das RiskSettings (uma fonte de verdade).

    Import tardio para nao acoplar a camada de config ao pacote `sizing`.
    """
    from sizing.dynamic import DynamicSizer

    s = settings or get_risk_settings()
    return DynamicSizer(
        kelly_cap=s.conviction_kelly_cap,
        max_risk_pct=s.conviction_max_risk_pct,
        min_risk_pct=s.conviction_min_risk_pct,
        confidence_floor=s.conviction_confidence_floor,
        default_win_loss_ratio=s.conviction_win_loss_ratio,
    )
