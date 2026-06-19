"""Configuracao do sleeve de cripto (Sistema 2) — defaults CONSERVADORES.

Sistema 1 = buy-and-hold (equity principal, Alpaca). Sistema 2 = caca de alpha
intradiario em cripto perp (Binance). Sao separados por uma PAREDE DE FOGO: o
sleeve opera APENAS o capital isolado `sleeve_capital_usd` e NUNCA pode consumir
capital do Sistema 1.

Valores em FRACAO do capital DO SLEEVE (0.01 = 1%), exceto onde dito em USD.
Defaults pensados para sobreviver ao suicidio classico do varejo cripto:
- leverage_cap: 2x por padrao, com CEILING RIGIDO de 3x (nunca mais, mesmo se
  configurarem 50x — perp permite, a trava existe justamente p/ impedir).
- risk_per_trade: 0.5% do capital do sleeve por trade.
- daily_loss_limit: -3% do capital do sleeve no dia => para de abrir risco.
- max_drawdown: -10% pico-a-vale do sleeve => halt.
- max_concurrent_positions / max_gross_exposure: limita concentracao e heat.

Tudo strategy-agnostic: serve p/ momentum, liquidacao ou market-making.
"""

from __future__ import annotations

from decimal import Decimal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Teto RIGIDO de alavancagem do sleeve. Hard cap a nivel de modulo: nenhuma
# configuracao (env, override, etc.) consegue ultrapassar isto. Perp na Binance
# permite ate 50x+; esta constante e a parede que impede o suicidio do varejo.
LEVERAGE_HARD_CEILING: Decimal = Decimal("3")


class CryptoSleeveSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Capital ISOLADO do sleeve, em USD. Separado do equity do Sistema 1.
    # Default conservador e pequeno: o sleeve prova edge antes de crescer.
    sleeve_capital_usd: Decimal = Field(Decimal("1000"), alias="CRYPTO_SLEEVE_CAPITAL_USD")

    # Alavancagem-alvo do sleeve. Default 2x. Mesmo que se configure mais, o
    # guard aplica o CEILING rigido (LEVERAGE_HARD_CEILING). O validator abaixo
    # ja clampa a configuracao para no maximo o ceiling (defesa em profundidade).
    leverage_cap: Decimal = Field(Decimal("2"), alias="CRYPTO_LEVERAGE_CAP")

    # Risco por trade (% do capital do sleeve). Default 0.5%.
    risk_per_trade_pct: Decimal = Field(Decimal("0.005"), alias="CRYPTO_RISK_PER_TRADE_PCT")

    # Kill de perda diaria do sleeve (% do capital do sleeve). Default -3%.
    daily_loss_limit_pct: Decimal = Field(Decimal("0.03"), alias="CRYPTO_DAILY_LOSS_LIMIT_PCT")

    # Kill de drawdown pico-a-vale do sleeve. Default -10%.
    max_drawdown_pct: Decimal = Field(Decimal("0.10"), alias="CRYPTO_MAX_DRAWDOWN_PCT")

    # Maximo de posicoes concorrentes abertas no sleeve.
    max_concurrent_positions: int = Field(3, alias="CRYPTO_MAX_CONCURRENT_POSITIONS")

    # Exposicao bruta maxima do sleeve, como multiplo do capital do sleeve
    # (notional somado / capital). Default 2x — abaixo do leverage_cap p/ deixar
    # margem; nunca pode exceder o leverage ceiling efetivo.
    max_gross_exposure_x: Decimal = Field(Decimal("2"), alias="CRYPTO_MAX_GROSS_EXPOSURE_X")

    @field_validator("leverage_cap")
    @classmethod
    def _clamp_leverage(cls, v: Decimal) -> Decimal:
        """Clampa a alavancagem configurada ao CEILING rigido (defesa 1/2).

        Mesmo se alguem setar 50x via env, aqui ja cai p/ o ceiling. O guard
        re-aplica o ceiling em runtime (defesa 2/2), entao a trava nao depende
        de a config ter passado por aqui.
        """
        if v < 1:
            return Decimal("1")
        return min(v, LEVERAGE_HARD_CEILING)


def get_crypto_sleeve_settings() -> CryptoSleeveSettings:
    return CryptoSleeveSettings()  # type: ignore[call-arg]
