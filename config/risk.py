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


class BetaGuardSettings(BaseSettings):
    """Limites do PortfolioRiskGuard CALIBRADOS ao perfil da estrategia de beta.

    PERFIL DE PRODUCAO = 2.0x (escolha consciente do usuario: mais retorno,
    aceitando mais DD). Ancora de dados: data/beta_frontier_report.txt, ponto
    vol-alvo 10% x alavancagem 2.0x (= PRODUCTION_LEVERAGE em beta_rebalancer):
      CAGR ~+10.9% · Sharpe 0.95 · MaxDD backtest -25.9% · DD AO-VIVO estimado
      -36.2% (mid) / -38.8% (pior) [= max(1.4x backtest, pior crise)] · pior ano
      -19.5% (2022) · vol nominal (expo-alvo) ~20% a.a. (sigma diario ~1.26%;
      a vol REALIZADA e menor, ~14%, pois o vol-target deixa caixa — calibramos
      pela nominal, mais conservadora) · concentracao MAXIMA legitima por nome
      ~200% (o 1/vol pode pôr TODA a alavancagem 2.0x num ativo de baixa vol).

    PRINCIPIO (a classe de bug das 4 rodadas): o guard so pode pegar ANOMALIA,
    NUNCA a operacao normal. Cada limite fica com FOLGA acima do comportamento
    legitimo do perfil 2.0x. ADITIVO: NAO altera RiskSettings (que rege as
    estrategias de acoes). Numeros ANCORADOS nos dados acima, nao inventados:

      - DD-HALT (max_drawdown_pct) = -43% (subiu de -28%, calibrado p/ 1.5x).
        Regra: ACIMA do DD AO-VIVO esperado (-36/-39%) com folga, p/ NUNCA haltar
        na operacao normal a 2.0x. Numero = max(1.4x o DD backtest, pior crise)
        + folga: 1.4x(-25.9%) = -36.2%; o pior caso estimado e -38.8%; somamos
        ~4pp de folga sobre o -38.8% -> -43%. Fica ABAIXO do beta cru ao-vivo
        (-54%/-58%), entao ainda para o bot num tombo PIOR que o do proprio
        produto (cisne negro alem do historico), sem truncar um drawdown normal
        de -25/-36%. (-28%, o valor de 1.5x, HALTARIA na operacao normal de 2.0x.)
      - DAILY-LOSS (daily_loss_limit_pct) = -8% (subiu de -6%). O book 2.0x tem
        vol nominal ~20% a.a. => sigma diario ~1.26%. -8% = ~6.3 sigmas (e ~9
        sigmas na vol realizada) — claramente anomalo; o PIOR dia em 27 anos de
        backtest foi -5.67%. -8% fica acima desse pior dia com folga, entao um
        dia tipico (ou ate de cauda moderada) do perfil 2.0x NAO dispara; so um
        choque intradia anormal halta. (-6% era 1.5x; a 2.0x um -6% e plausivel
        numa cauda e haltaria cedo demais.)
      - PER-SYMBOL (max_per_symbol_pct) = 2.10 (210%). A 2.0x o 1/vol pode
        concentrar TODA a alavancagem num unico nome de baixa vol: a concentracao
        MAXIMA legitima MEDIDA no painel real e EXATAMENTE 200% (ex.: SPY em
        2013-10-02, dias com algum nome >160% = 2.3%, >200% = 0%). O teto DEVE
        ficar acima de 200%, senao clipa o 1/vol e QUEBRA a paridade vivo==
        backtest. 210% = 200% + ~5% de folga: so pega anomalia, nunca toca a
        concentracao legitima (dias > 210% = 0% no historico). (160%, o teto de
        1.5x, clipava 2.3% dos dias a 2.0x — quebraria a paridade.)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # DD-halt pico-a-vale do book de beta (perfil 2.0x). Folga ACIMA do DD AO-VIVO
    # esperado (-36/-39%): max(1.4x backtest -25.9% = -36.2%, pior crise -38.8%)
    # + ~4pp -> -43%. Tolera o tombo MAIOR do perfil 2.0x antes da protecao; so
    # halta num cisne negro alem do historico do produto. -28% (de 1.5x) HALTARIA
    # a operacao normal de 2.0x — exatamente a classe de bug das 4 rodadas.
    max_drawdown_pct: Decimal = Field(Decimal("0.43"), alias="BETA_MAX_DRAWDOWN_PCT")
    # Perda diaria que engaja halt. Book 2.0x: vol nominal ~20% a.a. => sigma diario
    # ~1.26%; -8% e ~6.3 sigmas (pior dia em 27a de backtest = -5.67%). So pega
    # choque anomalo; um dia tipico/cauda moderada do perfil 2.0x NAO dispara.
    daily_loss_limit_pct: Decimal = Field(Decimal("0.08"), alias="BETA_DAILY_LOSS_LIMIT_PCT")
    # Teto de exposicao por simbolo. O 1/vol concentra no ativo de menor vol e sob
    # 2.0x pode alocar TODA a alavancagem num unico nome: a concentracao MAXIMA
    # legitima MEDIDA no painel real e EXATAMENTE 200% (ex.: SPY em 2013-10-02; a
    # concentracao se auto-limita pois o vol-target corta o peso quando o ativo
    # fica volatil). Teto em 210% > maximo legitimo (200%) => so pega anomalia e
    # PRESERVA a paridade vivo==backtest (clipar <=200% quebraria a paridade; 160%
    # de 1.5x clipava 2.3% dos dias a 2.0x). (Coder v4: 60% clipava 55% dos dias.)
    max_per_symbol_pct: Decimal = Field(Decimal("2.10"), alias="BETA_MAX_PER_SYMBOL_PCT")
    # Heat agregado: o rebalancer NAO usa heat por-stop (passa current_heat=0 ao
    # can_open, sem stops por posicao), entao este teto e inerte p/ o rebalance.
    # Ainda assim mantemos FOLGA acima do gross do perfil 2.0x (que chega a ~200%
    # no dia, medio ~166%): 3.00 (300%) > 200% => nunca barra o rebalance mesmo se
    # um caminho futuro passar heat>0. O DD-halt e a defesa real do book.
    max_portfolio_heat_pct: Decimal = Field(Decimal("3.00"), alias="BETA_MAX_HEAT_PCT")

    # --- CAPITAL do SLEEVE do beta (NOVO-1 / NOVO-3 simplificado) ----------
    # O NAV (base do guard e do sizing) tem DOIS modos, e SO um deles usa um
    # "capital alocado" separado:
    #
    #   - DEFAULT (paper SO-BETA, sleeve_capital nao setado): o beta E a conta
    #     inteira. O NAV correto e o equity da conta (broker.get_account().equity),
    #     que ja contem o P&L UMA vez — reusa a MESMA semantica do guard das acoes.
    #     NAO ha "alocado": no boot flat o NAV e o equity (caixa), positivo, e o
    #     guard nasce sadio (NOVO-1).
    #     [NOVO-3] O modo antigo "pct * equity + P&L" contava o P&L DUAS vezes
    #     (equity ja inclui P&L). Foi REMOVIDO. `sleeve_capital_pct` abaixo nao
    #     dirige mais o NAV — fica so p/ compat. de .env legado (ignorado).
    #
    #   - COEXISTENCIA com acoes (sleeve_capital ABSOLUTO > 0): carva um sleeve
    #     limpo com CAIXA FIXO. O NAV = caixa_fixo (constante) + P&L do beta. Como
    #     o caixa fixo NAO inclui P&L, o P&L entra UMA vez (sem double-count) e o
    #     sleeve fica isolado do equity total da conta.
    #
    # Em ambos os modos o NAV de boot e POSITIVO -> sem o halt fantasma do NOVO-1.
    sleeve_capital: Decimal = Field(Decimal("0"), alias="BETA_SLEEVE_CAPITAL")
    # DEPRECATED (NOVO-3): nao afeta mais o NAV. Mantido so p/ nao quebrar .env
    # que ainda defina BETA_SLEEVE_CAPITAL_PCT. Em paper so-beta o NAV e o equity
    # da conta direto; p/ carvar um sleeve, use o ABSOLUTO sleeve_capital.
    sleeve_capital_pct: Decimal = Field(Decimal("1.0"), alias="BETA_SLEEVE_CAPITAL_PCT")


def get_beta_guard_settings() -> BetaGuardSettings:
    return BetaGuardSettings()  # type: ignore[call-arg]


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
