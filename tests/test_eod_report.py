"""Teste do EOD combinado: capture_and_report grava NAV e gera o tearsheet.

Broker-agnostico (FakeBroker), em tmp, sem rede. Garante que o ponto de plug do
ciclo (NAV + relatorio) funciona ponta a ponta e produz um HTML real.
"""

from __future__ import annotations

from decimal import Decimal

from broker.fake_broker import FakeBroker
from reporting.nav_repo import NavHistoryRepo
from reporting.quantstats_report import capture_and_report


def test_capture_and_report_gera_tearsheet(tmp_path):
    broker = FakeBroker(
        cash=Decimal("100000"),
        prices={"SPY": Decimal("400"), "IEF": Decimal("100")},
    )
    db = tmp_path / "nav.sqlite"
    out = tmp_path / "tearsheet.html"

    # Dia 1: 1 ponto -> ainda nao da serie; report deve ser None (so 1 retorno < 2).
    repo = NavHistoryRepo(db_path=db)
    try:
        r1 = capture_and_report(
            broker, repo=repo, db_path=db, output=out, benchmark=False, day="2026-06-15"
        )
        assert r1["nav_row"]["date"] == "2026-06-15"
        assert r1["report"] is None  # 1 ponto = 0 retornos < 2

        # Dia 2: equity muda -> 2 pontos = 1 retorno, ainda < 2 -> sem tearsheet.
        broker.seed_position("SPY", Decimal("10"), Decimal("400"))
        broker.set_price("SPY", Decimal("410"))
        r2 = capture_and_report(
            broker, repo=repo, db_path=db, output=out, benchmark=False, day="2026-06-16"
        )
        assert r2["nav_row"]["date"] == "2026-06-16"
        assert r2["report"] is None

        # Dia 3: 3 pontos = 2 retornos -> tearsheet sai.
        broker.set_price("SPY", Decimal("395"))
        r3 = capture_and_report(
            broker, repo=repo, db_path=db, output=out, benchmark=False, day="2026-06-17"
        )
        assert r3["nav_row"]["date"] == "2026-06-17"
        assert r3["report"] == str(out)
        assert out.exists() and out.stat().st_size > 1000
    finally:
        repo.close()
