"""Teste do spike NautilusTrader — prova que o engine roda e casa ordens.

Deterministico (seed fixa), em memoria, sem rede/broker. Garante que a adocao do
Nautilus (Fase 1 do radar de stack) continua funcional nesta maquina.
"""

from __future__ import annotations

from simulation.nautilus_spike import run_spike


def test_spike_executa_e_casa_ordens():
    r = run_spike(n_bars=400, seed=7, quiet=True)
    assert r["bars"] == 400
    assert r["fills"] > 0, "engine nao casou nenhuma ordem (esperado >0 com LAST bars)"
    assert r["positions"] > 0
    assert r["account_rows"] > 0
