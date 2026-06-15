"""Dashboard de monitoramento.

Painel web leve (stdlib `http.server`, sem dependencias novas) para acompanhar
em tempo (quase) real o que o sistema esta operando: posicoes, ganhos/perdas,
principais trades, decisoes por regime e a trilha de auditoria.

Fonte primaria de dados: o mesmo SQLite do sistema (`data/trading.sqlite`),
lido em modo somente-leitura. Opcionalmente enriquece com dados ao vivo da
corretora (equity, preco atual, P&L nao-realizado) quando disponivel.

Uso:
    uv run python -m dashboard                 # porta 8787, com broker ao vivo
    uv run python -m dashboard --port 9000     # outra porta
    uv run python -m dashboard --no-broker     # so SQLite (offline, sem rede)
"""

from __future__ import annotations

__all__ = ["build_dashboard_data"]

from dashboard.queries import build_dashboard_data
