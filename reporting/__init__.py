"""Track record auditavel — a espinha dorsal de captacao de AUM.

Infra AGNOSTICA de estrategia: grava a curva de equity (NAV) mark-to-market do
fim de cada dia numa tabela append-only com cadeia de hash (a prova de
reescrita), calcula metricas since-inception (CAGR/Sharpe/Sortino/MaxDD/Calmar/
retornos mensais/drawdowns) reusando as formulas ja testadas em simulation/, e
compara sempre contra benchmark vivo (SPY buy&hold + 60/40 sintetico).

Modulos:
- nav_repo:     tabela `nav_history` (append-only + hash chain + idempotencia diaria).
- benchmarks:   NAV sintetico 60/40 incremental (reusa a convencao do backtest).
- capture:      captura EOD broker-agnostica (o ponto de plug do caminho vivo).
- track_record: metricas since-inception + relatorio de texto.
- verify_chain: verificador de integridade (a ferramenta do auditor externo).
"""
