"""Verificador de integridade do track record — a ferramenta do AUDITOR EXTERNO.

Recomputa a cadeia de hash de nav_history a partir SO dos campos crus e compara
com o que esta gravado. Detecta:
  - linha ALTERADA  -> seu row_hash recomputado nao bate;
  - linha REMOVIDA  -> o prev_hash da seguinte deixa de bater (elo quebrado);
  - linha INSERIDA / fora de ordem -> o encadeamento de hash quebra;
  - prev_hash gravado != hash real da linha anterior -> elo quebrado.

Aponta a PRIMEIRA linha quebrada (date + motivo). Verde = serie integra.

COMO UM TERCEIRO VERIFICA, DE FORMA INDEPENDENTE
------------------------------------------------
1.  uv run python -m reporting.verify_chain --db data/trading.sqlite
    -> recomputa a cadeia inteira do banco vivo.
2.  Cruzar com o BACKUP append-only externo (recomendado no plano: commit git
    diario de um export CSV de nav_history). Como o git guarda o historico
    imutavel, qualquer linha que difira entre o banco de hoje e o CSV commitado
    no dia daquele snapshot denuncia reescrita — independente de o atacante ter
    recomputado a cadeia no banco. O hash interno + o backup externo juntos
    fecham a prova: nem editar o banco nem reescrever o passado passam batido.

Retorno do processo: 0 se integro, 1 se ha quebra (util em CI / cron de auditoria).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from reporting.nav_repo import (
    DEFAULT_DB_PATH,
    GENESIS_PREV_HASH,
    NavHistoryRepo,
    compute_row_hash,
)


@dataclass
class ChainBreak:
    index: int            # posicao na serie (0-based)
    date: str
    reason: str           # "row_hash" | "prev_hash"
    expected: str
    found: str


@dataclass
class VerifyResult:
    ok: bool
    n_rows: int
    breaks: list[ChainBreak]

    def summary(self) -> str:
        if self.n_rows == 0:
            return "nav_history vazio — nada a verificar (cadeia trivialmente integra)."
        if self.ok:
            return f"CADEIA INTEGRA: {self.n_rows} linhas verificadas, nenhuma adulteracao."
        first = self.breaks[0]
        return (
            f"CADEIA QUEBRADA na linha {first.index} (date={first.date}, motivo="
            f"{first.reason}).\n  esperado={first.expected[:16]}...  "
            f"encontrado={first.found[:16]}...\n  Total de quebras: {len(self.breaks)}."
        )


def verify(
    db_path: Path | str = DEFAULT_DB_PATH,
    *,
    repo: NavHistoryRepo | None = None,
) -> VerifyResult:
    """Recomputa e valida a cadeia inteira. Para na 1a quebra do encadeamento
    (mas reporta todas as linhas cujo row_hash proprio nao bate)."""
    own = repo is None
    r = repo or NavHistoryRepo(db_path=db_path)
    try:
        rows = r.all_rows()
    finally:
        if own:
            r.close()

    breaks: list[ChainBreak] = []
    prev_hash = GENESIS_PREV_HASH
    for i, row in enumerate(rows):
        fields = {
            "date": row["date"],
            "equity": row["equity"],
            "cash": row["cash"],
            "long_market_value": row["long_market_value"],
            "gross_exposure": row["gross_exposure"],
            "net_exposure": row["net_exposure"],
            "regime": row["regime"],
            "bench_spy": row["bench_spy"],
            "bench_6040": row["bench_6040"],
            "source": row["source"],
        }
        # (a) o prev_hash gravado deve ser o row_hash real da linha anterior.
        if row["prev_hash"] != prev_hash:
            breaks.append(
                ChainBreak(
                    index=i, date=row["date"], reason="prev_hash",
                    expected=prev_hash, found=row["prev_hash"],
                )
            )
        # (b) o row_hash gravado deve bater com o recomputado dos campos crus
        #     encadeado ao prev_hash GRAVADO (assim distinguimos alteracao de
        #     campo de quebra de elo).
        recomputed = compute_row_hash(fields, row["prev_hash"])
        if recomputed != row["row_hash"]:
            breaks.append(
                ChainBreak(
                    index=i, date=row["date"], reason="row_hash",
                    expected=recomputed, found=row["row_hash"],
                )
            )
        prev_hash = row["row_hash"]

    return VerifyResult(ok=not breaks, n_rows=len(rows), breaks=breaks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verifica a integridade (cadeia de hash) do nav_history."
    )
    parser.add_argument("--db", default=str(DEFAULT_DB_PATH), help="caminho do SQLite")
    args = parser.parse_args(argv)

    result = verify(args.db)
    print(result.summary())
    if not result.ok:
        for b in result.breaks:
            print(f"  - linha {b.index} (date={b.date}): {b.reason} divergente")
    return 0 if result.ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
