"""Sizing dinamico (por edge/confianca) — doc 12.

A camada que faz o TAMANHO da posicao variar com a CONVICCAO: mais confianca
(P(win) do ML / score da decisao) -> mais risco, dentro de limites; sem edge ->
tamanho zero. Usa Kelly FRACIONARIO (uma fracao do Kelly cheio — o Kelly cheio
e otimo no crescimento mas brutal no drawdown).

Compoe as primitivas de `risk/sizing.py` (fixed fractional, cap de exposicao):
aqui decidimos QUANTO arriscar; la converte risco -> quantidade. Ver doc 02.
"""

from sizing.dynamic import DynamicSizer, kelly_fraction

__all__ = ["DynamicSizer", "kelly_fraction"]
