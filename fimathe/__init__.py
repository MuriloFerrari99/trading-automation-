"""Camada 1 — features FIMATHE.

Implementa a `FimatheEngine` especificada no doc 08: nucleo analitico que
traduz a tecnica FIMATHE (doc 07) em colunas/feature sobre um DataFrame OHLC.
Usavel em modo rule-based puro (coluna `signal`) ou como FEATURE GENERATOR
para ML (`get_features_for_ml`).

Escopo: forex/ouro (precos em pip). Ver docs/research/08-fimathe-engine.md.
"""

from fimathe.engine import FimatheEngine

__all__ = ["FimatheEngine"]
