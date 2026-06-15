"""ModelStore — onde mora o classificador PROMOVIDO, em disco.

A peca que faltava para tirar o ML do shadow mode na pratica: o job de retreino
(ml/retraining.py) grava aqui o modelo que passou no gate; o runtime carrega
daqui no boot e injeta no DecisionEnricher. Antes, a promocao era so um booleano
no relatorio — nada persistia, entao nada influenciava decisoes de verdade.

Formato: um JSON unico (atomico via arquivo temporario + replace) com o modelo
serializado (SetupClassifier.to_dict) e metadados de auditoria (quando, com que
metricas de walk-forward/calibracao foi promovido).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ml.setup_classifier import SetupClassifier

DEFAULT_MODEL_PATH = Path("data/models/setup_classifier.json")


@dataclass
class PromotedRecord:
    classifier: SetupClassifier
    promoted_at: str
    metadata: dict


class ModelStore:
    """Persiste/carrega o champion promovido. Um arquivo por store."""

    def __init__(self, path: Path | str = DEFAULT_MODEL_PATH) -> None:
        self.path = Path(path)

    # ------------------------------ escrita ------------------------------ #

    def save(self, classifier: SetupClassifier, *, metadata: dict | None = None) -> PromotedRecord:
        """Grava o modelo de forma atomica. So faz sentido com modelo treinado."""
        if not classifier.is_trained:
            raise ValueError("recusando persistir classificador nao treinado")
        promoted_at = datetime.now(timezone.utc).isoformat()
        payload = {
            "promoted_at": promoted_at,
            "metadata": metadata or {},
            "model": classifier.to_dict(),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)  # atomico: nunca deixa o arquivo meio-escrito
        return PromotedRecord(classifier=classifier, promoted_at=promoted_at, metadata=payload["metadata"])

    def clear(self) -> None:
        """Remove o modelo promovido (volta ao shadow). No-op se nao existe."""
        self.path.unlink(missing_ok=True)

    # ------------------------------ leitura ------------------------------ #

    def exists(self) -> bool:
        return self.path.exists()

    def load(self) -> SetupClassifier | None:
        """Carrega o classificador promovido, ou None se nao ha (-> shadow)."""
        rec = self.load_record()
        return rec.classifier if rec else None

    def load_record(self) -> PromotedRecord | None:
        if not self.path.exists():
            return None
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            clf = SetupClassifier.from_dict(payload["model"])
        except (ValueError, KeyError, OSError):
            return None  # arquivo corrompido/incompativel -> trata como ausente
        if not clf.is_trained:
            return None
        return PromotedRecord(
            classifier=clf,
            promoted_at=payload.get("promoted_at", ""),
            metadata=payload.get("metadata", {}),
        )


def load_promoted_classifier(path: Path | str = DEFAULT_MODEL_PATH) -> SetupClassifier | None:
    """Atalho para o wiring no runtime: o classificador promovido ou None."""
    return ModelStore(path).load()
