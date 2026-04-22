#!/usr/bin/env python3
"""Export structured Operator/Predictor metadata for the visual builder.

This exporter intentionally uses explicit imports for v1.  The project has
many LM-backed components whose imports may require large optional
dependencies; keeping the initial set curated makes the script predictable.
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_export_stubs() -> None:
    """Stub optional runtime-only dependencies that are irrelevant to docs."""
    if "pigmento" not in sys.modules:
        pigmento = types.ModuleType("pigmento")
        pigmento.pnt = lambda *args, **kwargs: None
        sys.modules["pigmento"] = pigmento

    if "torch" not in sys.modules:
        class _DummyModule:
            def __init__(self, *args, **kwargs):
                pass

        class _DummyTensor:
            pass

        class _DummyNN(types.ModuleType):
            def __getattr__(self, name):
                value = type(name, (_DummyModule,), {})
                setattr(self, name, value)
                return value

        torch = types.ModuleType("torch")
        nn = _DummyNN("torch.nn")
        functional = types.ModuleType("torch.nn.functional")
        init = types.ModuleType("torch.nn.init")

        nn.Module = _DummyModule
        nn.Parameter = lambda value=None, *args, **kwargs: value
        nn.functional = functional
        nn.init = init
        init.xavier_uniform_ = lambda tensor, *args, **kwargs: tensor
        init.calculate_gain = lambda *args, **kwargs: 1

        torch.Tensor = _DummyTensor
        torch.nn = nn
        torch.empty = lambda *args, **kwargs: _DummyTensor()
        torch.zeros = lambda *args, **kwargs: _DummyTensor()

        sys.modules["torch"] = torch
        sys.modules["torch.nn"] = nn
        sys.modules["torch.nn.functional"] = functional
        sys.modules["torch.nn.init"] = init

    if "unitok" not in sys.modules:
        class _DummyVocab:
            def __init__(self, name="__dummy_vocab__", *args, **kwargs):
                self.name = name
                self._size = 0

            def append(self, token):
                self._size += 1
                return self._size - 1

        unitok = types.ModuleType("unitok")
        unitok.UniTok = type("UniTok", (), {})
        unitok.Vocab = _DummyVocab
        sys.modules["unitok"] = unitok

    if "loader.env" not in sys.modules:
        loader_env = types.ModuleType("loader.env")
        loader_env.Env = type("Env", (), {"device": "cpu"})
        sys.modules["loader.env"] = loader_env

    if "loader.embedding_hub" not in sys.modules:
        embedding_hub = types.ModuleType("loader.embedding_hub")
        embedding_hub.EmbeddingHub = type("EmbeddingHub", (), {})
        sys.modules["loader.embedding_hub"] = embedding_hub


_install_export_stubs()

from model.operators.ada_operator import AdaOperator
from model.operators.attention_operator import AttentionOperator
from model.operators.cnn_cat_operator import CNNCatOperator
from model.operators.cnn_operator import CNNOperator
from model.operators.gru_operator import GRUOperator
from model.operators.null_operator import NullConcatOperator, NullSimpleOperator
from model.operators.poly_attention_operator import PolyAttentionOperator
from model.operators.pooling_operator import PoolingOperator
from model.predictors.dcn_predictor import DCNPredictor
from model.predictors.din_predictor import DINPredictor
from model.predictors.dnn_predictor import DNNPredictor
from model.predictors.dot_predictor import DotPredictor
from model.predictors.miner_predictor import MINERPredictor


OPERATORS = [
    CNNOperator,
    CNNCatOperator,
    AttentionOperator,
    AdaOperator,
    GRUOperator,
    PoolingOperator,
    PolyAttentionOperator,
    NullSimpleOperator,
    NullConcatOperator,
]

PREDICTORS = [
    DotPredictor,
    DNNPredictor,
    DCNPredictor,
    DINPredictor,
    MINERPredictor,
]


def build_payload() -> dict:
    return {
        "schemaVersion": 1,
        "source": "legommenders-algorithm",
        "operators": [operator.describe() for operator in OPERATORS],
        "predictors": [predictor.describe() for predictor in PREDICTORS],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts" / "operator_metadata.json",
        help="JSON file to write.",
    )
    args = parser.parse_args()

    payload = build_payload()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"Exported {len(payload['operators'])} operators and {len(payload['predictors'])} predictors to {args.output}")


if __name__ == "__main__":
    main()
