"""Structured metadata helpers for model building UIs.

The training code already knows how to instantiate Operators and Predictors,
but a visual builder also needs machine-readable contracts: editable params,
Inputer shape, output kind, and compatibility notes.  This module keeps that
format close to the algorithm code without making the UI parse ``forward``.
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, Iterable, Mapping, Type


BASE_CONFIG_KEYS = {"hidden_size", "input_dim", "inputer_config"}


def describe_operator(operator_cls: Type) -> Dict[str, Any]:
    """Return a JSON-serializable description for an Operator class."""
    metadata = dict(getattr(operator_cls, "metadata", {}) or {})
    param_overrides = metadata.pop("params", {})
    inputer_cls = getattr(operator_cls, "inputer_class", None)
    params = _config_params(getattr(operator_cls, "config_class", None))

    doc = {
        "kind": "operator",
        "id": metadata.get("id", _component_id(operator_cls.__name__, "Operator")),
        "className": operator_cls.__name__,
        "displayName": metadata.get("displayName", _humanize_class_name(operator_cls.__name__, "Operator")),
        "roles": metadata.get("roles", ["item", "user"]),
        "explain": metadata.get("explain", ""),
        "inputer": {
            "className": inputer_cls.__name__ if inputer_cls else None,
            "outputSingleSequence": getattr(inputer_cls, "output_single_sequence", None),
        },
        "flags": {
            "allowCaching": getattr(operator_cls, "allow_caching", True),
            "flattenMode": getattr(operator_cls, "flatten_mode", False),
        },
        "params": _merge_param_metadata(params, param_overrides),
        "defaults": {param["key"]: param["default"] for param in params if "default" in param},
        "contract": {
            "inputShape": _default_input_shape(inputer_cls),
            "maskShape": _default_mask_shape(inputer_cls),
            "outputShape": "tensor[N, hidden_size]",
            "outputKind": "vector",
            "outputDimPolicy": "config.hidden_size",
        },
        "notes": [],
        "warnings": [],
    }
    return _deep_merge(doc, metadata)


def describe_predictor(predictor_cls: Type) -> Dict[str, Any]:
    """Return a JSON-serializable description for a Predictor class."""
    metadata = dict(getattr(predictor_cls, "metadata", {}) or {})
    param_overrides = metadata.pop("params", {})
    params = _config_params(getattr(predictor_cls, "config_class", None), base_keys={"hidden_size"})

    doc = {
        "kind": "predictor",
        "id": metadata.get("id", _component_id(predictor_cls.__name__, "Predictor")),
        "className": predictor_cls.__name__,
        "displayName": metadata.get("displayName", _humanize_class_name(predictor_cls.__name__, "Predictor")),
        "explain": metadata.get("explain", ""),
        "flags": {
            "allowRanking": getattr(predictor_cls, "allow_ranking", True),
            "allowMatching": getattr(predictor_cls, "allow_matching", True),
            "keepInputDim": getattr(predictor_cls, "keep_input_dim", False),
        },
        "params": _merge_param_metadata(params, param_overrides),
        "defaults": {param["key"]: param["default"] for param in params if "default" in param},
        "contract": {
            "inputShape": "user + item embeddings",
            "outputShape": "tensor[B] or tensor[B, K]",
            "objective": "ranking or matching",
        },
        "notes": [],
        "warnings": [],
    }
    return _deep_merge(doc, metadata)


def _config_params(config_cls: Type | None, base_keys: Iterable[str] = BASE_CONFIG_KEYS) -> list[Dict[str, Any]]:
    if config_cls is None:
        return []

    try:
        signature = inspect.signature(config_cls.__init__)
    except (TypeError, ValueError):
        return []

    params = []
    skipped = set(base_keys)
    for name, param in signature.parameters.items():
        if name == "self" or name in skipped or param.kind == inspect.Parameter.VAR_KEYWORD:
            continue
        entry = {
            "key": name,
            "label": name,
            "type": _param_type(param),
        }
        if param.default is not inspect.Parameter.empty:
            entry["default"] = _jsonable(param.default)
        params.append(entry)
    return params


def _merge_param_metadata(params: list[Dict[str, Any]], overrides: Mapping[str, Mapping[str, Any]]) -> list[Dict[str, Any]]:
    if not overrides:
        return params
    merged = []
    seen = set()
    for param in params:
        seen.add(param["key"])
        merged.append(_deep_merge(param, overrides.get(param["key"], {})))
    for key, override in overrides.items():
        if key not in seen:
            merged.append({"key": key, "label": key, **override})
    return merged


def _param_type(param: inspect.Parameter) -> str:
    annotation = param.annotation
    default = None if param.default is inspect.Parameter.empty else param.default
    if annotation is bool or isinstance(default, bool):
        return "boolean"
    if annotation in {int, float} or isinstance(default, (int, float)):
        return "number"
    if annotation in {list, tuple} or isinstance(default, (list, tuple)):
        return "list"
    return "text"


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    return repr(value)


def _default_input_shape(inputer_cls: Type | None) -> str:
    name = inputer_cls.__name__ if inputer_cls else ""
    if name == "SimpleInputer":
        return "dict[col -> tensor[N, Lc, input_dim]]"
    if name == "ConcatInputer":
        return "tensor[N, L, input_dim]"
    if name == "FlattenSeqInputer":
        return "tensor[N, L_flat, input_dim]"
    if name == "SingleColumnInputer":
        return "tensor[N, L, input_dim]"
    return "operator-specific"


def _default_mask_shape(inputer_cls: Type | None) -> str:
    name = inputer_cls.__name__ if inputer_cls else ""
    if name == "SimpleInputer":
        return "dict[col -> tensor[N, Lc]]"
    if name in {"ConcatInputer", "FlattenSeqInputer", "SingleColumnInputer"}:
        return "tensor[N, L]"
    return "operator-specific"


def _component_id(class_name: str, suffix: str) -> str:
    return class_name[:-len(suffix)] if class_name.endswith(suffix) else class_name


def _humanize_class_name(class_name: str, suffix: str) -> str:
    name = _component_id(class_name, suffix)
    chars = []
    for index, char in enumerate(name):
        prev = name[index - 1] if index else ""
        nxt = name[index + 1] if index + 1 < len(name) else ""
        if index and char.isupper() and (not prev.isupper() or nxt.islower()):
            chars.append(" ")
        chars.append(char)
    return "".join(chars)


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = _jsonable(value)
    return result
