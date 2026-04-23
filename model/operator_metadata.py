"""Source-based component metadata extraction.

Most metadata for the visual builder can be inferred from normal Python code:
class names, config signatures, Inputer classes, and capability flags.  The
small amount of semantic information that cannot be inferred safely lives in
docstrings or an external override file, not in class-level UI dictionaries.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping


BASE_OPERATOR_CONFIG_KEYS = {"hidden_size", "input_dim", "inputer_config"}
BASE_PREDICTOR_CONFIG_KEYS = {"hidden_size"}


@dataclass
class ClassInfo:
    name: str
    kind: str
    node: ast.ClassDef
    source: str
    path: Path
    bases: list[str]
    assignments: dict[str, Any] = field(default_factory=dict)
    doc: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConfigInfo:
    name: str
    node: ast.ClassDef
    source: str
    bases: list[str]
    params: list[dict[str, Any]]


def export_metadata(root: Path, overrides_path: Path | None = None) -> dict[str, Any]:
    """Build the global Operator/Predictor metadata payload from source."""
    root = root.resolve()
    overrides = _read_overrides(overrides_path)
    inputer_index = _scan_inputers(root / "model" / "inputer")

    operator_classes, operator_configs = _scan_component_dir(
        root / "model" / "operators",
        kind="operator",
        suffix="Operator",
        base_class="BaseOperator",
    )
    predictor_classes, predictor_configs = _scan_component_dir(
        root / "model" / "predictors",
        kind="predictor",
        suffix="Predictor",
        base_class="BasePredictor",
    )

    operators = [
        _build_component_doc(
            info=info,
            all_classes=operator_classes,
            configs=operator_configs,
            inputer_index=inputer_index,
            suffix="Operator",
            overrides=overrides,
        )
        for info in _concrete_components(operator_classes, "Operator")
    ]
    predictors = [
        _build_component_doc(
            info=info,
            all_classes=predictor_classes,
            configs=predictor_configs,
            inputer_index=inputer_index,
            suffix="Predictor",
            overrides=overrides,
        )
        for info in _concrete_components(predictor_classes, "Predictor")
    ]

    return {
        "schemaVersion": 2,
        "source": "legommenders-algorithm",
        "extraction": {
            "mode": "ast+docstring+overrides",
            "overrideFile": _safe_relative_path(overrides_path, root) if overrides_path else None,
        },
        "operators": sorted(operators, key=lambda item: item["className"]),
        "predictors": sorted(predictors, key=lambda item: item["className"]),
    }


def _scan_component_dir(
    directory: Path,
    *,
    kind: str,
    suffix: str,
    base_class: str,
) -> tuple[dict[str, ClassInfo], dict[str, ConfigInfo]]:
    classes: dict[str, ClassInfo] = {}
    configs: dict[str, ConfigInfo] = {}

    for path in sorted(directory.glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        for node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            bases = [_name_of(base) for base in node.bases]
            assignments = _class_assignments(node, source)
            doc = _parse_legommenders_doc(ast.get_docstring(node))
            if node.name.endswith("Config"):
                configs[node.name] = ConfigInfo(
                    name=node.name,
                    node=node,
                    source=source,
                    bases=bases,
                    params=_own_config_params(node, source, kind),
                )
                continue
            if node.name.endswith(suffix) and node.name != base_class:
                classes[node.name] = ClassInfo(
                    name=node.name,
                    kind=kind,
                    node=node,
                    source=source,
                    path=path,
                    bases=bases,
                    assignments=assignments,
                    doc=doc,
                )
    return classes, configs


def _scan_inputers(directory: Path) -> dict[str, dict[str, Any]]:
    inputers = {}
    for path in sorted(directory.glob("*.py")):
        source = path.read_text()
        tree = ast.parse(source)
        for node in [n for n in tree.body if isinstance(n, ast.ClassDef)]:
            if not node.name.endswith("Inputer"):
                continue
            assignments = _class_assignments(node, source)
            inputers[node.name] = {
                "className": node.name,
                "outputSingleSequence": assignments.get("output_single_sequence", True),
            }
    return inputers


def _concrete_components(classes: Mapping[str, ClassInfo], suffix: str) -> list[ClassInfo]:
    skipped = {f"Base{suffix}", "LMOperator"}
    result = []
    for info in classes.values():
        if info.name in skipped:
            continue
        if any(base == "ABC" or base.endswith(".ABC") for base in info.bases):
            continue
        result.append(info)
    return result


def _build_component_doc(
    *,
    info: ClassInfo,
    all_classes: Mapping[str, ClassInfo],
    configs: Mapping[str, ConfigInfo],
    inputer_index: Mapping[str, dict[str, Any]],
    suffix: str,
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    config_name = _resolve_attr(info, all_classes, "config_class")
    inputer_name = _resolve_attr(info, all_classes, "inputer_class") if info.kind == "operator" else None
    flags = _infer_flags(info, all_classes)
    params = _resolve_config_params(config_name, configs, info.kind)
    contract = _infer_contract(info, all_classes, inputer_name, inputer_index, flags)

    doc = {
        "kind": info.kind,
        "id": _component_id(info.name, suffix),
        "className": info.name,
        "displayName": _humanize_class_name(info.name, suffix),
        "module": _module_path(info.path),
        "roles": ["item", "user"] if info.kind == "operator" else None,
        "explain": "",
        "configClass": config_name,
        "inputer": inputer_index.get(inputer_name, {"className": inputer_name}) if inputer_name else None,
        "flags": flags,
        "params": params,
        "defaults": {param["key"]: param["default"] for param in params if "default" in param},
        "contract": contract,
        "notes": [],
        "warnings": [],
        "metadataSource": {
            "base": "ast",
            "docstring": bool(info.doc),
            "overrides": False,
        },
    }
    if info.kind == "predictor":
        doc.pop("roles", None)
        doc.pop("inputer", None)

    doc = _merge_docstring(doc, info.doc)
    override = (overrides.get("components") or {}).get(info.name, {})
    if override:
        doc["metadataSource"]["overrides"] = True
        doc = _merge_override(doc, override)

    _add_display_contract_fields(doc["contract"])
    return doc


def _infer_flags(info: ClassInfo, all_classes: Mapping[str, ClassInfo]) -> dict[str, Any]:
    if info.kind == "operator":
        return {
            "allowCaching": _resolve_attr(info, all_classes, "allow_caching", True),
            "flattenMode": _resolve_attr(info, all_classes, "flatten_mode", False),
        }
    allow_ranking = _resolve_attr(info, all_classes, "allow_ranking", True)
    allow_matching = _resolve_attr(info, all_classes, "allow_matching", True)
    flags = {
        "allowRanking": allow_ranking,
        "allowMatching": allow_matching,
        "keepInputDim": _resolve_attr(info, all_classes, "keep_input_dim", False),
    }
    if allow_matching and allow_ranking:
        flags["objective"] = "ranking_or_matching"
    elif allow_matching:
        flags["objective"] = "matching_only"
    else:
        flags["objective"] = "ranking_only"
    return flags


def _infer_contract(
    info: ClassInfo,
    all_classes: Mapping[str, ClassInfo],
    inputer_name: str | None,
    inputer_index: Mapping[str, dict[str, Any]],
    flags: Mapping[str, Any],
) -> dict[str, Any]:
    if info.kind == "predictor":
        return {
            "inputs": [{"name": "user_item", "shape": {"type": "any"}, "source": "unknown"}],
            "outputs": [{"name": "score", "shape": _shape("tensor", ["B"]), "source": "default"}],
            "objective": flags.get("objective", "ranking_or_matching"),
            "requires": [],
        }

    inputs, masks = _inputer_contract(inputer_name, inputer_index)
    output = _operator_output(info, all_classes)
    return {
        "inputs": inputs,
        "masks": masks,
        "outputs": [output],
        "requires": [],
    }


def _inputer_contract(inputer_name: str | None, inputer_index: Mapping[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if inputer_name == "SimpleInputer":
        return (
            [{"name": "embeddings", "shape": {"type": "dict", "key": "col", "value": _shape("tensor", ["N", "Lc", "input_dim"])}, "source": "inputer"}],
            [{"name": "mask", "shape": {"type": "dict", "key": "col", "value": _shape("tensor", ["N", "Lc"])}, "source": "inputer"}],
        )
    if inputer_name in {"ConcatInputer", "FlattenSeqInputer", "SingleColumnInputer"}:
        length_axis = "L_flat" if inputer_name == "FlattenSeqInputer" else "L"
        return (
            [{"name": "embeddings", "shape": _shape("tensor", ["N", length_axis, "input_dim"]), "source": "inputer"}],
            [{"name": "mask", "shape": _shape("tensor", ["N", length_axis]), "source": "inputer"}],
        )
    return (
        [{"name": "embeddings", "shape": {"type": "operator-specific"}, "source": "inputer"}],
        [{"name": "mask", "shape": {"type": "operator-specific"}, "source": "inputer"}],
    )


def _operator_output(info: ClassInfo, all_classes: Mapping[str, ClassInfo]) -> dict[str, Any]:
    policy = _output_dim_policy(info, all_classes)
    axes = ["N", "hidden_size"]
    if policy == "config.input_dim":
        axes = ["N", "input_dim"]
    elif "num_columns" in policy or "num_fields" in policy:
        axes = ["N", "hidden_size * num_fields"]
    return {
        "name": "embedding",
        "kind": "vector",
        "shape": _shape("tensor", axes),
        "dimPolicy": policy,
        "source": "output_dim" if policy != "config.hidden_size" else "default",
    }


def _output_dim_policy(info: ClassInfo, all_classes: Mapping[str, ClassInfo]) -> str:
    expression = _find_output_dim_return(info)
    if expression:
        return _normalize_output_dim_expr(expression)
    for base in info.bases:
        parent = all_classes.get(base)
        if parent:
            parent_policy = _output_dim_policy(parent, all_classes)
            if parent_policy:
                return parent_policy
    return "config.hidden_size"


def _find_output_dim_return(info: ClassInfo) -> str | None:
    for stmt in info.node.body:
        if not isinstance(stmt, ast.FunctionDef) or stmt.name != "output_dim":
            continue
        for child in ast.walk(stmt):
            if isinstance(child, ast.Return) and child.value is not None:
                return ast.get_source_segment(info.source, child.value)
    return None


def _normalize_output_dim_expr(expression: str) -> str:
    return (
        expression
        .replace("self.", "")
        .replace("config.", "config.")
        .replace("num_columns", "num_fields")
    )


def _resolve_config_params(config_name: str | None, configs: Mapping[str, ConfigInfo], kind: str) -> list[dict[str, Any]]:
    if not config_name or config_name not in configs:
        return []
    seen = set()
    params = []
    for param in _config_params_with_bases(config_name, configs, kind):
        if param["key"] in seen:
            continue
        seen.add(param["key"])
        params.append(param)
    return params


def _config_params_with_bases(config_name: str, configs: Mapping[str, ConfigInfo], kind: str) -> list[dict[str, Any]]:
    info = configs.get(config_name)
    if not info:
        return []
    params = []
    for base in info.bases:
        params.extend(_config_params_with_bases(base, configs, kind))
    skip = BASE_OPERATOR_CONFIG_KEYS if kind == "operator" else BASE_PREDICTOR_CONFIG_KEYS
    params.extend(param for param in info.params if param["key"] not in skip)
    return params


def _own_config_params(node: ast.ClassDef, source: str, kind: str) -> list[dict[str, Any]]:
    init_fn = next((stmt for stmt in node.body if isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"), None)
    if not init_fn:
        return []
    args = init_fn.args.args
    defaults = [None] * (len(args) - len(init_fn.args.defaults)) + list(init_fn.args.defaults)
    params = []
    skip = BASE_OPERATOR_CONFIG_KEYS if kind == "operator" else BASE_PREDICTOR_CONFIG_KEYS
    for arg, default_node in zip(args, defaults):
        if arg.arg == "self" or arg.arg in skip:
            continue
        data_type = _annotation_name(arg.annotation)
        default = _literal(default_node) if default_node is not None else None
        ui_type = _ui_type(data_type, default)
        param = {
            "key": arg.arg,
            "label": arg.arg,
            "type": ui_type,
            "dataType": data_type or _data_type_from_default(default),
            "source": "config_signature",
        }
        if default_node is not None:
            param["default"] = default
        params.append(param)
    return params


def _class_assignments(node: ast.ClassDef, source: str) -> dict[str, Any]:
    assignments = {}
    for stmt in node.body:
        if not isinstance(stmt, ast.Assign):
            continue
        for target in stmt.targets:
            if isinstance(target, ast.Name):
                assignments[target.id] = _literal_or_name(stmt.value, source)
    return assignments


def _resolve_attr(info: ClassInfo, all_classes: Mapping[str, ClassInfo], attr: str, default: Any = None) -> Any:
    if attr in info.assignments:
        return info.assignments[attr]
    for base in info.bases:
        parent = all_classes.get(base)
        if parent:
            value = _resolve_attr(parent, all_classes, attr, None)
            if value is not None:
                return value
    return default


def _parse_legommenders_doc(docstring: str | None) -> dict[str, Any]:
    if not docstring or "Legommenders:" not in docstring:
        return {}
    _, payload = docstring.split("Legommenders:", 1)
    payload = payload.strip()
    try:
        return json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Legommenders docstring JSON: {exc}") from exc


def _merge_docstring(doc: dict[str, Any], docstring_meta: Mapping[str, Any]) -> dict[str, Any]:
    if not docstring_meta:
        return doc
    result = dict(doc)
    if "summary" in docstring_meta:
        result["explain"] = docstring_meta["summary"]
    for key in ("roles", "notes", "warnings"):
        if key in docstring_meta:
            result[key] = docstring_meta[key]
    if "contract" in docstring_meta:
        result["contract"] = _deep_merge(result["contract"], docstring_meta["contract"])
    return result


def _merge_override(doc: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = _deep_merge(doc, {key: value for key, value in override.items() if key != "params"})
    param_overrides = override.get("params", {})
    if param_overrides:
        result["params"] = [_deep_merge(param, param_overrides.get(param["key"], {})) for param in result["params"]]
        known = {param["key"] for param in result["params"]}
        for key, value in param_overrides.items():
            if key not in known:
                result["params"].append({"key": key, "label": key, **value})
        result["defaults"] = {param["key"]: param["default"] for param in result["params"] if "default" in param}
    return result


def _add_display_contract_fields(contract: dict[str, Any]) -> None:
    inputs = contract.get("inputs") or []
    masks = contract.get("masks") or []
    outputs = contract.get("outputs") or []
    if inputs:
        contract["inputShape"] = _format_ports(inputs)
    if masks:
        contract["maskShape"] = _format_ports(masks)
    if outputs:
        contract["outputShape"] = _format_ports(outputs)
        first = outputs[0]
        if "kind" in first:
            contract["outputKind"] = first["kind"]
        if "dimPolicy" in first:
            contract["outputDimPolicy"] = first["dimPolicy"]


def _format_ports(ports: list[Mapping[str, Any]]) -> str:
    if len(ports) == 1:
        return _format_shape(ports[0].get("shape", {}))
    return ", ".join(f"{port.get('name')}: {_format_shape(port.get('shape', {}))}" for port in ports)


def _format_shape(shape: Mapping[str, Any]) -> str:
    if "oneOf" in shape:
        return " or ".join(_format_shape(item) for item in shape["oneOf"])
    shape_type = shape.get("type")
    if shape_type == "tensor":
        return f"tensor[{', '.join(shape.get('axes', []))}]"
    if shape_type == "dict":
        if "fields" in shape:
            fields = ", ".join(f"{key}: {_format_shape(value)}" for key, value in shape["fields"].items())
            return f"dict{{{fields}}}"
        if "value" in shape:
            key = shape.get("key", "key")
            return f"dict[{key} -> {_format_shape(shape['value'])}]"
        if "keys" in shape:
            return f"dict{{{', '.join(shape['keys'])}}}"
        return "dict"
    return str(shape_type or "unknown")


def _shape(shape_type: str, axes: list[str]) -> dict[str, Any]:
    return {"type": shape_type, "axes": axes}


def _read_overrides(path: Path | None) -> dict[str, Any]:
    if not path or not path.exists():
        return {}
    return json.loads(path.read_text())


def _literal_or_name(node: ast.AST, source: str) -> Any:
    value = _literal(node)
    if value is not None:
        return value
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _name_of(node)
    return ast.get_source_segment(source, node)


def _literal(node: ast.AST | None) -> Any:
    if node is None:
        return None
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return None


def _name_of(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _name_of(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    return ""


def _annotation_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    if isinstance(node, ast.Constant):
        return str(node.value)
    return None


def _ui_type(data_type: str | None, default: Any) -> str:
    if data_type == "bool" or isinstance(default, bool):
        return "boolean"
    if data_type in {"int", "float"} or isinstance(default, (int, float)):
        return "number"
    if data_type in {"list", "tuple", "List"} or isinstance(default, (list, tuple)):
        return "list"
    return "text"


def _data_type_from_default(default: Any) -> str | None:
    if default is None:
        return None
    if isinstance(default, bool):
        return "bool"
    if isinstance(default, int):
        return "int"
    if isinstance(default, float):
        return "float"
    if isinstance(default, str):
        return "str"
    if isinstance(default, list):
        return "list"
    return type(default).__name__


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


def _module_path(path: Path) -> str:
    return ".".join(path.with_suffix("").parts[-3:])


def _safe_relative_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root))
    except ValueError:
        return path.name


def _deep_merge(base: Dict[str, Any], override: Mapping[str, Any]) -> Dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result
