"""Flat parameter table from pydantic model introspection."""

import json
import pathlib
import types as _types
from typing import Literal, Union, get_args, get_origin

from pydantic import BaseModel

_TYPE_MAP = {
    int: "integer", float: "number", str: "string", bool: "boolean",
    dict: "object", list: "array", pathlib.Path: "string",
}


def _unwrap(annotation):
    """Strip Annotated wrapper."""
    return get_args(annotation)[0] if hasattr(annotation, "__metadata__") else annotation


def _is_model(cls):
    return isinstance(cls, type) and issubclass(cls, BaseModel)


def _is_union(annotation):
    return get_origin(annotation) is Union or isinstance(annotation, _types.UnionType)


def _classify(annotation):
    """Classify a field annotation into (model_classes, is_list, is_optional)."""
    annotation = _unwrap(annotation)
    is_list = get_origin(annotation) is list
    if is_list:
        args = get_args(annotation)
        annotation = _unwrap(args[0]) if args else None
        if annotation is None:
            return [], True, False

    if _is_union(annotation):
        args = get_args(annotation)
        return [a for a in args if _is_model(a)], is_list, type(None) in args
    if _is_model(annotation):
        return [annotation], is_list, False
    return [], is_list, False


def _variant_name(model_cls):
    """Extract variant name from a Literal 'name'/'kind'/'processing' field."""
    for key in ("name", "kind", "processing"):
        if key in model_cls.model_fields:
            args = get_args(model_cls.model_fields[key].annotation)
            if args and all(isinstance(a, str) for a in args):
                return args[0]
    return None


def _type_str(annotation):
    """Human-readable type string."""
    annotation = _unwrap(annotation)
    origin = get_origin(annotation)

    if origin is list:
        return "array"
    if origin is Literal:
        vals = get_args(annotation)
        return repr(vals[0]) if len(vals) == 1 else "|".join(str(v) for v in vals)
    if _is_union(annotation):
        args = get_args(annotation)
        has_none = type(None) in args
        members = [a for a in args if a is not type(None)]
        if has_none and len(members) == 1:
            return f"{_type_str(members[0])}|null"
        non_model = [m for m in members if not _is_model(m)]
        if non_model:
            return "|".join(_type_str(m) for m in non_model)

    if annotation in _TYPE_MAP:
        return _TYPE_MAP[annotation]
    if isinstance(annotation, type):
        for base, name in _TYPE_MAP.items():
            if issubclass(annotation, base):
                return name
    return str(annotation)


def _default_str(field_info):
    """Compact default value string."""
    if field_info.is_required():
        return "REQUIRED"
    val = field_info.default
    if val is None:
        return "null"
    if _is_model(type(val)):
        return type(val).__name__
    try:
        s = json.dumps(val)
    except (TypeError, ValueError):
        s = repr(val)
    return s[:30] + "..." if len(s) > 30 else s


def _shared_base(variants):
    """Most specific common BaseModel ancestor (excluding BaseModel and variants themselves)."""
    if len(variants) < 2:
        return None
    mros = [[b for b in v.__mro__ if _is_model(b) and b is not BaseModel] for v in variants]
    common = set(mros[0]).intersection(*mros[1:]) - set(variants)
    return min(common, key=lambda c: mros[0].index(c)) if common else None


def _flatten(cls, prefix="", _seen=None, exclude=frozenset()):
    """Recursively flatten a pydantic model into (path, type, default, desc, is_section) rows."""
    if _seen is None:
        _seen = set()
    if cls in _seen:
        return []
    _seen.add(cls)
    rows = []

    for field_name, info in cls.model_fields.items():
        if field_name in exclude:
            continue
        path = f"{prefix}.{field_name}" if prefix else field_name
        models, is_list, is_optional = _classify(info.annotation)

        if models:
            base_path = f"{path}[]" if is_list else path

            # Emit section header when structure is non-trivial
            if len(models) > 1 or is_list or is_optional:
                parts = []
                if is_list:
                    parts.append("list")
                if is_optional:
                    parts.append("optional")
                parts.append(f"default: {_default_str(info)}")
                names = [_variant_name(m) or m.__name__ for m in models]
                desc = f"variants: {', '.join(names)}" if len(models) > 1 else ""
                rows.append((base_path, f"[{', '.join(parts)}]", "", desc, True))

            if len(models) > 1:
                base = _shared_base(models)
                shared = set(base.model_fields.keys()) if base else set()
                if base:
                    rows.extend(_flatten(base, base_path, _seen.copy()))
                for m in models:
                    vname = _variant_name(m) or m.__name__
                    rows.extend(_flatten(m, f"{base_path}.{vname}", _seen.copy(), exclude=shared))
            else:
                rows.extend(_flatten(models[0], base_path, _seen.copy()))
        else:
            ann = _unwrap(info.annotation)
            desc = (info.description or "").split("\n")[0][:60]
            rows.append((path, _type_str(ann), _default_str(info), desc, False))

    _seen.discard(cls)
    return rows


def print_flat(model_cls):
    """Print all parameters as a flat table."""
    rows = _flatten(model_cls)
    if not rows:
        return
    leaves = [r for r in rows if not r[4]]
    wp = max((len(r[0]) for r in leaves), default=10)
    wt = max((len(r[1]) for r in leaves), default=10)
    wd = max((len(r[2]) for r in leaves), default=10)
    for path, typ, default, desc, is_section in rows:
        if is_section:
            line = f"\n{path:<{wp}}  {typ}"
        else:
            line = f"{path:<{wp}}  {typ:<{wt}}  {default:<{wd}}"
        if desc:
            line += f"  {desc}"
        print(line)
