"""Generate a flat parameter table by introspecting pydantic model classes directly.

This avoids the complexity of walking JSON schema with $ref resolution,
oneOf/anyOf handling, etc. Instead, it uses the class hierarchy that already
encodes inheritance (shared fields), Union types (variants), and Optional
(optional sections).
"""

import json
from typing import Optional, Union, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo


def _is_union(annotation) -> bool:
    return get_origin(annotation) is Union


def _is_optional(annotation) -> bool:
    """Check if a type annotation is Optional[X] (i.e. Union[X, None])."""
    if not _is_union(annotation):
        return False
    return type(None) in get_args(annotation)


def _get_union_members(annotation) -> list[type]:
    """Get non-None members of a Union type."""
    return [a for a in get_args(annotation) if a is not type(None)]


def _is_model(cls) -> bool:
    return isinstance(cls, type) and issubclass(cls, BaseModel)


def _is_list_of(annotation):
    """If annotation is list[X], return X. Otherwise return None."""
    origin = get_origin(annotation)
    if origin is list:
        args = get_args(annotation)
        return args[0] if args else None
    return None


def _get_variant_name(model_cls: type[BaseModel]) -> Optional[str]:
    """Extract the variant name from a Literal[...] 'name' or 'kind' field."""
    for disc_field in ("name", "kind", "processing"):
        if disc_field in model_cls.model_fields:
            info = model_cls.model_fields[disc_field]
            args = get_args(info.annotation)
            # Literal["gmnn"] -> args = ("gmnn",)
            if args and all(isinstance(a, str) for a in args):
                return args[0]
    return None


def _unwrap_annotated(annotation):
    """Strip typing.Annotated wrapper, returning the base type."""
    if hasattr(annotation, "__metadata__"):  # Annotated[X, ...]
        return get_args(annotation)[0]
    return annotation


def _type_str(annotation) -> str:
    """Human-readable type string from a type annotation."""
    from typing import Literal

    annotation = _unwrap_annotated(annotation)
    origin = get_origin(annotation)

    if origin is list:
        return "array"

    if get_origin(annotation) is Literal:
        vals = get_args(annotation)
        if len(vals) == 1:
            return repr(vals[0])
        return "|".join(str(v) for v in vals)

    # Optional[X] -> type_str(X)|null
    if _is_optional(annotation):
        members = _get_union_members(annotation)
        if len(members) == 1:
            return f"{_type_str(members[0])}|null"

    # Union of non-model types (e.g. str | Path)
    if _is_union(annotation):
        members = [a for a in get_args(annotation) if a is not type(None)]
        non_model = [m for m in members if not _is_model(m)]
        if non_model:
            return "|".join(_type_str(m) for m in non_model)

    # Simple types
    import pathlib

    type_map = {
        int: "integer", float: "number", str: "string", bool: "boolean",
        dict: "object", list: "array", pathlib.Path: "string",
    }
    if annotation in type_map:
        return type_map[annotation]

    # Pydantic constrained types (PositiveInt, NonNegativeFloat, etc.)
    if isinstance(annotation, type):
        for base, name in type_map.items():
            if issubclass(annotation, base):
                return name

    # types.UnionType (Python 3.10+ `X | Y` syntax)
    import types as _types

    if isinstance(annotation, _types.UnionType):
        members = [a for a in get_args(annotation) if a is not type(None)]
        return "|".join(_type_str(m) for m in members)

    return str(annotation)


def _default_str(field_info: FieldInfo) -> str:
    """Compact default value string."""
    if field_info.is_required():
        return "REQUIRED"
    val = field_info.default
    if val is None:
        return "null"
    if _is_model(type(val)):
        return str(type(val).__name__)
    try:
        s = json.dumps(val)
    except (TypeError, ValueError):
        s = repr(val)
    return s[:30] + "..." if len(s) > 30 else s


def _description(field_info: FieldInfo) -> str:
    """First line of description, truncated."""
    desc = field_info.description or ""
    return desc.split("\n")[0][:60]


def _find_shared_base(variants: list[type[BaseModel]]) -> Optional[type[BaseModel]]:
    """Find the common base class that holds shared fields (if any).

    Returns the most specific common ancestor that is not BaseModel itself.
    """
    if not variants:
        return None
    bases_per_variant = []
    for v in variants:
        bases_per_variant.append(
            [b for b in v.__mro__ if _is_model(b) and b is not BaseModel]
        )
    # Intersect MROs
    common = set(bases_per_variant[0])
    for bases in bases_per_variant[1:]:
        common &= set(bases)
    # Remove the variant classes themselves
    common -= set(variants)
    if not common:
        return None
    # Pick the most specific (closest to variants in MRO)
    # Sort by MRO position of first variant
    first_mro = bases_per_variant[0]
    common_sorted = sorted(common, key=lambda c: first_mro.index(c))
    return common_sorted[0]


def _flatten_model(
    model_cls: type[BaseModel],
    prefix: str = "",
    _seen: set | None = None,
    exclude_fields: set[str] | None = None,
) -> list[tuple[str, str, str, str, bool]]:
    """Flatten a pydantic model into (path, type, default, description, is_section) rows."""
    if _seen is None:
        _seen = set()
    if exclude_fields is None:
        exclude_fields = set()

    if model_cls in _seen:
        return []
    _seen.add(model_cls)

    rows = []

    for field_name, field_info in model_cls.model_fields.items():
        if field_name in exclude_fields:
            continue

        path = f"{prefix}.{field_name}" if prefix else field_name
        annotation = _unwrap_annotated(field_info.annotation)

        # --- Union of model classes (discriminated union) ---
        if _is_union(annotation):
            members = _get_union_members(annotation)
            model_members = [m for m in members if _is_model(m)]

            if len(model_members) > 1:
                # Discriminated union
                optional = _is_optional(annotation)
                names = [_get_variant_name(m) or m.__name__ for m in model_members]
                info_parts = []
                if optional:
                    info_parts.append("optional")
                info_parts.append(f"default: {_default_str(field_info)}")
                info = ", ".join(info_parts)
                rows.append((path, f"[{info}]", "", f"variants: {', '.join(names)}", True))

                # Find shared base and emit shared fields once
                shared_base = _find_shared_base(model_members)
                if shared_base:
                    shared_fields = set(shared_base.model_fields.keys())
                    rows.extend(_flatten_model(shared_base, path, _seen.copy(), set()))
                else:
                    shared_fields = set()

                # Emit variant-specific fields
                for m in model_members:
                    vname = _get_variant_name(m) or m.__name__
                    vpath = f"{path}.{vname}"
                    rows.extend(
                        _flatten_model(m, vpath, _seen.copy(), exclude_fields=shared_fields)
                    )
                continue

            elif len(model_members) == 1:
                # Optional[SomeModel] — single optional section
                m = model_members[0]
                optional = _is_optional(annotation)
                if optional:
                    rows.append((path, f"[optional, default: {_default_str(field_info)}]", "", "", True))
                rows.extend(_flatten_model(m, path, _seen.copy()))
                continue

        # --- List of models ---
        item_type = _is_list_of(annotation)
        if item_type is not None:
            actual_item = _unwrap_annotated(item_type)

            if _is_union(actual_item):
                members = _get_union_members(actual_item)
                model_members = [m for m in members if _is_model(m)]
            elif _is_model(actual_item):
                model_members = [actual_item]
            else:
                model_members = []

            if model_members:
                names = [_get_variant_name(m) or m.__name__ for m in model_members]
                desc = f"variants: {', '.join(names)}" if len(model_members) > 1 else ""
                rows.append((f"{path}[]", f"[list, default: {_default_str(field_info)}]", "", desc, True))

                if len(model_members) > 1:
                    shared_base = _find_shared_base(model_members)
                    shared_fields = set(shared_base.model_fields.keys()) if shared_base else set()
                    if shared_base:
                        rows.extend(_flatten_model(shared_base, f"{path}[]", _seen.copy(), set()))
                    for m in model_members:
                        vname = _get_variant_name(m) or m.__name__
                        rows.extend(
                            _flatten_model(m, f"{path}[].{vname}", _seen.copy(), exclude_fields=shared_fields)
                        )
                else:
                    rows.extend(_flatten_model(model_members[0], f"{path}[]", _seen.copy()))
                continue

        # --- Nested single model ---
        if _is_model(annotation):
            rows.extend(_flatten_model(annotation, path, _seen.copy()))
            continue

        # --- Leaf field ---
        rows.append((path, _type_str(annotation), _default_str(field_info), _description(field_info), False))

    _seen.discard(model_cls)
    return rows


def print_flat(model_cls: type[BaseModel]):
    """Print all parameters of a pydantic config as a flat table."""
    rows = _flatten_model(model_cls)
    if not rows:
        return
    leaf_rows = [r for r in rows if not r[4]]
    if leaf_rows:
        w_path = max(len(r[0]) for r in leaf_rows)
        w_type = max(len(r[1]) for r in leaf_rows)
        w_default = max(len(r[2]) for r in leaf_rows)
    else:
        w_path = w_type = w_default = 10
    for path, type_str, default_str, desc, is_section in rows:
        if is_section:
            line = f"\n{path:<{w_path}}  {type_str}"
            if desc:
                line += f"  {desc}"
        else:
            line = f"{path:<{w_path}}  {type_str:<{w_type}}  {default_str:<{w_default}}"
            if desc:
                line += f"  {desc}"
        print(line)
