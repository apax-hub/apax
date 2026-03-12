"""JSON schema navigation helpers for --keywords and section drill-down.

These operate on the raw JSON schema dict produced by pydantic's
model_json_schema(), resolving $ref pointers and walking dotted paths.
"""

import json


def resolve_ref(node, defs):
    """Resolve a single $ref pointer."""
    if isinstance(node, dict) and "$ref" in node:
        ref_name = node["$ref"].split("/")[-1]
        if ref_name in defs:
            return defs[ref_name]
    return node


def shallow_resolve(node, defs):
    """Resolve $ref pointers one level deep (no recursion into children)."""
    node = resolve_ref(node, defs)
    if not isinstance(node, dict):
        return node

    result = {}
    for k, v in node.items():
        if k == "properties" and isinstance(v, dict):
            result[k] = {pk: resolve_ref(pv, defs) for pk, pv in v.items()}
        elif k in ("oneOf", "anyOf") and isinstance(v, list):
            result[k] = [resolve_ref(item, defs) for item in v]
        else:
            result[k] = resolve_ref(v, defs) if isinstance(v, dict) else v
    return result


def find_variant(node, name, defs):
    """Find a named variant in a oneOf/anyOf discriminated union."""
    for key in ("oneOf", "anyOf"):
        variants = node.get(key, [])
        for variant in variants:
            resolved = resolve_ref(variant, defs)
            title = resolved.get("title", "").lower().replace("config", "").replace("options", "")
            variant_name_const = None
            for prop in resolved.get("properties", {}).values():
                if "const" in prop:
                    variant_name_const = prop["const"]
                    break
            if name.lower() in (
                title.strip(),
                (variant_name_const or "").lower(),
                resolved.get("title", "").lower(),
            ):
                return resolved
    return None


def filter_schema(schema, section):
    """Extract a section from the schema, supporting dotted paths like 'model.gmnn'.

    Returns (resolved_node, None) on success, or (None, error_message) on failure.
    """
    defs = schema.get("$defs", {})
    props = schema.get("properties", {})

    parts = section.split(".")
    first = parts[0]

    if first in props:
        node = resolve_ref(props[first], defs)
    elif first in defs:
        node = defs[first]
    else:
        available = sorted(list(props.keys()) + list(defs.keys()))
        return None, f"Section '{first}' not found. Available: {', '.join(available)}"

    for part in parts[1:]:
        node = resolve_ref(node, defs)
        node_props = node.get("properties", {})
        if part in node_props:
            node = resolve_ref(node_props[part], defs)
            continue
        variant = find_variant(node, part, defs)
        if variant is not None:
            node = variant
            continue
        available_parts = sorted(node_props.keys())
        for key in ("oneOf", "anyOf"):
            for v in node.get(key, []):
                resolved = resolve_ref(v, defs)
                t = resolved.get("title", "")
                if t:
                    available_parts.append(t)
        prefix = ".".join(parts[: parts.index(part)])
        return None, f"'{part}' not found in '{prefix}'. Available: {', '.join(sorted(available_parts))}"

    return shallow_resolve(node, defs), None


def list_keywords(node, defs):
    """List navigable keywords at the current schema node."""
    node = resolve_ref(node, defs)
    keywords = []
    for name, prop in node.get("properties", {}).items():
        resolved = resolve_ref(prop, defs)
        desc = resolved.get("description", "")
        short = desc.split("\n")[0][:80] if desc else ""
        keywords.append((name, short))
    for key in ("oneOf", "anyOf"):
        for variant in node.get(key, []):
            resolved = resolve_ref(variant, defs)
            title = resolved.get("title", "")
            variant_name = None
            for prop in resolved.get("properties", {}).values():
                if "const" in prop:
                    variant_name = prop["const"]
                    break
            desc = resolved.get("description", "").split("\n")[0][:80]
            keywords.append((variant_name or title, desc))
    return keywords


def print_keywords(schema, section, defs):
    """Resolve path and print keywords at that level."""
    if section:
        node, error = filter_schema(schema, section)
        if error:
            return error
    else:
        node = schema
    keywords = list_keywords(node, defs)
    if not keywords:
        print(json.dumps(shallow_resolve(node, defs), indent=2))
        return None
    for name, desc in keywords:
        if desc:
            print(f"  {name:30s} {desc}")
        else:
            print(f"  {name}")
    return None
