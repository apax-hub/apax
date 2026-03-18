"""JSON schema navigation for --keywords and section drill-down."""

import json


def _resolve(node, defs):
    """Resolve a single $ref pointer."""
    if isinstance(node, dict) and "$ref" in node:
        ref_name = node["$ref"].split("/")[-1]
        if ref_name in defs:
            return defs[ref_name]
    return node


def _shallow_resolve(node, defs):
    """Resolve $ref pointers one level deep."""
    node = _resolve(node, defs)
    if not isinstance(node, dict):
        return node
    result = {}
    for k, v in node.items():
        if k == "properties" and isinstance(v, dict):
            result[k] = {pk: _resolve(pv, defs) for pk, pv in v.items()}
        elif k in ("oneOf", "anyOf") and isinstance(v, list):
            result[k] = [_resolve(item, defs) for item in v]
        else:
            result[k] = _resolve(v, defs) if isinstance(v, dict) else v
    return result


def _iter_variants(node, defs):
    """Yield (discriminator_name, resolved_node) for oneOf/anyOf variants."""
    for key in ("oneOf", "anyOf"):
        for variant in node.get(key, []):
            resolved = _resolve(variant, defs)
            vname = None
            for prop in resolved.get("properties", {}).values():
                if "const" in prop:
                    vname = prop["const"]
                    break
            yield vname or resolved.get("title", ""), resolved


def filter_schema(schema, section):
    """Walk dotted path, return (resolved_node, None) or (None, error)."""
    defs = schema.get("$defs", {})
    props = schema.get("properties", {})
    parts = section.split(".")

    first = parts[0]
    if first in props:
        node = _resolve(props[first], defs)
    elif first in defs:
        node = defs[first]
    else:
        available = sorted(list(props.keys()) + list(defs.keys()))
        return None, f"Section '{first}' not found. Available: {', '.join(available)}"

    for part in parts[1:]:
        node = _resolve(node, defs)
        node_props = node.get("properties", {})
        if part in node_props:
            node = _resolve(node_props[part], defs)
            continue
        # Try matching as a variant name
        matched = None
        for vname, resolved in _iter_variants(node, defs):
            title = resolved.get("title", "").lower().replace("config", "").replace("options", "")
            if part.lower() in (title.strip(), vname.lower(), resolved.get("title", "").lower()):
                matched = resolved
                break
        if matched:
            node = matched
            continue
        available_parts = sorted(node_props.keys()) + [n for n, _ in _iter_variants(node, defs)]
        prefix = ".".join(parts[: parts.index(part)])
        return None, f"'{part}' not found in '{prefix}'. Available: {', '.join(sorted(available_parts))}"

    return _shallow_resolve(node, defs), None


def print_keywords(schema, section):
    """Print navigable keywords at the given path. Returns error string or None."""
    defs = schema.get("$defs", {})
    if section:
        node, error = filter_schema(schema, section)
        if error:
            return error
    else:
        node = _resolve(schema, defs)

    keywords = []
    for name, prop in node.get("properties", {}).items():
        resolved = _resolve(prop, defs)
        desc = resolved.get("description", "")
        keywords.append((name, desc.split("\n")[0][:80] if desc else ""))
    for vname, resolved in _iter_variants(node, defs):
        desc = resolved.get("description", "").split("\n")[0][:80]
        keywords.append((vname, desc))

    if not keywords:
        print(json.dumps(_shallow_resolve(node, defs), indent=2))
        return None
    for name, desc in keywords:
        print(f"  {name:30s} {desc}" if desc else f"  {name}")
    return None
