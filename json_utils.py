import re
import json
import pandas as pd
from typing import Any, List, Dict, Union

def deep_flatten_json(record: dict, parent_key: str = "", sep: str = ".") -> dict:
    """Recursively flatten nested dicts and lists of dicts."""
    items: Dict[str, any] = {}
    for k, v in record.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(deep_flatten_json(v, new_key, sep))
        elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
            for idx, item in enumerate(v):
                for sub_k, sub_v in deep_flatten_json(item, "", sep).items():
                    items[f"{new_key}{sep}{idx}{sep}{sub_k}"] = sub_v
        else:
            items[new_key] = json.dumps(v) if isinstance(v, list) else v
    return items

def prepare_json_df_for_sql(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any remaining list or dict columns to JSON strings for SQLite."""
    for col in df.columns:
        if df[col].apply(lambda x: isinstance(x, (list, dict))).any():
            df[col] = df[col].apply(
                lambda x: json.dumps(x) if isinstance(x, (list, dict)) else x
            )
    return df

def flatten_json_schema(schema: Union[Dict[str, Any], List[Any]]) -> pd.DataFrame:
    """Recursively flatten a JSON Schema into a structured DataFrame."""
    rows: List[Dict[str, Any]] = []
    
    def recurse(obj: Dict[str, Any], path: List[str]) -> None:
        if not isinstance(obj, dict): return
        props = obj.get("properties")
        if not isinstance(props, dict): return
        reqs = obj.get("required", [])

        for prop_name, prop_def in props.items():
            prop_def = prop_def if isinstance(prop_def, dict) else {}
            full_path = ".".join(path + [prop_name])
            
            t = prop_def.get("type") or (prop_def["$ref"].rsplit("/", 1)[-1] if "$ref" in prop_def else "object")
            t = ",".join(map(str, t)) if isinstance(t, list) else t

            rows.append({
                "field": full_path, "type": t,
                "required": prop_name in reqs, "description": prop_def.get("description", "")
            })

            if t == "object" and isinstance(prop_def.get("properties"), dict):
                recurse(prop_def, path + [prop_name])
            
            items = prop_def.get("items")
            if isinstance(items, dict):
                item_type = items.get("type")
                if (item_type == "object" and "properties" in items) or "$ref" in items:
                    recurse(items, path + [f"{prop_name}[]"])

    # This logic is now updated to look inside 'definitions' or 'components/schemas'
    if isinstance(schema, dict):
        defs = schema.get("definitions", schema.get("components", {}).get("schemas"))
        if isinstance(defs, dict):
            for def_name, def_obj in defs.items():
                recurse(def_obj, [def_name])
        else:
            # Fallback for simple schemas with only top-level properties
            recurse(schema, [])
    elif isinstance(schema, list):
        for idx, item in enumerate(schema):
            if isinstance(item, dict): recurse(item, [str(idx)])
    else:
        raise ValueError("Schema must be a dict or list of dicts")
        
    return pd.DataFrame(rows, columns=["field", "type", "required", "description"])
def parse_json_schema(schema: Dict) -> pd.DataFrame:
    """Parses JSON Schema definitions into a DataFrame."""
    records = []
    defs = schema.get("definitions", schema.get("components", {}).get("schemas", {}))
    if "properties" in schema:
        defs["<root>"] = {"properties": schema.get("properties", {}), "required": schema.get("required", [])}

    for cls, cls_def in defs.items():
        for prop, prop_def in cls_def.get("properties", {}).items():
            t = prop_def.get("$ref", "#/").split("/")[-1] if "$ref" in prop_def else prop_def.get("type", "object")
            records.append({
                "class": cls, "property": prop, "type": t,
                "required": prop in cls_def.get("required", []),
                "description": prop_def.get("description", "")
            })
    return pd.DataFrame(records)

def generate_class_diagram_dot(schema: Dict) -> str:
    """Creates a basic Graphviz DOT string from a JSON schema."""
    defs = schema.get("definitions", schema.get("components", {}).get("schemas", {}))
    if "properties" in schema:
        defs["<root>"] = {"properties": schema.get("properties", {}), "required": schema.get("required", [])}
    
    dot = ["digraph G {", "  node [shape=record];"]
    refs = []
    for cls, cls_def in defs.items():
        node_id = re.sub(r"[^0-9A-Za-z_]", "_", cls)
        fields = []
        for prop, prop_def in cls_def.get("properties", {}).items():
            t = prop_def.get("$ref", "#/").split("/")[-1] if "$ref" in prop_def else prop_def.get("type", "object")
            fields.append(f"{prop}: {t}")
            if "$ref" in prop_def:
                refs.append((node_id, re.sub(r"[^0-9A-Za-z_]", "_", t)))
        
        label = "{" + cls + "|" + "\\l".join(fields) + "\\l}"
        dot.append(f'  "{node_id}" [label="{label}"];')

    for src, tgt in refs:
        dot.append(f'  "{src}" -> "{tgt}";')
    dot.append("}")
    return "\n".join(dot)

def generate_advanced_class_diagram_dot(schema):
    def traverse_schema(name, obj, definitions, parent=None, connections=None):
        if connections is None:
            connections = []
        if isinstance(obj, dict):
            properties = obj.get('properties', {})
            definitions[name] = properties
            if parent:
                connections.append((parent, name))
            for prop, details in properties.items():
                if details.get('type') == 'object':
                    child_name = prop.capitalize()
                    traverse_schema(child_name, details, definitions, name, connections)
        return definitions, connections

    definitions, connections = traverse_schema('Root', schema, {})

    dot = "digraph G {\nnode [shape=record];\n"

    for class_name, props in definitions.items():
        fields = "\\l".join([f"{key}: {value.get('type', 'object')}" for key, value in props.items()])
        dot += f'{class_name} [label="{{{class_name}|{fields}\\l}}"];\n'

    for parent, child in connections:
        dot += f'{parent} -> {child};\n'

    dot += "}"
    return dot

def find_keys_recursive(obj, search_key, current_path=""):
    matches = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            path = f"{current_path}.{k}" if current_path else k
            if search_key.lower() in k.lower():
                matches.append(path)
            matches.extend(find_keys_recursive(v, search_key, path))
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            path = f"{current_path}[{idx}]"
            matches.extend(find_keys_recursive(item, search_key, path))
    return matches
