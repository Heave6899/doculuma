import os
import json
import re
import pandas as pd
import requests
from pathlib import Path
import glob
# Constants
MODULE_FOLDER = "data_saves/module_defs"
REMOTE_ALL_MODULES_URL = "https://nyl-uapp-staging.unqork.io/fbu/uapi/modules"
REMOTE_SINGLE_MODULE_URL = "https://nyl-uapp-staging.unqork.io/fbu/form"

# Ensure folder exists
os.makedirs(MODULE_FOLDER, exist_ok=True)

def save_module_to_folder(mod_data):
    """Save one module's full definition to a local JSON file in MODULE_FOLDER."""
    mid = mod_data.get("_id")
    if not mid:
        return
    out_path = os.path.join(MODULE_FOLDER, f"{mid}.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(mod_data, f, indent=2)
    except Exception as e:
        print(f"[ERROR] Could not save module {mid} to {out_path}: {e}")

def fetch_list_of_modules(token):
    """Fetch all module summaries from Unqork API."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    all_modules = []
    offset = 0
    batch_size = 50
    while True:
        url = f"{REMOTE_ALL_MODULES_URL}?offset={offset}"
        resp = requests.get(url, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if not data:
            break
        all_modules.extend(data)
        offset += batch_size
        if len(data) < batch_size:
            break
    return all_modules

def fetch_full_definition_for_each_module(module_list, token):
    """Fetch full module definitions and save them locally."""
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    for mod_summary in module_list:
        mid = mod_summary.get("id")
        if not mid:
            continue
        url = f"{REMOTE_SINGLE_MODULE_URL.rstrip('/')}/{mid}"
        try:
            resp = requests.get(url, headers=headers, timeout=15)
            if resp.status_code == 404:
                continue
            resp.raise_for_status()
            mod_data = resp.json()
            save_module_to_folder(mod_data)
        except Exception as e:
            print(f"[ERROR] Could not fetch module {mid}: {e}")

def fetch_all_modules_from_remote(token):
    """Download all modules and save them to disk."""
    module_list = fetch_list_of_modules(token)
    fetch_full_definition_for_each_module(module_list, token)

def extract_raw_components(component, ancestors=None):
    """
    Returns a list of dicts with every primitive attribute of every component,
    plus its 'ancestors' path.
    """
    raws = []
    if ancestors is None:
        ancestors = []
    comp_ancestors = ancestors + [component.get('key', component.get('componentId', ''))]

    # Capture all fields we care about (flattening nested dicts into JSON strings if needed)
    raw = {
        "component_id": component.get("componentId", ""),
        "key":          component.get("key", ""),
        "ancestors":    " > ".join(str(a) for a in comp_ancestors if a),
        "type":         component.get("type", ""),
        "label":        component.get("label", ""),
        "inputType":    component.get("inputType", None),
        "elementType":  component.get("elementType", None),
        "path":         component.get("path", None),
        "tags":         json.dumps(component.get("tags", [])),
        "customClass":  component.get("customClass", None),
        # add any other top-level keys you need
    }
    raws.append(raw)

    # recurse into children
    for k in ('components', 'columns', 'rows', 'children'):
        subs = component.get(k)
        if isinstance(subs, list):
            for sub in subs:
                if isinstance(sub, dict):
                    raws += extract_raw_components(sub, comp_ancestors)
    return raws

def extract_integrations(component, ancestors=None):
    ints = []
    if ancestors is None:
        ancestors = []
    comp_ancestors = ancestors + [component.get('key', component.get('componentId', ''))]

    if component.get("type") == "integration":
        ints.append({
            "component_id": component.get("componentId", ""),
            "key":          component.get("key", ""),
            "ancestors":    " > ".join(str(a) for a in comp_ancestors if a),
            "config":       json.dumps(component.get("integrationConfig", {}))
        })

    # recurse
    for k in ('components', 'columns', 'rows', 'children'):
        subs = component.get(k)
        if isinstance(subs, list):
            for sub in subs:
                if isinstance(sub, dict):
                    ints += extract_integrations(sub, comp_ancestors)
    return ints

def extract_plugins(component, ancestors=None):
    plugs = []
    if ancestors is None:
        ancestors = []
    comp_ancestors = ancestors + [component.get('key', component.get('componentId', ''))]

    if component.get("type") == "plugin":
        plugs.append({
            "component_id": component.get("componentId", ""),
            "key":          component.get("key", ""),
            "ancestors":    " > ".join(str(a) for a in comp_ancestors if a),
            "settings":     json.dumps(component.get("pluginSettings", {}))
        })

    # recurse
    for k in ('components', 'columns', 'rows', 'children'):
        subs = component.get(k)
        if isinstance(subs, list):
            for sub in subs:
                if isinstance(sub, dict):
                    plugs += extract_plugins(sub, comp_ancestors)
    return plugs

def serialize(obj):
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)

def find_module_references_deep(obj):
    """Recursively find module references in nested dict/list."""
    refs = set()
    if isinstance(obj, dict):
        for key in ('integratorUrl', 'url'):
            val = obj.get(key)
            if isinstance(val, str):
                m = re.search(r'/modules/([A-Za-z0-9]+)/execute', val)
                if m:
                    refs.add(m.group(1))
        type_val = obj.get('type')
        t = type_val.lower() if isinstance(type_val, str) else ''
        comp_name = obj.get('componentName')
        cname = comp_name.lower() if isinstance(comp_name, str) else ''
        if t == 'modulereference' or cname == 'modulereference':
            mid = obj.get('moduleId') or obj.get('referenceModuleId')
            if mid:
                refs.add(mid)
        for v in obj.values():
            if isinstance(v, (dict, list)):
                refs |= find_module_references_deep(v)
    elif isinstance(obj, list):
        for item in obj:
            if isinstance(item, (dict, list)):
                refs |= find_module_references_deep(item)
    return refs

def extract_fields(components, ancestors=None):
    """Recursively extract input fields from components."""
    fields = []
    if ancestors is None:
        ancestors = []
    for comp in components:
        if not isinstance(comp, dict):
            continue
        comp_ancestors = ancestors + [comp.get('key', comp.get('componentId', ''))]
        is_input = comp.get('input', False)
        is_hidden = comp.get('type') == 'hidden'
        is_number = comp.get('type') == 'number'
        is_basic_field = comp.get('type') in ('textfield', 'number', 'radio', 'dateinput', 'basicDropdown', 'email', 'phoneNumber', 'checkboxv2')
        if is_input or is_hidden or is_number or is_basic_field:
            fields.append({
                'component_id': comp.get('componentId', ''),
                'input_key': comp.get('key', ''),
                'input_type': comp.get('type', ''),
                'label': comp.get('label', ''),
                'description': comp.get('description', ''),
                'placeholder': comp.get('placeholder', ''),
                'default_value': serialize(comp.get('defaultValue')),
                'data_src': comp.get('dataSrc', ''),
                'data_ref_key': (comp.get('data', {}) or {}).get('dataReferenceKey', ''),
                'values': serialize(comp.get('values') or (comp.get('data', {}) or {}).get('values')),
                'validate': serialize(comp.get('validate', {})),
                'linked_inputs': serialize(comp.get('linked', {}).get('inputs', [])),
                'linked_outputs': serialize(comp.get('linked', {}).get('outputs', [])),
                'integrator_url': (comp.get('integratorData', {}) or {}).get('integratorUrl') or comp.get('url'),
                'service_type': comp.get('serviceType', ''),
                'persistent': comp.get('persistent', False),
                'was_dropped': comp.get('wasDropped', False),
                'ancestors': ' > '.join([str(a) for a in comp_ancestors if a]),
            })
        for k in ('components', 'columns', 'rows', 'children'):
            subs = comp.get(k)
            if isinstance(subs, list):
                if k in ('columns', 'rows'):
                    for col in subs:
                        if isinstance(col, dict) and 'components' in col:
                            fields.extend(extract_fields(col['components'], comp_ancestors + [col.get('key', '')]))
                        else:
                            fields.extend(extract_fields([col], comp_ancestors))
                else:
                    fields.extend(extract_fields(subs, comp_ancestors))
    return fields

# def extract_unqork_module_inputs(conn, folder_path: str, table_name: str, persist: bool = False) -> pd.DataFrame:
#     """Extract input fields from all module JSONs in a folder and optionally persist to SQLite."""
#     processed = set()
#     to_process = []
#     all_records = []

#     # Build lookup of module_id -> path
#     lookup = {}
#     for fn in os.listdir(folder_path):
#         if fn.lower().endswith('.json'):
#             p = os.path.join(folder_path, fn)
#             try:
#                 m = json.load(open(p, encoding='utf-8'))
#                 lookup[m.get('_id')] = p
#             except Exception:
#                 pass

#     to_process = list(lookup.keys())

#     while to_process:
#         mid = to_process.pop(0)
#         if mid in processed:
#             continue
#         processed.add(mid)
#         path = lookup.get(mid)
#         if not path:
#             continue
#         try:
#             data = json.load(open(path, encoding='utf-8'))
#         except Exception:
#             continue
#         title = data.get('title', '')
#         mod_at = data.get('modified', '')
#         mod_by = data.get('modifier', '')
#         comps = data.get('components') or data.get('children') or []
#         for fld in extract_fields(comps):
#             rec = {
#                 'module_id': mid,
#                 'module_title': title,
#                 'modified_at': mod_at,
#                 'modified_by': mod_by
#             }
#             rec.update(fld)
#             all_records.append(rec)
#         for ref in find_module_references_deep(data):
#             if ref not in processed and ref not in to_process:
#                 to_process.append(ref)

#     df = pd.DataFrame(all_records)
#     if persist and not df.empty:
#         import db_utils
#         snap = db_utils.persist_version(
#             conn, df, table_name, 0, df.columns.tolist(), {c: c for c in df.columns}
#         )
#         print(f"Saved extracted inputs to versioned table: {snap}")
#     return df

def extract_validations(component, ancestors=None):
    validations = []
    if ancestors is None:
        ancestors = []
    comp_ancestors = ancestors + [component.get('key', component.get('componentId', ''))]
    # 1. Field-level validate
    if 'validate' in component and component.get('validate'):
        validations.append({
            "component_id": component.get("componentId", ""),
            "key": component.get("key", ""),
            "ancestors": " > ".join([str(a) for a in comp_ancestors if a]),
            "validation_type": "field_validate",
            "validation": serialize(component['validate'])
        })
    # 2. Decision logic
    if component.get('type') == "decision" and "decisionData" in component:
        validations.append({
            "component_id": component.get("componentId", ""),
            "key": component.get("key", ""),
            "ancestors": " > ".join([str(a) for a in comp_ancestors if a]),
            "validation_type": "decision",
            "validation": serialize(component['decisionData'])
        })
    # 3. Dataworkflow logic
    if component.get('type') == "dataworkflow" and "dataworkflowData" in component:
        validations.append({
            "component_id": component.get("componentId", ""),
            "key": component.get("key", ""),
            "ancestors": " > ".join([str(a) for a in comp_ancestors if a]),
            "validation_type": "dataworkflow",
            "validation": serialize(component['dataworkflowData'])
        })
    # Recurse
    for k in ('components', 'columns', 'rows', 'children'):
        subs = component.get(k)
        if isinstance(subs, list):
            if k in ('columns', 'rows'):
                for col in subs:
                    if isinstance(col, dict) and 'components' in col:
                        validations += extract_validations(col, comp_ancestors + [col.get('key', '')])
                    else:
                        validations += extract_validations(col, comp_ancestors)
            else:
                for sub in subs:
                    validations += extract_validations(sub, comp_ancestors)
    return validations

# --------- CSS/HTML/JS EXTRACTION ---------

def extract_custom_css_html(component, ancestors=None):
    css_html = []
    if ancestors is None:
        ancestors = []
    comp_ancestors = ancestors + [component.get('key', component.get('componentId', ''))]
    if component.get("type") in ["content", "htmlelement"]:
        for k in ["html", "content"]:
            if k in component and component[k]:
                css_html.append({
                    "component_id": component.get("componentId", ""),
                    "key": component.get("key", ""),
                    "ancestors": " > ".join([str(a) for a in comp_ancestors if a]),
                    "content_type": component["type"],
                    "snippet_type": k,
                    "snippet": component[k]
                })
    # Recurse
    for k in ('components', 'columns', 'rows', 'children'):
        subs = component.get(k)
        if isinstance(subs, list):
            if k in ('columns', 'rows'):
                for col in subs:
                    if isinstance(col, dict) and 'components' in col:
                        css_html += extract_custom_css_html(col, comp_ancestors + [col.get('key', '')])
                    else:
                        css_html += extract_custom_css_html(col, comp_ancestors)
            else:
                for sub in subs:
                    css_html += extract_custom_css_html(sub, comp_ancestors)
    return css_html

# --------- DATAWORKFLOW LOGIC EXTRACTION ---------

def extract_dataworkflows(component, ancestors=None):
    dwfs = []
    if ancestors is None:
        ancestors = []
    comp_ancestors = ancestors + [component.get('key', component.get('componentId', ''))]
    if component.get("type") == "dataworkflow" and "dataworkflowData" in component:
        dwfs.append({
            "component_id": component.get("componentId", ""),
            "key": component.get("key", ""),
            "ancestors": " > ".join([str(a) for a in comp_ancestors if a]),
            "dataworkflowData": serialize(component["dataworkflowData"])
        })
    # Recurse
    for k in ('components', 'columns', 'rows', 'children'):
        subs = component.get(k)
        if isinstance(subs, list):
            if k in ('columns', 'rows'):
                for col in subs:
                    if isinstance(col, dict) and 'components' in col:
                        dwfs += extract_dataworkflows(col, comp_ancestors + [col.get('key', '')])
                    else:
                        dwfs += extract_dataworkflows(col, comp_ancestors)
            else:
                for sub in subs:
                    dwfs += extract_dataworkflows(sub, comp_ancestors)
    return dwfs

# --------- HELPER TO RECURSE AND FLATTEN ALL COMPONENTS (optional use) ---------

def find_all_components(components):
    all_comps = []
    def _recurse(items):
        for comp in items:
            if not isinstance(comp, dict): continue
            all_comps.append(comp)
            for k in ('components', 'columns', 'rows', 'children'):
                subs = comp.get(k)
                if isinstance(subs, list):
                    if k in ('columns', 'rows'):
                        for col in subs:
                            if isinstance(col, dict) and 'components' in col:
                                _recurse(col['components'])
                            else:
                                _recurse([col])
                    else:
                        _recurse(subs)
    _recurse(components)
    return all_comps

# --------- MAIN DRIVER FUNCTION ---------
def extract_unqork_module_inputs(conn, folder_path: str, table_name: str, persist: bool = False) -> pd.DataFrame:
    """
    Extract integrated config (fields, validations, css/html, dataworkflow) from all module JSONs in a folder,
    and optionally persist to SQLite, using the merged extraction approach.
    """
    json_files = glob.glob(os.path.join(folder_path, "*.json"))
    merged_dfs = []
    for path in json_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                module_json = json.load(f)
            df_merged = extract_all_module_details_merged(module_json)
            if not df_merged.empty:
                # Add module-level info to every row
                module_id = module_json.get('_id', '')
                module_title = module_json.get('title', '')
                modified_at = module_json.get('modified', '')
                modified_by = module_json.get('modifier', '')
                df_merged['module_id'] = module_id
                df_merged['module_title'] = module_title
                df_merged['modified_at'] = modified_at
                df_merged['modified_by'] = modified_by
                merged_dfs.append(df_merged)
        except Exception as e:
            print(f"[ERROR] Could not extract from {path}: {e}")

    if merged_dfs:
        df = pd.concat(merged_dfs, ignore_index=True)
    else:
        df = pd.DataFrame()

    if persist and not df.empty:
        import db_utils
        snap = db_utils.persist_version(
            conn, df, table_name, 0, df.columns.tolist(), {c: c for c in df.columns}
        )
        print(f"Saved extracted inputs to versioned table: {snap}")
    return df

def extract_all_module_details(module_json):
    components = module_json.get("components", [])
    fields = extract_fields(components)
    validations = extract_validations({"components": components})
    css_html = extract_custom_css_html({"components": components})
    dataworkflows = extract_dataworkflows({"components": components})

    df_fields = pd.DataFrame(fields)
    df_validations = pd.DataFrame(validations)
    df_css_html = pd.DataFrame(css_html)
    df_dataworkflows = pd.DataFrame(dataworkflows)

    # Return as a dict of DataFrames
    return {
        "fields": df_fields,
        "validations": df_validations,
        "css_html": df_css_html,
        "dataworkflows": df_dataworkflows,
    }

def extract_all_module_details_merged(module_json):
    """
    Extracts and merges all possible details from a Unqork module JSON.
    Includes raw attributes, fields, validations, CSS/HTML snippets,
    data-workflows, integrations, and plugins in one DataFrame.
    """
    components = module_json.get("components", [])

    # Step 0: Raw components, integrations & plugins
    raws          = extract_raw_components({'components': components}) if 'extract_raw_components' in globals() else []
    integrations  = extract_integrations({'components': components}) if 'extract_integrations' in globals() else []
    plugins       = extract_plugins({'components': components}) if 'extract_plugins' in globals() else []

    # Step 1: Standard extractors
    fields        = extract_fields(components)
    validations   = extract_validations({"components": components})
    css_html      = extract_custom_css_html({"components": components})
    dataflows     = extract_dataworkflows({"components": components})

    # Convert to DataFrames
    df_fields        = pd.DataFrame(fields)
    df_validations   = pd.DataFrame(validations)
    df_css_html      = pd.DataFrame(css_html)
    df_dataflows     = pd.DataFrame(dataflows)

    # Step 2: Ensure join keys exist in every DF
    join_cols = ["component_id", "key", "ancestors"]
    for df in (df_fields, df_validations, df_css_html, df_dataflows):
        for col in join_cols:
            if col not in df.columns:
                df[col] = None

    # Step 3: Build a base DataFrame of every component seen
    non_empty = [d for d in (df_fields, df_validations, df_css_html, df_dataflows) if not d.empty]
    if non_empty:
        base_keys = pd.concat([d[join_cols] for d in non_empty], ignore_index=True).drop_duplicates()
    else:
        base_keys = pd.DataFrame(columns=join_cols)
    df = base_keys.copy()

    # Re-add label (and other field-specific cols) from df_fields
    if not df_fields.empty:
        label_df = df_fields[join_cols + ["label"]].drop_duplicates()
        df = pd.merge(df, label_df, how="left", on=join_cols)
    else:
        df["label"] = None

    # Merge: validations
    if not df_validations.empty:
        v_agg = df_validations.groupby(join_cols).agg({
            "validation_type": lambda x: list(x),
            "validation":      lambda x: list(x)
        }).reset_index()
        v_agg.columns = join_cols + ["validation_types", "validations"]
        df = pd.merge(df, v_agg, how="left", on=join_cols)
    else:
        df["validation_types"] = None
        df["validations"]      = None

    # Merge: CSS/HTML snippets
    if not df_css_html.empty:
        c_agg = df_css_html.groupby(join_cols).agg({
            "content_type": lambda x: list(x),
            "snippet_type": lambda x: list(x),
            "snippet":      lambda x: list(x)
        }).reset_index()
        c_agg.columns = join_cols + ["content_types", "snippet_types", "snippets"]
        df = pd.merge(df, c_agg, how="left", on=join_cols)
    else:
        df["content_types"] = None
        df["snippet_types"] = None
        df["snippets"]      = None

    # Merge: data-workflows
    if not df_dataflows.empty:
        d_agg = df_dataflows.groupby(join_cols).agg({
            "dataworkflowData": lambda x: list(x)
        }).reset_index()
        d_agg.columns = join_cols + ["dataworkflowData"]
        df = pd.merge(df, d_agg, how="left", on=join_cols)
    else:
        df["dataworkflowData"] = None

    # Step 4: Merge in raw component attributes
    if raws:
        df_raw = pd.DataFrame(raws).drop_duplicates(subset=join_cols)
        df = pd.merge(df, df_raw, how="left", on=join_cols)
    # Step 5: Merge in integrations
    if integrations:
        df_int = pd.DataFrame(integrations).drop_duplicates(subset=join_cols)
        df = pd.merge(df, df_int, how="left", on=join_cols)
    else:
        df["config"] = None
    # Step 6: Merge in plugins
    if plugins:
        df_plug = pd.DataFrame(plugins).drop_duplicates(subset=join_cols)
        df = pd.merge(df, df_plug, how="left", on=join_cols)
    else:
        df["settings"] = None

    # Optional: add module-level metadata
    df["module_id"]    = module_json.get("_id", None)
    df["module_title"] = module_json.get("title", None)
    df["modified_at"] = module_json.get("modified", None)
    df["modified_by"] = module_json.get("modifier", None)
    return df
