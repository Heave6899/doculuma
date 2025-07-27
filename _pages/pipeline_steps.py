import streamlit as st
import pandas as pd
import uuid
import db_utils
import config
from llm_manager import LLMManager
import json
import re
import sqlite3
from file_processors import embed_and_save_to_chroma, ensure_stable_id
import streamlit.components.v1 as components
from pyvis.network import Network
import networkx as nx
import tempfile

# --- Globals & Initializers ---
chroma_client = db_utils.get_chroma_client()
embedding_fn = db_utils.get_embedding_function()
conn = db_utils.get_connection(str(config.DB_PATH))


import hashlib
# --- Pipeline Persistence ---
def save_pipeline_to_db(name, definition):
    if not name:
        st.error("Pipeline name cannot be empty.")
        return
    try:
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO saved_pipelines (name, definition) VALUES (?, ?)",
                (name, json.dumps(definition))
            )
        st.success(f"Pipeline '{name}' saved successfully!")
        st.session_state["pipeline_outputs"] = {}
        load_pipelines_from_db.clear()  # Clears all Streamlit cached data
        st.toast("Pipeline outputs cache cleared.")
    except Exception as e:
        st.error(f"Failed to save pipeline: {e}")

def load_pipelines_from_db(_conn):
    try:
        df = pd.read_sql("SELECT name, definition FROM saved_pipelines ORDER BY name", _conn)
        return {row['name']: json.loads(row['definition']) for _, row in df.iterrows()}
    except Exception:
        return {}

# --- Step Execution Logic ---
def _execute_input_step(step_config, db_conn):
    table_name = step_config.get("table_name")
    if not table_name:
        raise ValueError("Input step requires a table name.")
    return pd.read_sql_query(f'SELECT * FROM "{table_name}"', db_conn)

def _execute_sql_step(step_config, db_conn, input_df=None):
    query = step_config.get("query")
    if not query:
        raise ValueError("SQL step requires a query.")
    if input_df is not None:
        with sqlite3.connect(":memory:") as temp_conn:
            input_df.to_sql("input_df", temp_conn, index=False, if_exists="replace")
            return pd.read_sql_query(query, temp_conn)
    else:
        return pd.read_sql_query(query, db_conn)

def _execute_semantic_join_step(step_config, available_outputs):
    loop_input_id, search_input_id, query_cols, search_table_name, top_k = (
        step_config.get("loop_input_id"),
        step_config.get("search_input_id"),
        step_config.get("query_cols"),
        step_config.get("search_table_name"),
        step_config.get("top_k", 5),
    )

    if not all([loop_input_id, search_input_id, query_cols, search_table_name]):
        raise ValueError("Semantic Join step is missing required configuration.")

    loop_df = available_outputs.get(loop_input_id)
    search_df = available_outputs.get(search_input_id)

    if loop_df is None:
        raise ValueError(f"Loop input DataFrame '{loop_input_id}' not found.")

    if search_df is None:
        raise ValueError(f"Search input DataFrame '{search_input_id}' not found.")

    if "id" not in search_df.columns:
        raise ValueError("The 'Search Input' DataFrame must have a 'id' column.")

    collection_name = f"{config.CHROMA_COLLECTION_PREFIX}_{search_table_name}"

    try:
        collection = chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(f"Could not find Chroma collection '{collection_name}'. Error: {e}")

    all_results = []

    for _, row in loop_df.iterrows():
        query_text_parts = [str(row[col]) for col in query_cols if isinstance(row[col], str) and row[col].strip()]
        query_text = " ".join(query_text_parts)
        # query_text = row.get(query_col)
        if not isinstance(query_text, str) or not query_text.strip():
            continue  # skip empty or non-string queries

        try:
            results = collection.query(query_texts=[query_text], n_results=top_k)
        except Exception as e:
            st.warning(f"Semantic query failed for '{query_text}': {e}")
            continue

        if not results or not results.get("metadatas") or not results["metadatas"][0]:
            continue  # no matches

        matched_stable_ids = [meta["stable_id"] for meta in results["metadatas"][0] if "stable_id" in meta]
        matched_rows_df = search_df[search_df["id"].isin(matched_stable_ids)].copy()

        for _, matched_row in matched_rows_df.iterrows():
            combined_row = {f"loop_{c}": v for c, v in row.items()}
            combined_row.update({f"match_{c}": v for c, v in matched_row.items()})
            all_results.append(combined_row)

    if not all_results:
        st.info("No semantic matches were found.")
        return pd.DataFrame()  # return empty DataFrame if no matches found

    return pd.DataFrame(all_results)

def _execute_llm_enrich_step(step_config, input_df):
    if input_df is None:
        raise ValueError("LLM Enrich step requires an input DataFrame.")
    prompt_template = step_config.get("prompt")
    new_col_name = step_config.get("new_col_name", "llm_response")
    if not prompt_template:
        raise ValueError("LLM Enrich step requires a prompt template.")
    llm_mgr = LLMManager()
    responses = [llm_mgr.generate(prompt_template.format(**row.to_dict())) for _, row in input_df.iterrows()]
    st.info(responses)
    input_df[new_col_name] = responses
    return input_df

def visualize_pipeline(steps):
    G = nx.DiGraph()

    execution_order = []

    def flatten_steps(step_list):
        for step in step_list:
            execution_order.append(step)
            if step.get("children"):
                flatten_steps(step["children"])

    flatten_steps(steps)

    # Add nodes first
    for step in execution_order:
        step_id = step['id']
        step_label = f"{step['config'].get('step_name', step['type'])}\n[{step['type']}]"
        G.add_node(step_id, label=step_label, title=step_label)

    # Connect sequential steps
    for i in range(len(execution_order)-1):
        from_step = execution_order[i]['id']
        to_step = execution_order[i+1]['id']
        G.add_edge(from_step, to_step, color='gray', label='next')

    # Connect explicit dependencies clearly
    for step in execution_order:
        cfg = step.get('config', {})
        deps = [cfg.get(k) for k in ['input_id', 'left_id', 'right_id', 'loop_input_id', 'search_input_id'] if cfg.get(k)]
        for dep in deps:
            if dep and dep != step['id']:  # Avoid self-loop
                G.add_edge(dep, step['id'], color='blue', label='depends')

    net = Network(height="600px", width="100%", directed=True, notebook=False)
    net.from_nx(G)

    # Styling for better distinction
    for edge in net.edges:
        edge['title'] = edge.get('label', '')
        if edge['label'] == 'depends':
            edge['color'] = 'blue'
            edge['width'] = 2
        else:  # sequential edges
            edge['color'] = 'gray'
            edge['width'] = 1
            edge['dashes'] = True

    net.repulsion(node_distance=250, central_gravity=0.3)

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix='.html') as tmp_file:
        net.save_graph(tmp_file.name)
        html_content = open(tmp_file.name, 'r', encoding='utf-8').read()

    return html_content


def _execute_merge_step(step_config, available_outputs):
    left_id, right_id = step_config.get("left_id"), step_config.get("right_id")
    if not left_id or not right_id:
        raise ValueError("Merge step requires both a left and a right input step.")
    left_df, right_df = available_outputs.get(left_id), available_outputs.get(right_id)
    if left_df is None or right_df is None:
        raise ValueError("Input DataFrames for merge not found.")
    left_on = [k.strip() for k in step_config.get("left_on", "").split(",")]
    right_on = [k.strip() for k in step_config.get("right_on", "").split(",")]
    return pd.merge(left_df, right_df, left_on=left_on, right_on=right_on, how=step_config.get("how", "inner"))

def _execute_save_step(step_config, db_conn, input_df):
    if input_df is None:
        raise ValueError("Save & Index step requires an input DataFrame.")
    table_name = step_config.get("table_name")
    if not table_name or not re.match(config.TABLE_NAME_REGEX, table_name):
        raise ValueError("A valid table name is required for saving.")
    df_to_save = ensure_stable_id(input_df.copy())
    db_utils.persist_version(db_conn, df_to_save, table_name, 0, df_to_save.columns.tolist(), {c:c for c in df_to_save.columns})
    st.toast(f"✅ Saved result to `{table_name}`.")
    if embed_cols := step_config.get("embed_cols"):
        with st.spinner(f"Generating embeddings for `{table_name}`..."):
            embed_and_save_to_chroma(chroma_client, df_to_save, table_name, embed_cols, embedding_fn)
        st.toast(f"✨ Indexed `{table_name}` for semantic search.")
    return df_to_save


# --- Pipeline Runner ---
def execute_pipeline():
    steps = st.session_state.get("pipeline_steps", [])
    if not steps:
        st.warning("No steps in the pipeline to run.")
        return
    def get_all_steps(step_list):
        all_s = []
        for s in step_list:
            all_s.append(s)
            if s.get("children"):
                all_s.extend(get_all_steps(s["children"]))
        return all_s
    all_steps = get_all_steps(steps)
    execution_order, executed_ids = [], set()
    while len(execution_order) < len(all_steps):
        made_progress = False
        for step in all_steps:
            if step['id'] in executed_ids:
                continue
            cfg = step.get('config', {})
            deps = {cfg.get(k) for k in ['input_id', 'left_id', 'right_id', 'loop_input_id', 'search_input_id'] if cfg.get(k)}
            if all(dep_id in executed_ids for dep_id in deps):
                execution_order.append(step)
                executed_ids.add(step['id'])
                made_progress = True
        if not made_progress:
            st.error("🔴 Pipeline Error: A circular dependency was detected or a step's input is invalid. Please check step inputs.")
            return
    outputs = {}
    st.session_state['pipeline_outputs'] = outputs
    for step in execution_order:
        step_id, step_type, cfg = step['id'], step['type'], step.get('config', {})
        try:
            with st.spinner(f"Running Step: '{cfg.get('step_name', step_id)}' [{step_type}]..."):
                input_df = outputs.get(cfg.get("input_id"))
                step_map = {
                    "Input": (_execute_input_step, [cfg, conn]),
                    "SQL": (_execute_sql_step, [cfg, conn, input_df]),
                    "Semantic Join (Loop)": (_execute_semantic_join_step, [cfg, outputs]),
                    "LLM Enrich": (_execute_llm_enrich_step, [cfg, input_df]),
                    "Merge": (_execute_merge_step, [cfg, outputs]),
                    "Save & Index": (_execute_save_step, [cfg, conn, input_df]),
                }
                if step_type in step_map:
                    func, args = step_map[step_type]
                    outputs[step_id] = func(*args)
        except Exception as e:
            st.error(f"Error in step '{cfg.get('step_name', step_id)}' [{step_type}]: {e}")
            return
    st.success("✅ Pipeline executed successfully!")


# --- UI Rendering ---
def init_pipeline():
    if "pipeline_steps" not in st.session_state:
        st.session_state["pipeline_steps"] = []
    if "pipeline_outputs" not in st.session_state:
        st.session_state["pipeline_outputs"] = {}
    if "pipeline_name" not in st.session_state:
        st.session_state["pipeline_name"] = ""

def add_pipeline_step(parent_steps):
    parent_steps.append({"id": str(uuid.uuid4()), "type": "Input", "config": {"step_name": ""}, "children": []})

def find_step_and_parent(target_id, steps, parent=None):
    for step in steps:
        if step['id'] == target_id:
            return step, parent
        if 'children' in step:
            found, p = find_step_and_parent(target_id, step['children'], step)
            if found:
                return found, p
    return None, None

def delete_step_by_id(target_id):
    step, parent = find_step_and_parent(target_id, st.session_state.pipeline_steps)
    if step:
        container = parent['children'] if parent else st.session_state.pipeline_steps
        container.remove(step)
        st.session_state.pipeline_outputs.pop(target_id, None)

def render_step(step, all_steps_flat, step_counter, level=0):
    step.setdefault("config", {})
    cfg = step['config']

    default_name = f"Step {step_counter[0]}"
    step_counter[0] += 1  # Increment step number

    cfg['step_name'] = st.text_input(
        "Step Name", 
        value=cfg.get('step_name', default_name), 
        key=f"name_{step['id']}"
    )

    expander_label = f"🧩 {cfg.get('step_name') or default_name}: [{step['type']}]"

    with st.expander(expander_label, expanded=True):
        step_types = ["Input", "SQL", "Semantic Join (Loop)", "LLM Enrich", "Merge", "Save & Index"]
        cfg['type'] = st.selectbox(
            "Step type", 
            step_types, 
            key=f"type_{step['id']}", 
            index=step_types.index(step.get('type', 'Input'))
        )
        step['type'] = cfg['type']

        available_inputs = {
            s['id']: s.get('config', {}).get('step_name', f"Step {i+1}") 
            for i, s in enumerate(all_steps_flat) if s['id'] != step['id']
        }
        def format_func(x): return "None" if x is None else available_inputs.get(x, "Invalid Step")

        if step['type'] == 'Input':
            tables_list = db_utils.list_tables(conn, 0)
            cfg['table_name'] = st.selectbox(
                "Database Table:", tables_list,
                key=f"table_{step['id']}",
                index=tables_list.index(cfg.get('table_name')) if cfg.get('table_name') in tables_list else 0
            )

        elif step['type'] == 'SQL':
            input_keys = [None] + list(available_inputs.keys())
            cfg['input_id'] = st.selectbox(
                "Input from Step:", input_keys, 
                format_func=format_func, key=f"input_{step['id']}",
                index=input_keys.index(cfg.get('input_id')) if cfg.get('input_id') in input_keys else 0
            )
            if cfg.get('input_id'):
                st.info("ℹ️ **Hint:** Use `input_df` as table name for querying previous step's result.")
            else:
                st.info("ℹ️ **Hint:** Query main DB tables directly.")
            cfg['query'] = st.text_area(
                "SQL Query:", cfg.get('query', ''),
                key=f"query_{step['id']}"
            )

        elif step['type'] == 'Semantic Join (Loop)':
            cfg['loop_input_id'] = st.selectbox(
                "Loop Input (rows to enrich):", list(available_inputs.keys()),
                format_func=format_func, key=f"loop_input_{step['id']}",
                index=list(available_inputs.keys()).index(cfg.get('loop_input_id')) if cfg.get('loop_input_id') in available_inputs else 0
            )
            cfg['search_input_id'] = st.selectbox(
                "Search Input (reference data):", list(available_inputs.keys()),
                format_func=format_func, key=f"search_input_{step['id']}",
                index=list(available_inputs.keys()).index(cfg.get('search_input_id')) if cfg.get('search_input_id') in available_inputs else 0
            )
            loop_df = st.session_state['pipeline_outputs'].get(cfg.get('loop_input_id'))
            if loop_df is not None:
                cfg['query_cols'] = st.multiselect(
                    "Columns from Loop Input to Query With:",
                    loop_df.columns.tolist(),
                    default=cfg.get('query_cols', []),
                    key=f"query_cols_{step['id']}"
                )
            else:
                st.warning("Run the pipeline first to populate loop input columns.")
            cfg['search_table_name'] = st.text_input(
                "Original Search Table Name (Chroma):", cfg.get('search_table_name', ''),
                key=f"search_table_{step['id']}"
            )
            cfg['top_k'] = st.number_input(
                "Top K Matches per Row:", 1, 20, cfg.get('top_k', 5),
                key=f"top_k_{step['id']}"
            )

        elif step['type'] == 'LLM Enrich':
            cfg['input_id'] = st.selectbox(
                "Input Step:", list(available_inputs.keys()),
                format_func=format_func, key=f"llm_input_{step['id']}",
                index=list(available_inputs.keys()).index(cfg.get('input_id')) if cfg.get('input_id') in available_inputs else 0
            )
            cfg['prompt'] = st.text_area(
                "LLM Prompt Template:", cfg.get('prompt', ''),
                key=f"prompt_{step['id']}"
            )
            cfg['new_col_name'] = st.text_input(
                "New Column Name:", cfg.get('new_col_name', 'llm_response'),
                key=f"new_col_{step['id']}"
            )

        elif step['type'] == 'Merge':
            cfg['left_id'] = st.selectbox(
                "Left Input Step:", list(available_inputs.keys()),
                format_func=format_func, key=f"left_input_{step['id']}",
                index=list(available_inputs.keys()).index(cfg.get('left_id')) if cfg.get('left_id') in available_inputs else 0
            )
            cfg['right_id'] = st.selectbox(
                "Right Input Step:", list(available_inputs.keys()),
                format_func=format_func, key=f"right_input_{step['id']}",
                index=list(available_inputs.keys()).index(cfg.get('right_id')) if cfg.get('right_id') in available_inputs else 0
            )
            cfg['left_on'] = st.text_input(
                "Left Join Columns (comma-separated):", cfg.get('left_on', ''),
                key=f"left_on_{step['id']}"
            )
            cfg['right_on'] = st.text_input(
                "Right Join Columns (comma-separated):", cfg.get('right_on', ''),
                key=f"right_on_{step['id']}"
            )
            cfg['how'] = st.selectbox(
                "Join Type:", ["inner", "left", "right", "outer"],
                index=["inner", "left", "right", "outer"].index(cfg.get('how', 'inner')),
                key=f"how_{step['id']}"
            )

        elif step['type'] == 'Save & Index':
            available_inputs = {
                s['id']: s.get('config', {}).get('step_name', f"Step {i+1}") 
                for i, s in enumerate(all_steps_flat) if s['id'] != step['id']
            }
            
            input_keys = [None] + list(available_inputs.keys())
            
            cfg['input_id'] = st.selectbox(
                "Input Step to Save:",
                input_keys,
                format_func=lambda x: available_inputs.get(x, "None") if x else "None",
                key=f"save_input_{step['id']}",
                index=input_keys.index(cfg.get('input_id')) if cfg.get('input_id') in input_keys else 0
            )

            cfg['table_name'] = st.text_input(
                "Table Name for Saving:", cfg.get('table_name', ''),
                key=f"save_table_{step['id']}"
            )

            cfg['embed_cols'] = st.text_input(
                "Embedding Columns (comma-separated):", cfg.get('embed_cols', ''),
                key=f"embed_cols_{step['id']}"
            )

        

        # Display output if exists
        output_df = st.session_state.get("pipeline_outputs", {}).get(step['id'])
        if output_df is not None and isinstance(output_df, pd.DataFrame):
                st.markdown("---")
                st.write("Output:")
                st.dataframe(output_df)

        # Render children steps recursively
        for child_step in step.get("children", []):
            st.markdown("---")
            render_step(child_step, all_steps_flat, step_counter, level + 1)

def clear_pipeline():
    st.session_state["pipeline_steps"] = []
    st.session_state["pipeline_outputs"] = {}
    st.session_state["pipeline_name"] = ""
    st.session_state.toast_message = "Pipeline cleared. Ready to create a new pipeline."

def confirm_clear_pipeline():
    if st.session_state["pipeline_steps"]:
        with st.warning("⚠️ Confirm New Pipeline"):
            st.warning("You have unsaved changes that will be lost. Continue?")
            col1, col2 = st.columns(2)
            if col1.button("✅ Confirm", use_container_width=True):
                clear_pipeline()
                st.rerun()
            if col2.button("❌ Cancel", use_container_width=True):
                st.toast("Operation canceled.", icon="⚠️")
    else:
        clear_pipeline()
        st.rerun()

def render(conn):
    """Main render function for the pipeline UI."""
    st.header("🔗 Dynamic Data Pipeline")
    init_pipeline()

    # --- NEW: Sync state before rendering any UI that depends on it ---
    # This ensures that when a sidebar button is clicked, the UI elements that
    # depend on step names (like the sidebar dropdowns) get the latest values.
    def get_all_steps_flat(step_list):
        all_s = []
        for s in step_list:
            all_s.append(s)
            if s.get("children"):
                all_s.extend(get_all_steps_flat(s["children"]))
        return all_s
    
    all_steps_flat = get_all_steps_flat(st.session_state.pipeline_steps)
    for step in all_steps_flat:
        name_key = f"name_{step['id']}"
        if name_key in st.session_state:
            step['config']['step_name'] = st.session_state[name_key]

    # Now build the options for the sidebar with the synced names
    step_options = {s['id']: s.get('config', {}).get('step_name') or f"Step {all_steps_flat.index(s)+1}: [{s['type']}]" for s in all_steps_flat}

    # --- Sidebar for structure editing ---
    with st.sidebar:
        st.title("Pipeline Controls")
        col1, col2 = st.columns(2)
        col1.button("➕ Add Root Step", on_click=add_pipeline_step, args=(st.session_state.pipeline_steps,))
        col2.button("🆕 New Pipeline", on_click=confirm_clear_pipeline)
        if step_options:
            st.markdown("---")
            st.subheader("Edit Step Structure")
            step_to_edit = st.selectbox("Select a step to modify:", options=list(step_options.keys()), format_func=lambda x: step_options.get(x, "Invalid Step"))
            col1, col2 = st.columns(2)
            if col1.button("Add Child ➕"):
                step, _ = find_step_and_parent(step_to_edit, st.session_state.pipeline_steps)
                if step: add_pipeline_step(step.setdefault("children", []))
            if col2.button("Delete 🗑️", type="secondary"):
                delete_step_by_id(step_to_edit)
                st.rerun()
        st.markdown("---")
        st.subheader("Save/Load Pipeline")
        saved_pipelines = load_pipelines_from_db(conn)
        st.session_state.pipeline_name = st.text_input("Pipeline Name", st.session_state.pipeline_name)
        if st.button("💾 Save Pipeline"):
            save_pipeline_to_db(st.session_state.pipeline_name, st.session_state.pipeline_steps)
        if saved_pipelines:
            pipeline_to_load = st.selectbox("Load a saved pipeline:", options=[""] + list(saved_pipelines.keys()))
            if st.button("📂 Load Pipeline"):
                if pipeline_to_load:
                    st.session_state.pipeline_steps = saved_pipelines[pipeline_to_load]
                    st.session_state.pipeline_name = pipeline_to_load
                    st.session_state.pipeline_outputs = {}
                    st.rerun()

    # --- Main canvas wrapped in a single form ---
    tab1, tab2 = st.tabs(["🛠️ Pipeline Builder", "🗺️ Pipeline Visualization"])
    with tab1:
        with st.form(key="pipeline_form"):
            if not st.session_state["pipeline_steps"]:
                st.info("No pipeline steps. Add a step from the sidebar to begin.")
            else:
                step_counter = [1]
                for step in st.session_state["pipeline_steps"]:
                    render_step(step, all_steps_flat,step_counter)
                    st.markdown("---")
            
            run_button_clicked = st.form_submit_button("🚀 Run Entire Pipeline")

        if run_button_clicked:
            execute_pipeline()
        # Per-step "Run" buttons and output rendering, outside the form
        # for step in st.session_state["pipeline_steps"]:
        #     with st.expander(f"{step['config'].get('step_name')} [{step['type']}]", expanded=True):
        #         if st.button(f"▶️ Run {step['config'].get('step_name') or step['type']}", key=f"run_{step['id']}"):
        #             try:
        #                 outputs = st.session_state.get("pipeline_outputs", {})
        #                 step_map = {
        #                     "Input": (_execute_input_step, [step["config"], conn]),
        #                     "SQL": (_execute_sql_step, [step["config"], conn, outputs.get(step["config"].get("input_id"))]),
        #                     "Semantic Join (Loop)": (_execute_semantic_join_step, [step["config"], outputs]),
        #                     "LLM Enrich": (_execute_llm_enrich_step, [step["config"], outputs.get(step["config"].get("input_id"))]),
        #                     "Merge": (_execute_merge_step, [step["config"], outputs]),
        #                     "Save & Index": (_execute_save_step, [step["config"], conn, outputs.get(step["config"].get("input_id"))]),
        #                 }
        #                 if step['type'] in step_map:
        #                     func, args = step_map[step['type']]
        #                     df = func(*args)
        #                     outputs[step['id']] = df
        #                     st.session_state["pipeline_outputs"] = outputs
        #                     st.success(f"Step {step['config'].get('step_name') or step['type']} ran successfully.")
        #             except Exception as e:
        #                 st.error(f"Error in step: {e}")
        #         outputs = st.session_state.get("pipeline_outputs", {}).get(step['id'])
        #         for step in st.session_state["pipeline_steps"]:
        #             render_per_step_run(step, outputs, conn, str(uuid.uuid1()))
        #         # if output_df is not None:
        #         #     st.write(output_df)


    with tab2:
        st.subheader("🗺 Pipeline Dependency Graph")
        if st.session_state["pipeline_steps"]:
            html_graph = visualize_pipeline(st.session_state["pipeline_steps"])
            components.html(html_graph, height=550, scrolling=True)
        else:
            st.info("Add steps to your pipeline to visualize dependencies.")