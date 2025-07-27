import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import json
import duckdb
import db_utils
import config
import traceback
import llm_manager 
# --- Page Configuration ---
st.set_page_config(
    page_title="Dynamic Data Pipelines",
    page_icon="✨",
    layout="wide"
)

# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    """Create and cache the database connection."""
    return db_utils.get_connection(str(config.DB_PATH))

conn = get_db_connection()

# --- Session State Initialization ---
def initialize_state():
    """Initializes session state variables if they don't exist."""
    if 'pipeline_steps' not in st.session_state:
        st.session_state.pipeline_steps = []
    if 'pipeline_results' not in st.session_state:
        st.session_state.pipeline_results = {}

# --- Helper Functions ---
@st.cache_data
def get_tables_list():
    """Fetches and caches the list of tables from the SQLite database."""
    try:
        return db_utils.list_tables(conn, 0)
    except Exception as e:
        st.error(f"Failed to load tables: {e}")
        return []

@st.cache_data
def get_indexed_collections_list():
    """Fetches and caches the list of indexed collections from ChromaDB."""
    try:
        return db_utils.get_all_indexed_collections()
    except Exception as e:
        st.error(f"Failed to load indexed collections: {e}")
        return []

def get_pipeline_step_ids(current_step_id=None):
    """Returns a list of available step IDs to use as inputs."""
    return [step['id'] for step in st.session_state.pipeline_steps if step['id'] != current_step_id]

@st.cache_data
def draw_pipeline_graph(_steps):
    """Draws a dependency graph of the pipeline."""
    G = nx.DiGraph()
    fig, ax = plt.subplots(figsize=(10, 6))

    if not _steps:
        ax.text(0.5, 0.5, "Pipeline is empty", ha='center', va='center', fontsize=12, color='gray')
        ax.axis('off')
        return fig

    node_colors = {
        "Input": "#a8d8ea", "SQL": "#e6cba3", "LLM Enrich": "#f0b9b9",
        "Merge": "#c4a3e6", "Validate": "#f7d59c", "Deduplicate": "#a3e6d8",
        "Save & Index": "#b4e6a3", "Semantic Loop": "#ffb347"
    }

    for step in _steps:
        G.add_node(step['id'], label=f"{step['id']}\n({step['type']})")
        if step.get('input_step_1'): G.add_edge(step['input_step_1'], step['id'])
        if step.get('input_step_2'): G.add_edge(step['input_step_2'], step['id'])

    try:
        pos = nx.spring_layout(G, k=0.9, iterations=50)
    except nx.NetworkXError:
        pos = {}

    labels = nx.get_node_attributes(G, 'label')
    colors = [node_colors.get(s['type'], "#d3d3d3") for s in _steps if s['id'] in G.nodes()]

    nx.draw(G, pos, labels=labels, with_labels=True, node_size=3500, node_color=colors,
            font_size=9, font_weight="bold", ax=ax, edge_color="#7d7d7d", width=1.5,
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))

    plt.title("Pipeline Flow")
    return fig

# --- Core Execution Logic ---
def execute_steps(steps_to_run):
    """Executes a specific list of steps in topological order."""
    G = nx.DiGraph()
    all_steps = st.session_state.pipeline_steps
    for s in all_steps:
        if s.get('input_step_1'): G.add_edge(s['input_step_1'], s['id'])
        if s.get('input_step_2'): G.add_edge(s['input_step_2'], s['id'])

    subgraph_nodes = set()
    for step_id in steps_to_run:
        try:
            subgraph_nodes.add(step_id)
            subgraph_nodes.update(nx.ancestors(G, step_id))
        except nx.NetworkXError: # Handle case where a node might not be in the graph
            continue


    execution_order = [node for node in list(nx.topological_sort(G)) if node in subgraph_nodes]
    # Add nodes with no dependencies
    for node in subgraph_nodes:
        if node not in execution_order:
            execution_order.insert(0, node)


    progress_bar = st.progress(0, text="Preparing to execute steps...")
    for i, step_id in enumerate(execution_order):
        step = next((s for s in all_steps if s['id'] == step_id), None)
        if not step: continue

        progress_bar.progress((i + 1) / len(execution_order), text=f"Running {step_id}: {step['type']}...")
        run_single_step(step)

    st.success("Execution complete!")
    st.rerun()

def run_single_step(step):
    """Logic to execute one pipeline step."""
    step_id = step['id']
    try:
        if step['type'] == 'Input':
            df = pd.read_sql(f"SELECT * FROM `{step['table_name']}`", conn)
            st.session_state.pipeline_results[step_id] = df

        elif step['type'] == 'SQL':
            con = duckdb.connect(database=':memory:', read_only=False)

            # Register all previous pipeline results as tables in DuckDB
            for prev_step_id, prev_df in st.session_state.pipeline_results.items():
                if isinstance(prev_df, pd.DataFrame):
                    con.register(prev_step_id, prev_df)
            
            # Now, execute the user's query against this connection
            # DuckDB will correctly resolve "Step_1", "Step_2", etc.
            result_df = con.execute(step['sql_query']).fetchdf()
            st.session_state.pipeline_results[step_id] = result_df

        elif step['type'] == 'LLM Enrich':
            df = st.session_state.pipeline_results[step['input_step_1']].copy()
            new_column_name = step['new_column_name']
            prompt_template = step['prompt']
            
            # --- CORRECTED LLM CALL IMPLEMENTATION ---
            llm_instance = llm_manager.LLMManager() # Create an instance of the manager
            results = []
            progress_text = "Enriching rows with LLM... please wait."
            llm_progress_bar = st.progress(0, text=progress_text)
            
            for i, row in df.iterrows():
                prompt = prompt_template.format(**row.to_dict())
                # Correctly call the generate method on the instance
                response = llm_instance.generate(prompt) 
                results.append(response)
                llm_progress_bar.progress((i + 1) / len(df), text=progress_text)

            df[new_column_name] = results
            st.session_state.pipeline_results[step_id] = df
            
        elif step['type'] == 'Merge':
            df1 = st.session_state.pipeline_results[step['input_step_1']]
            df2 = st.session_state.pipeline_results[step['input_step_2']]
            result_df = pd.merge(df1, df2, left_on=step['left_on'], right_on=step['right_on'], how=step['how'])
            st.session_state.pipeline_results[step_id] = result_df

        elif step['type'] == 'Deduplicate':
            df = st.session_state.pipeline_results[step['input_step_1']]
            result_df = df.drop_duplicates(subset=step.get('subset') or None)
            st.session_state.pipeline_results[step_id] = result_df

        elif step['type'] == 'Validate':
            df = st.session_state.pipeline_results[step['input_step_1']]
            col = step['column']
            rule = step['rule']
            details = {}
            if rule == 'not_null':
                is_valid = df[col].notna().all()
                details = {'null_count': int(df[col].isna().sum())}
            elif rule == 'is_unique':
                is_valid = df[col].is_unique
                details = {'duplicate_count': int(df[col].duplicated().sum())}
            st.session_state.pipeline_results[step_id] = {'is_valid': bool(is_valid), 'details': details}

        elif step['type'] == 'Save & Index':
            df = st.session_state.pipeline_results[step['input_step_1']]
            db_utils.persist_version(conn, df, step['table_name'], 0, df.columns.tolist(), {})
            st.session_state.pipeline_results[step_id] = df

        elif step['type'] == 'Semantic Loop':
            loop_df = st.session_state.pipeline_results[step['input_step_1']].copy()
            chroma_client = db_utils.get_chroma_client()
            
            # --- REAL SEMANTIC LOOP LOGIC ---
            all_matches = []
            queries = loop_df[step['query_column']].tolist()
            top_k = step.get("top_k", 5)
            similarity_threshold = step.get("similarity_threshold", 0.7)
            join_type = step.get("join_type", "left")
            # The semantic_search_chroma function needs a single query, not a list.
            # We must loop and call it for each query.
            progress_text = "Running Semantic Loop... please wait."
            loop_progress_bar = st.progress(0, text=progress_text)

            for i, query_text in enumerate(queries):
                # Get the original row's data
                original_row = loop_df.iloc[i].to_dict()
                
                # Perform the search for the current row's query text
                search_results = db_utils.semantic_search_chroma(
                    chroma_client,
                    query_text,
                    step['search_collection_prefix'],
                    top_k=top_k
                )
                found_match_for_row = False

                for table_name, matches in search_results.items():
                    for match in matches:
                        # Apply similarity filter
                        if match.get('similarity', 0) >= similarity_threshold:
                            found_match_for_row = True
                            new_row = original_row.copy()
                            new_row['match_table'] = table_name
                            new_row['match_text'] = match.get('text')
                            new_row['match_similarity'] = match.get('similarity')
                            new_row.update(match.get('metadata', {}))
                            all_matches.append(new_row)
                
                # If it's a left join and no matches were found, add the original row
                if join_type == 'left' and not found_match_for_row:
                    all_matches.append(original_row)
                
                loop_progress_bar.progress((i + 1) / len(queries), text=progress_text)
            
            st.session_state.pipeline_results[step_id] = pd.DataFrame(all_matches)


    except Exception as e:
        st.error(f"Error in step {step_id} ({step['type']}): {e}")
        st.code(traceback.format_exc())
        st.stop()


# --- Main UI Rendering ---
# Replace the main() function in dynamic_pipeline.py with this one

def main():
    """Renders the main Streamlit page."""
    initialize_state()
    st.title("✨ Dynamic Data Pipelines")
    st.markdown("Visually create, configure, and run complex data processing workflows.")

    # --- Sidebar for Pipeline Controls ---
    with st.sidebar:
        st.header("🛠️ Pipeline Controls")

        # --- CORRECTED: Save/Load Functionality ---
        st.subheader("💾 Manage Pipeline")
        
        # Load existing pipelines
        saved_pipelines = db_utils.get_saved_pipeline_names(conn)
        
        # We use a selectbox to either load a pipeline or allow for a new name
        pipeline_name_to_load = st.selectbox("Load Existing or Name New Pipeline", [""] + saved_pipelines)
        
        if pipeline_name_to_load:
            # If an existing pipeline is selected, load it
            if st.button("Load Selected Pipeline", use_container_width=True):
                st.session_state.pipeline_steps = db_utils.load_pipeline_definition(conn, pipeline_name_to_load)
                st.session_state["pipeline_name"] = pipeline_name_to_load
                st.success(f"Pipeline '{pipeline_name_to_load}' loaded!")
                st.rerun()
        else:
            # If no pipeline is selected, allow creating a new one
            new_pipeline_name = st.text_input("New Pipeline Name", key="new_pipeline_name")
            if st.button("Save New Pipeline", use_container_width=True):
                if not new_pipeline_name.strip():
                    st.warning("Please enter a name for the new pipeline.")
                else:
                    db_utils.save_pipeline(conn, new_pipeline_name, st.session_state.pipeline_steps)
                    st.success(f"Pipeline '{new_pipeline_name}' saved!")
                    # Refresh the list of saved pipelines
                    st.cache_data.clear()
                    st.rerun()

        st.markdown("---")
        # --- Add/Manage Steps ---
        st.subheader("Add & Run Steps")
        step_type = st.selectbox("Add New Step", ["Input", "SQL", "LLM Enrich", "Semantic Loop", "Merge", "Deduplicate", "Validate", "Save & Index"])
        if st.button(f"Add {step_type} Step", use_container_width=True):
            step_id = f"Step_{len(st.session_state.pipeline_steps) + 1}"
            st.session_state.pipeline_steps.append({"id": step_id, "type": step_type})
            st.rerun()

        st.markdown("---")
        if st.button("▶️ Run Full Pipeline", type="primary", use_container_width=True, disabled=not st.session_state.pipeline_steps):
            execute_steps([s['id'] for s in st.session_state.pipeline_steps])

    if not st.session_state.pipeline_steps:
        st.info("Get started by adding a step or loading a saved pipeline from the sidebar.")
        return

    st.header("Pipeline Configuration")
    for i, step in enumerate(st.session_state.pipeline_steps):
        with st.expander(f"**{step['id']}: {step['type']}**", expanded=True):
            render_step_ui(step, i)

    st.markdown("---")
    st.header("Execution & Results")
    col_graph, col_results = st.columns([0.6, 0.4])
    with col_graph:
        st.subheader("Pipeline Flow")
        fig = draw_pipeline_graph(st.session_state.pipeline_steps)
        st.pyplot(fig)
        
    with col_results:
        st.subheader("Step Results Preview")
        if not st.session_state.pipeline_results:
            st.info("Run the pipeline or a step to see results.")
        else:
            step_ids_with_results = list(st.session_state.pipeline_results.keys())
            selected_step_id = st.selectbox("Select step to preview", step_ids_with_results)
            if selected_step_id:
                result = st.session_state.pipeline_results[selected_step_id]
                st.write(f"**{selected_step_id} Output**")
                if isinstance(result, pd.DataFrame):
                    st.write(f"`{result.shape[0]}` rows, `{result.shape[1]}` columns")
                    st.dataframe(result.head(), use_container_width=True)
                else:
                    st.json(result)
                    
def render_step_ui(step, index):
    """Renders the UI controls for a single pipeline step."""
    step_id = step['id']
    input_ids = get_pipeline_step_ids(step_id)

    def get_cols(step_id_key):
        """Helper to get columns from a selected input step's dataframe."""
        input_step_id = step.get(step_id_key)
        
        # Priority 1: Use the live result if it exists
        if input_step_id in st.session_state.pipeline_results:
            input_df = st.session_state.pipeline_results[input_step_id]
            if isinstance(input_df, pd.DataFrame):
                return input_df.columns.tolist()

        # Priority 2: If no live result, trace back to the source
        if input_step_id:
             parent_step = next((s for s in st.session_state.pipeline_steps if s['id'] == input_step_id), None)
             # If the parent is a simple "Input" step, we can get its schema directly
             if parent_step and parent_step['type'] == 'Input' and parent_step.get('table_name'):
                 try:
                    # Read schema directly from the database for Input steps
                    return pd.read_sql(f"SELECT * FROM `{parent_step['table_name']}` LIMIT 1", conn).columns.tolist()
                 except Exception:
                    return [] # Return empty if the table doesn't exist or there's an error
        
        # Fallback if schema can't be determined
        return []

    # --- Step-specific UI Configurations ---
    if step['type'] == 'Input':
        step['table_name'] = st.selectbox("Select Table", get_tables_list(), key=f"table_{index}")

    elif step['type'] == 'SQL':
        step['input_step_1'] = st.selectbox("Input Step", input_ids, key=f"sql_input_{index}")
        st.info("Use `df` to refer to the input DataFrame (e.g., `SELECT * FROM df`).")
        step['sql_query'] = st.text_area("SQL Query (DuckDB)", step.get('sql_query', "SELECT * FROM df"), key=f"sql_query_{index}")

    elif step['type'] == 'LLM Enrich':
        step['input_step_1'] = st.selectbox("Input Step", input_ids, key=f"llm_input_{index}")
        step['new_column_name'] = st.text_input("New Column Name", step.get('new_column_name', "enriched_col"), key=f"llm_col_{index}")
        st.info("Use {column_name} to reference values from other columns in your prompt.")
        step['prompt'] = st.text_area("LLM Prompt", step.get('prompt', "Based on the value in {column_to_use}, generate a new value."), key=f"llm_prompt_{index}")

    elif step['type'] == 'Semantic Loop':
        st.info("Searches an indexed collection for matches to text in a loop table.")
        col1, col2 = st.columns(2)
        with col1:
            step['input_step_1'] = st.selectbox("Loop Table (Input)", input_ids, key=f"sem_in1_{index}")
            step['query_column'] = st.selectbox("Column with Text to Search", get_cols('input_step_1'), key=f"sem_q_col_{index}")
        with col2:
            step['search_collection_prefix'] = st.selectbox("Indexed Collection", get_indexed_collections_list(), key=f"sem_in2_{index}")
        c1, c2, c3 = st.columns(3)
        with c1:
            step['top_k'] = st.number_input("Top K", min_value=1, value=step.get('top_k', 5), key=f"sem_top_k_{index}")
        with c2:
            step['similarity_threshold'] = st.slider("Similarity Threshold", 0.0, 1.0, value=step.get('similarity_threshold', 0.7), step=0.01, key=f"sem_threshold_{index}")
        with c3:
            step['join_type'] = st.selectbox("Join Type", ["left", "inner"], index=["left", "inner"].index(step.get('join_type', 'left')), key=f"sem_join_type_{index}")

    elif step['type'] == 'Merge':
        st.info("Joins two data steps together based on key columns.")
        col1, col2 = st.columns(2)
        with col1:
            step['input_step_1'] = st.selectbox("Left Input Step", input_ids, key=f"merge_in1_{index}")
            step['left_on'] = st.selectbox("Left Join Key", get_cols('input_step_1'), key=f"merge_left_on_{index}")
        with col2:
            step['input_step_2'] = st.selectbox("Right Input Step", input_ids, key=f"merge_in2_{index}")
            step['right_on'] = st.selectbox("Right Join Key", get_cols('input_step_2'), key=f"merge_right_on_{index}")
        step['how'] = st.selectbox("Join Type", ['inner', 'left', 'right', 'outer'], key=f"merge_how_{index}")

    elif step['type'] == 'Deduplicate':
        step['input_step_1'] = st.selectbox("Input Step", input_ids, key=f"dedup_input_{index}")
        st.info("If no columns are selected, all columns will be used to identify duplicate rows.")
        step['subset'] = st.multiselect("Columns to consider for duplicates", get_cols('input_step_1'), key=f"dedup_cols_{index}")

    elif step['type'] == 'Validate':
        step['input_step_1'] = st.selectbox("Input Step", input_ids, key=f"val_input_{index}")
        col1, col2 = st.columns(2)
        with col1:
            step['column'] = st.selectbox("Column to Validate", get_cols('input_step_1'), key=f"val_col_{index}")
        with col2:
            step['rule'] = st.selectbox("Validation Rule", ['not_null', 'is_unique'], key=f"val_rule_{index}")

    elif step['type'] == 'Save & Index':
        step['input_step_1'] = st.selectbox("Input Step", input_ids, key=f"save_input_{index}")
        step['table_name'] = st.text_input("New Base Table Name", step.get('table_name', f"pipeline_output_{step_id}"), key=f"save_name_{index}")

    # --- Step Management Buttons ---
    st.markdown("---")
    cols = st.columns([1, 1, 1, 1, 1.5])
    with cols[0]:
        if st.button("🗑️ Delete", key=f"del_{index}", use_container_width=True):
            st.session_state.pipeline_steps.pop(index)
            if step_id in st.session_state.pipeline_results:
                del st.session_state.pipeline_results[step_id]
            st.rerun()
    with cols[1]:
        if index > 0:
            if st.button("🔼 Move Up", key=f"up_{index}", use_container_width=True):
                st.session_state.pipeline_steps.insert(index - 1, st.session_state.pipeline_steps.pop(index))
                st.rerun()
    with cols[2]:
        if index < len(st.session_state.pipeline_steps) - 1:
            if st.button("🔽 Move Down", key=f"down_{index}", use_container_width=True):
                st.session_state.pipeline_steps.insert(index + 1, st.session_state.pipeline_steps.pop(index))
                st.rerun()
    with cols[4]:
        if st.button("▶️ Run Just This Step", key=f"run_{index}", use_container_width=True):
            execute_steps([step_id])
            
if __name__ == "__main__":
    main()