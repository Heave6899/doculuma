import file_processors
import llm_manager
import streamlit as st
import pandas as pd
import sqlparse
import re
from streamlit_autocomplete import st_textcomplete_autocomplete
import config
import db_utils
import llm_utils
chroma_client = db_utils.get_chroma_client()
embedding_fn = db_utils.get_embedding_function()
def render(conn):
    """Renders the SQL Console page."""
    st.header("2️⃣ Run SQL Query")

    if "sql_input_box" not in st.session_state:
        st.session_state["sql_input_box"] = ""
    if "sql_results" not in st.session_state:
        st.session_state["sql_results"] = []

    db_mtime = config.DB_PATH.stat().st_mtime if config.DB_PATH.exists() else 0
    all_tables = db_utils.list_tables(conn, db_mtime)

    with st.expander("🤖 Ask a question in natural language (LLM-powered)", expanded=False):
        schema_table_list = [t for t in all_tables if t != "versions"]
        selected_tables = st.multiselect(
            "Which tables should the LLM see?",
            schema_table_list,
            default=schema_table_list[:5] if len(schema_table_list) > 5 else schema_table_list
        )
        show_samples = st.checkbox("Include sample rows in context?", value=True)
        nl_question = st.text_area("Your question:", key="llm_nlq")
        if st.button("Generate SQL from question", key="llm_gen_sql"):
            if nl_question.strip():
                with st.spinner("Generating SQL..."):
                    llm = llm_manager.LLMManager()
                    schema_text = llm_utils.get_db_schema_for_prompt(conn, selected_tables, with_samples=show_samples)
                    prompt = (
                        f"You are an expert SQLite assistant. Here is the database schema:\n"
                        f"{schema_text}\n"
                        f"Write a valid SQLite SQL query to answer the question below. "
                        f"Return ONLY the SQL code, nothing else.\n"
                        f"Question: {nl_question}\nSQL:"
                    )
                    sql_result = llm.generate(prompt)

                    # --- THIS SECTION WAS MISSING ---
                    # Check if the generation was successful and display the result
                    if "Error:" not in sql_result and sql_result:
                        st.success("SQL Generated:")
                        st.code(sql_result, language="sql")
                    # The llm_manager already shows an st.error on failure, so no 'else' is needed.
                    # --- END OF FIX ---
            else:
                st.warning("Please enter a question.")

    with st.expander("🗂️ Load a Saved Query"):
        try:
            saved_queries_df = pd.read_sql_query("SELECT name, sql_text FROM saved_queries ORDER BY saved_at DESC", conn)
            if saved_queries_df.empty:
                st.info("No queries have been saved yet. Save a query result as a new table to add it here.")
            else:
                query_map = pd.Series(saved_queries_df.sql_text.values, index=saved_queries_df.name).to_dict()
                query_name_to_load = st.selectbox("Select a saved query to load", options=query_map.keys())
                
                if st.button("Load Selected Query"):
                    st.session_state.sql_input_box = query_map[query_name_to_load]
                    st.rerun()

        except Exception as e:
            st.error(f"Could not load saved queries. Error: {e}")

    sql_query_text = st_textcomplete_autocomplete(
        label="Enter SQL query (you can copy from above or load a saved query)",
        options=all_tables,
        placeholder="SELECT * FROM ...",
        height=200,
        max_count=10,
        key="sql_input_box"
    )

    multi_query_mode = st.checkbox("Multi-query mode (split by ';')", value=True)

    if st.button("Run SQL", key="run_sql"):
        st.session_state["sql_results"] = []
        st.session_state["multi_sql_errors"] = []
        query_input = sql_query_text or ""
        if query_input.strip():
            queries = sqlparse.split(query_input) if multi_query_mode else [query_input]
            cursor = conn.cursor()
            for idx, q in enumerate(queries, start=1):
                if not q.strip(): continue
                try:
                    if q.strip().lower().startswith(("select", "with", "pragma")):
                        df = pd.read_sql_query(q, conn)
                        st.session_state["sql_results"].append((f"Result {idx}", df, q))
                    else:
                        cursor.execute(q)
                        conn.commit()
                        st.session_state["sql_results"].append((f"Result {idx}", f"Command executed. Rows affected: {cursor.rowcount}", q))
                except Exception as e:
                    st.session_state["multi_sql_errors"].append((idx, q, str(e)))

    if "sql_results" in st.session_state and st.session_state["sql_results"]:
        st.markdown("---")
        st.subheader("Query Results")
        for idx, result, sql in st.session_state["sql_results"]:
            st.markdown(f"**{idx}:**")
            st.code(sql, language="sql")
            if isinstance(result, pd.DataFrame):
                display_cols = [c for c in result.columns if not c.startswith("embedding_")]
                st.dataframe(result[display_cols], use_container_width=True)
            else:
                st.success(result)
    
    if "multi_sql_errors" in st.session_state and st.session_state["multi_sql_errors"]:
        st.error("Some queries produced errors:")
        for idx, sql, err in st.session_state["multi_sql_errors"]:
            st.markdown(f"**Query {idx}:**")
            st.code(sql, language="sql")
            st.code(err)

    if "sql_results" in st.session_state:
        saveable_results = [(i, result, sql) for i, (idx, result, sql) in enumerate(st.session_state["sql_results"]) if isinstance(result, pd.DataFrame) and not result.empty]
        if saveable_results:
            st.markdown("---")
            st.subheader("Save a Result")
            options = {f"Result {i+1}: {sql[:50]}...": (df, sql) for i, df, sql in saveable_results}
            choice = st.selectbox("Select a result to save as a new table", options.keys())
            if choice:
                df_to_save, sql_to_log = options[choice]
                with st.form(key="save_sql_result_form"):
                    new_tbl_name = st.text_input("New table name").strip()
                    embed_cols = st.multiselect(
                                "Select columns to embed:",
                                options=[col for col in df_to_save.columns if col != "id"]  # or just df_result.columns.tolist()
                            )
                    submitted = st.form_submit_button("Save as Versioned Table")
                    if submitted:
                        if not re.match(config.TABLE_NAME_REGEX, new_tbl_name):
                            st.error("Invalid table name.")
                        else:
                            snap = db_utils.persist_version(conn, df_to_save, new_tbl_name, 0, df_to_save.columns.tolist(), {c:c for c in df_to_save.columns})
                            conn.execute("INSERT INTO saved_queries (name, base_table, sql_text) VALUES (?, ?, ?)",(new_tbl_name, snap, sql_to_log))
                            conn.commit()
                            st.session_state.toast_message = f"Saved result to `{new_tbl_name}` and logged SQL."
                            if embed_cols:
                                with st.spinner("Generating embeddings and saving to ChromaDB..."):
                                    file_processors.embed_and_save_to_chroma(chroma_client, df_to_save, new_tbl_name, embed_cols, embedding_fn)
                                # Success toast for ChromaDB
                                st.session_state.toast_message = f"Embeddings successfully saved in ChromaDB for table `{new_tbl_name}` 🚀"
                            st.rerun()