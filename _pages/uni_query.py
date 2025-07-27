from file_processors import embed_and_save_to_chroma
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

# Helper function to detect if a query is SQL
def is_sql_query(query):
    try:
        parsed = sqlparse.parse(query)
        return bool(parsed and parsed[0].get_type() != 'UNKNOWN')
    except:
        return False

# Render Unified Query Console
def render(conn):
    st.header("🧠 Unified Query Console")

    # --- Session state setup ---
    if "uni_query_input_box" not in st.session_state:
        st.session_state["uni_query_input_box"] = ""
    if "uni_query_results" not in st.session_state:
        st.session_state["uni_query_results"] = []
    if "last_sql_result" not in st.session_state:
        st.session_state["last_sql_result"] = None
    if "last_sql_table" not in st.session_state:
        st.session_state["last_sql_table"] = None

    db_mtime = config.DB_PATH.stat().st_mtime if config.DB_PATH.exists() else 0
    all_tables = db_utils.list_tables(conn, db_mtime)
    chroma_client = db_utils.get_chroma_client()
    with st.expander("🗂️ Load a Saved Query"):
        try:
            saved_queries_df = pd.read_sql_query("SELECT name, sql_text FROM saved_queries ORDER BY saved_at DESC", conn)
            if saved_queries_df.empty:
                st.info("No queries have been saved yet.")
            else:
                query_map = pd.Series(saved_queries_df.sql_text.values, index=saved_queries_df.name).to_dict()
                query_name_to_load = st.selectbox("Select a saved query to load", options=query_map.keys())

                if st.button("Load Selected Query"):
                    st.session_state.uni_query_input_box = query_map[query_name_to_load]
                    st.rerun()

        except Exception as e:
            st.error(f"Could not load saved queries. Error: {e}")

    query_text = st_textcomplete_autocomplete(
        label="Enter SQL or Natural Language query:",
        options=all_tables,
        placeholder="Enter your query here...",
        height=200,
        max_count=10,
        key="uni_query_input_box"
    )

    # --- SQL/NL query execution ---
    if st.button("Execute", key="run_uni_query"):
        if query_text.strip():
            if is_sql_query(query_text):
                st.info("Detected as SQL query.")
                try:
                    df_result = pd.read_sql_query(query_text, conn)
                    st.session_state["last_sql_result"] = df_result
                    # Extract table name for later (semantic search needs to know which collection)
                    match = re.search(r'from\s+([^\s;]+)', query_text, re.IGNORECASE)
                    if match:
                        table_name = match.group(1).replace('`', '').replace('"', '')
                    else:
                        table_name = None
                    st.session_state["last_sql_table"] = table_name
                    st.dataframe(df_result)

                    # Save table option
                    if not df_result.empty:
                        with st.form(key="save_sql_result_form"):
                            new_tbl_name = st.text_input("Save result as table", "")
                            default_cols = [col for col in df_result.columns if col != "id"]
                            embed_cols = st.multiselect(
                                "Select columns to embed:",
                                options=[col for col in df_result.columns if col != "id"],  # or just df_result.columns.tolist()
                                default=default_cols
                            )
                            submitted = st.form_submit_button("Save Result")
                            if submitted:
                                if not re.match(config.TABLE_NAME_REGEX, new_tbl_name):
                                    st.error("Invalid table name.")
                                else:
                                    snap = db_utils.persist_version(conn, df_result, new_tbl_name, 0, df_result.columns.tolist(), {c: c for c in df_result.columns})
                                    conn.execute("INSERT INTO saved_queries (name, base_table, sql_text) VALUES (?, ?, ?)", (new_tbl_name, snap, query_text))
                                    conn.commit()
                                    st.success(f"Saved result to `{new_tbl_name}`.")
                                    if embed_cols:
                                        with st.spinner("Generating embeddings and saving to ChromaDB..."):
                                            embed_and_save_to_chroma(
                                                chroma_client=chroma_client,
                                                df=df_result,
                                                table_name=new_tbl_name,
                                                embed_cols=embed_cols,
                                                embedding_fn=embedding_fn,
                                            )
                except Exception as e:
                    st.error(f"SQL Error: {e}")

            else:
                st.info("Detected as Natural Language query. Performing semantic search...")
                results = db_utils.semantic_search_chroma(chroma_client, query_text, config.CHROMA_COLLECTION_PREFIX)

                # ... in the block handling semantic search ...
                if results:
                    for table, matches in results.items():
                        st.subheader(f"Table: {table}")
                        # Collect matched IDs and similarity scores
                        row_ids = []
                        similarities = {}
                        for match in matches:
                            stable_id = match['metadata'].get('stable_id') or match['metadata'].get('id')
                            if stable_id:
                                row_ids.append(stable_id)
                                similarities[stable_id] = round(match['similarity'] * 100, 2)

                        if row_ids:
                            # Fetch the full rows from SQL table for these IDs
                            placeholders = ",".join("?" for _ in row_ids)
                            query = f'SELECT * FROM "{table}" WHERE id IN ({placeholders})'
                            full_rows_df = pd.read_sql_query(query, conn, params=row_ids)
                            if not full_rows_df.empty:
                                full_rows_df["Similarity %"] = full_rows_df["id"].map(similarities)
                                full_rows_df = full_rows_df.sort_values("Similarity %", ascending=False)
                                st.dataframe(full_rows_df)
                            else:
                                st.warning("No matching rows found in the database.")
                        else:
                            st.info("No matches found in semantic search.")
                else:
                    st.warning("No semantic matches found.")



    df_result = st.session_state.get("last_sql_result", None)
    table_name = st.session_state.get("last_sql_table", None)

    if isinstance(df_result, pd.DataFrame) and not df_result.empty:
        st.markdown("**Last SQL Results:**")
        st.dataframe(df_result)

    if isinstance(df_result, pd.DataFrame) and not df_result.empty and "id" in df_result.columns:
        st.markdown("**Semantic search within last SQL results:**")
        search_phrase = st.text_input("Enter semantic search (optional):", key="sql_semantic_search")
        if search_phrase.strip():
            if not table_name:
                table_name = st.text_input("Enter table name for semantic filtering:", key="table_name_semantic")
            if table_name:
                collection_name = f"{config.CHROMA_COLLECTION_PREFIX}_{table_name}"
                collection = db_utils.get_chroma_client().get_collection(collection_name)
                stable_ids = df_result['id'].tolist()
                results = collection.query(
                    query_texts=[search_phrase],
                    n_results=50,
                    where={"stable_id": {"$in": stable_ids}}
                )
                # Reworked: fetch full rows for matched IDs
                if results['documents'] and results['metadatas']:
                    matched_ids = []
                    similarities = {}
                    for meta, dist in zip(results['metadatas'][0], results['distances'][0]):
                        stable_id = meta.get("stable_id")
                        if stable_id:
                            matched_ids.append(stable_id)
                            similarities[stable_id] = round((1 - dist) * 100, 2)
                    if matched_ids:
                        # Restrict to IDs in current SQL result (safe to fetch from df_result)
                        filtered_rows = df_result[df_result['id'].isin(matched_ids)].copy()
                        filtered_rows["Similarity %"] = filtered_rows["id"].map(similarities)
                        filtered_rows = filtered_rows.sort_values("Similarity %", ascending=False)
                        st.dataframe(filtered_rows)
                    else:
                        st.warning("No matching IDs found in semantic results for current SQL output.")
                else:
                    st.warning("No semantic matches found in SQL output.")
