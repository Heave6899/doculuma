import streamlit as st
import pandas as pd
import re
import config
import db_utils
import file_processors
import unqork_integration_utils
import json

chroma_client = db_utils.get_chroma_client()
embedding_fn = db_utils.get_embedding_function()
def render(conn):
    st.header("🔌 Unqork Integration")

    if not st.session_state.get("integrations", {}).get("unqork", False):
        st.warning("Unqork integration is currently disabled. Enable it in the Settings page.")
        return

    st.markdown("This integration downloads Unqork modules, extracts input fields, and allows saving them to a versioned database table.")

    token = st.session_state.get("unqork_api_token", "")
    if not token:
        st.error("Unqork API token is missing. Please enter it in the Settings page.")
        return

    folder_path = "data_saves/module_defs"

    if st.button("🔄 Refresh Modules"):
        with st.spinner("Fetching module list from Unqork..."):
            module_list = unqork_integration_utils.fetch_list_of_modules(token)

        if not module_list:
            st.error("No modules found or failed to fetch module list.")
            return

        st.success(f"Found {len(module_list)} modules. Downloading definitions...")

        progress_bar = st.progress(0)
        total = len(module_list)
        status_text = st.empty()

        for i, mod in enumerate(module_list):
            status_text.text(f"Downloading module {i+1} of {total}")
            unqork_integration_utils.fetch_full_definition_for_each_module([mod], token)
            progress_bar.progress((i + 1) / total)

        st.success("Modules downloaded successfully.")
        st.session_state.pop("unqork_extracted_df", None)

    if st.button("🧩 Extract Config"):
        with st.spinner("Extracting input fields from downloaded modules..."):
            df = unqork_integration_utils.extract_unqork_module_inputs(
                conn, folder_path=folder_path, table_name="", persist=False
            )
        st.session_state["unqork_extracted_df"] = df

    df = st.session_state.get("unqork_extracted_df")
    if isinstance(df, pd.DataFrame) and not df.empty:
        st.markdown("### Extracted Input Fields")
        if len(df) > 1000 or len(df.columns) > 20:
            st.info("Large dataset detected: showing only a sample for preview.")
        preview_cols = [col for col in df.columns if not col.startswith('embedding_')]
        st.dataframe(df[preview_cols].head(50), use_container_width=True)
        st.caption("Showing sample rows and columns for preview only. Full data will be saved to database.")
        embed_cols = st.multiselect(
            "Optional: Select columns to embed as vectors (semantic search, similarity, etc):",
            options=list(df.columns),
            key="embedcols_unqork"
        )

        default_tbl = "unqork_inputs"
        tbl_name = st.text_input("Table name to save extracted inputs", value=default_tbl).strip()

        if not re.match(config.TABLE_NAME_REGEX, tbl_name):
            st.error("Invalid table name. Use letters, numbers, and underscores, starting with a letter or underscore.")
            return

        all_tables = db_utils.list_tables(conn, 0)
        overwrite = tbl_name in all_tables and st.checkbox("Overwrite existing table?", key="overwrite_unqork")

        if st.button("💾 Save Extracted Inputs"):
            for col in df.columns:
                if df[col].apply(lambda x: isinstance(x, list)).any():
                    df[col] = df[col].apply(json.dumps)
            if tbl_name in all_tables and not overwrite:
                st.error(f"Table '{tbl_name}' exists. Enable overwrite to proceed.")
            else:
                snap = db_utils.persist_version(
                    conn, df, tbl_name, 0, df.columns.tolist(), {c: c for c in df.columns}
                )
                st.session_state.toast_message = f"Saved extracted inputs to version `{snap}`"
                if embed_cols:
                        with st.spinner("Generating embeddings and saving to ChromaDB..."):
                            file_processors.embed_and_save_to_chroma(chroma_client, df, tbl_name, embed_cols, embedding_fn)
                        # Success toast for ChromaDB
                        st.success(f"Embeddings successfully saved in ChromaDB for table `{tbl_name}` 🚀")
                db_utils.list_tables.clear()
                st.rerun()
