# pages/table_editor.py
import streamlit as st
import pandas as pd
import json
import config
import db_utils
import dataframe_utils
import file_processors

def render(conn):
    """Renders the upgraded Full Table Editor with data profiling, advanced filtering, and row deletion."""
    st.header("5️⃣ Full Table Editor")

    db_mtime = config.DB_PATH.stat().st_mtime if config.DB_PATH.exists() else 0
    all_tables = db_utils.list_tables(conn, db_mtime)
    try:
        snapshot_tables = pd.read_sql_query("SELECT snapshot_table FROM versions", conn)["snapshot_table"].tolist()
    except Exception:
        snapshot_tables = []
    base_tables = [t for t in all_tables if t not in snapshot_tables and t != "versions"]

    tbl_to_edit = st.selectbox("Select a base table to edit", [""] + base_tables, help="Choose a main table to view and edit its data.")
    if not tbl_to_edit:
        st.info("Select a base table to begin editing.")
        return

    # --- Load and Cache Data ---
    @st.cache_data(ttl=3600)
    def load_data(table_name):
        return pd.read_sql_query(f"SELECT * FROM `{table_name}`", conn)

    df_original = load_data(tbl_to_edit)
    
    # --- Data Profiling Report ---
    with st.expander("📊 Data Profile & Stats"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", f"{df_original.shape[0]:,}")
        c2.metric("Columns", f"{df_original.shape[1]:,}")
        c3.metric("Missing Cells", f"{df_original.isna().sum().sum():,}")
        
        st.markdown("##### Column Details")
        null_counts = df_original.isna().sum()
        profile_df = pd.DataFrame({
            "Dtype": df_original.dtypes.astype(str),
            "Nulls": null_counts,
            "Nulls (%)": (null_counts / df_original.shape[0] * 100).round(2),
        })
        st.dataframe(profile_df, use_container_width=True)


    # --- Initialize a working copy of the dataframe in session state ---
    if f"df_edit_copy_{tbl_to_edit}" not in st.session_state:
         st.session_state[f"df_edit_copy_{tbl_to_edit}"] = df_original.copy()
    
    df_filtered = st.session_state[f"df_edit_copy_{tbl_to_edit}"]

    # --- Advanced Filtering and Sorting UI ---
    with st.expander("🔍 Filter & Sort Data"):
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Filter by Column Values")
            filter_cols = st.multiselect(
                "Filter by columns:", options=list(df_filtered.columns), default=list(df_filtered.columns[:2])
            )
            for col in filter_cols:
                if pd.api.types.is_numeric_dtype(df_filtered[col].dtype) and df_filtered[col].nunique() > 1:
                    minv, maxv = float(df_filtered[col].min()), float(df_filtered[col].max())
                    start, end = st.slider(f"Range for {col}", minv, maxv, (minv, maxv))
                    df_filtered = df_filtered[df_filtered[col].between(start, end)]
                elif pd.api.types.is_object_dtype(df_filtered[col].dtype):
                     search_term = st.text_input(f"Search in {col} (case insensitive)")
                     if search_term:
                         df_filtered = df_filtered[df_filtered[col].str.contains(search_term, case=False, na=False)]
                elif df_filtered[col].nunique() > 1:
                    options = sorted(df_filtered[col].dropna().unique().tolist())
                    selected = st.multiselect(f"Values for {col}", options, default=options)
                    df_filtered = df_filtered[df_filtered[col].isin(selected)]
        with c2:
            st.markdown("##### Sort Data")
            sort_col = st.selectbox("Sort by column", [""] + list(df_filtered.columns))
            if sort_col:
                sort_asc = st.toggle("Sort Ascending", True)
                df_filtered = df_filtered.sort_values(by=sort_col, ascending=sort_asc)
    
    # --- Editable Dataframe with Row Deletion ---
    st.markdown("### Edit Data")
    if "__delete__" not in df_filtered.columns:
        df_filtered.insert(0, "__delete__", False) # Add a column for deletion checkboxes
    edited_df = st.data_editor(
        df_filtered,
        num_rows="dynamic",
        use_container_width=True,
        key=f"editor_{tbl_to_edit}",
        disabled=list(df_filtered.columns.drop("__delete__"))
    )

    rows_to_delete = edited_df[edited_df["__delete__"] == True]
    chroma_client = db_utils.get_chroma_client()
    c1, c2 = st.columns([3, 1])
    with c1:
        if not rows_to_delete.empty:
            st.warning(f"You have selected {len(rows_to_delete)} row(s) for deletion.")
            if st.button(f"🗑️ Confirm Deletion of {len(rows_to_delete)} Row(s)"):
                row_ids = rows_to_delete.index.tolist()
                
                db_utils.delete_embeddings_from_chroma(chroma_client, tbl_to_edit, row_ids)
                st.session_state[f"df_edit_copy_{tbl_to_edit}"] = st.session_state[f"df_edit_copy_{tbl_to_edit}"].drop(rows_to_delete.index)
                st.session_state.toast_message = "Deleted rows and synchronized with ChromaDB."
                st.rerun()

    # --- Save Changes Logic ---
    with c2:
        if st.button("💾 Save All Changes", use_container_width=True, type="primary"):
            final_df = st.session_state[f"df_edit_copy_{tbl_to_edit}"].reset_index(drop=True)
            diff = dataframe_utils.get_simple_diff(df_original.reset_index(drop=True), final_df)

            if diff.empty:
                st.toast("No changes detected to save.")
            else:
                with st.expander("Review Changes Before Saving", expanded=True):
                    added = len(diff[diff.action == 'added'])
                    removed = len(diff[diff.action == 'removed'])
                    changed = len(diff[diff.action == 'changed'])
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Rows Added", added, delta_color="normal")
                    c2.metric("Rows Removed", removed, delta_color="inverse")
                    c3.metric("Cells Changed", changed)
                    st.dataframe(diff, use_container_width=True)

                meta = conn.execute("SELECT header_row, columns, rename_map FROM versions WHERE base_table = ? ORDER BY timestamp DESC LIMIT 1", (tbl_to_edit,)).fetchone()
                header_row, keep_cols, rename_map = (
                    (meta[0], json.loads(meta[1]), json.loads(meta[2])) if meta
                    else (0, final_df.columns.tolist(), {c: c for c in final_df.columns})
                )
                try:
                    snap = db_utils.persist_version(conn, final_df, tbl_to_edit, header_row, keep_cols, rename_map)
                    st.session_state.toast_message = f"Saved changes to new version `{snap}`"
                    embed_cols = [col.replace("embedding_", "") for col in final_df.columns if col.startswith("embedding_")]
                    if embed_cols:
                        with st.spinner("Updating embeddings in ChromaDB..."):
                            file_processors.embed_and_save_to_chroma(chroma_client, final_df, tbl_to_edit, embed_cols, db_utils.get_embedding_function())
                        st.success(f"ChromaDB embeddings updated for table `{tbl_to_edit}`")
                    del st.session_state[f"df_edit_copy_{tbl_to_edit}"]
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to save changes: {e}")