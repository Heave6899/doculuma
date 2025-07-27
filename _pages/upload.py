# pages/upload.py
import streamlit as st
import pandas as pd
from pathlib import Path
import re
import tempfile
import json
import config
import db_utils
import dataframe_utils
import json_utils
import file_processors
import io

chroma_client = db_utils.get_chroma_client()
embedding_fn = db_utils.get_embedding_function()
def handle_json_upload(_conn, file):
    """UI handler for processing a single JSON file upload."""
    st.subheader(f"Processing: `{file.name}`")
    try:
        file.seek(0)
        data = json.load(file)
    except Exception as e:
        st.error(f"Invalid JSON in {file.name}: {e}")
        return

    # --- Flatten data for storage/querying ---
    records = data if isinstance(data, list) else [data]
    flat_records = [json_utils.deep_flatten_json(rec) for rec in records]
    df_data = pd.DataFrame(flat_records)
    df_data = json_utils.prepare_json_df_for_sql(df_data)

    st.markdown("**Data Preview (first 5 rows)**")
    st.dataframe(df_data.head(5), use_container_width=True)

    default_tbl = Path(file.name).stem.replace(" ", "_").replace("-", "_")
    tbl_name = st.text_input(
        "Table name for JSON data",
        value=default_tbl,
        key=f"tbl_json_data_{file.name}"
    ).strip()

    if not re.match(config.TABLE_NAME_REGEX, tbl_name):
        st.error("Invalid table name. Use letters, numbers, and underscores, starting with a letter or underscore.")
        return

    # Add overwrite checkbox, a critical feature from the original app
    all_tables = db_utils.list_tables(_conn, 0) # Use 0 to force a refresh if needed
    overwrite = tbl_name in all_tables and st.checkbox(
        "Overwrite existing table?", key=f"ovw_json_data_{file.name}"
    )

    if st.button("Save JSON Data as Versioned Table", key=f"save_json_data_{file.name}"):
        if tbl_name in all_tables and not overwrite:
            st.error(f"Table '{tbl_name}' exists. Enable overwrite to proceed.")
        else:
            snap = db_utils.persist_version(
                _conn, df_data, tbl_name, 0,
                df_data.columns.tolist(),
                {c: c for c in df_data.columns}
            )
            st.write("DEBUG: Save action is running...") 
            st.session_state.toast_message = f"Saved data to version `{snap}`"
            db_utils.list_tables.clear()



            st.rerun()

    # --- Detect + persist JSON Schema structure ---
    if isinstance(data, dict) and any(k in data for k in ["properties", "definitions", "components"]):
        st.markdown("---")
        st.markdown("#### Schema Preview")
        df_schema = json_utils.flatten_json_schema(data)
        st.dataframe(df_schema, use_container_width=True)

        schema_tbl_name = st.text_input("Table name for schema", f"{tbl_name}_schema", key=f"tbl_schema_{file.name}").strip()
        overwrite_schema = schema_tbl_name in all_tables and st.checkbox(
            f"Overwrite existing `{schema_tbl_name}`?", key=f"ovw_schema_{file.name}"
        )

        if st.button("Save JSON Schema as Versioned Table", key=f"save_json_schema_{file.name}"):
            if schema_tbl_name in all_tables and not overwrite_schema:
                st.error(f"Schema table '{schema_tbl_name}' exists. Enable overwrite to proceed.")
            else:
                snap2 = db_utils.persist_version(
                    _conn, df_schema, schema_tbl_name, 0,
                    df_schema.columns.tolist(),
                    {c: c for c in df_schema.columns}
                )
                st.session_state.toast_message = f"Saved schema to version `{snap2}`"
                db_utils.list_tables.clear()
                st.rerun()


def render(conn):
    """Renders the main file upload and versioning page."""
    st.header("1️⃣ Upload & Save Versions")
    enable_folder_import = st.session_state.get("integrations", {}).get("folder_import", False)

    folder_files = []
    folder_path = None
    if enable_folder_import:
        st.markdown("### 📁 Import Files From a Folder")
        with st.expander("Import from local folder (advanced users)", expanded=False):
            folder_path = st.text_input(
                "Enter local folder path", value=st.session_state.get("upload_folder_path", "")
            )
            if folder_path:
                st.session_state["upload_folder_path"] = folder_path
                folder = Path(folder_path)
                if not folder.exists() or not folder.is_dir():
                    st.error("Invalid folder path.")
                else:
                    # Only show files with supported extensions
                    supported_exts = set(config.SUPPORTED_EXTS)
                    files = [f for f in folder.iterdir() if f.is_file() and f.suffix.lower().strip(".") in supported_exts]
                    if not files:
                        st.info("No supported files found in this folder.")
                    else:
                        selected_files = st.multiselect(
                            "Select files to import",
                            options=[f.name for f in files],
                            key="folder_import_selected_files"
                        )
                        # Open files and create file-like objects for each selected
                        for fname in selected_files:
                            file_path = folder / fname
                            st.write(f"Trying to open: {file_path}")  # DEBUG
                            if not file_path.exists():
                                st.error(f"File not found: {file_path}")
                                continue
                            with open(file_path, "rb") as f:
                                file_bytes = f.read()
                                virtual_file = io.BytesIO(file_bytes)
                                # Name it like an uploaded file
                                virtual_file.name = fname
                                folder_files.append(virtual_file)


    uploaded_files = st.file_uploader(
        "Upload files (.xlsx, .csv, .ods, .json)",
        type=list(config.SUPPORTED_EXTS),
        accept_multiple_files=True,
    )

    all_files = []

    if uploaded_files:
        all_files.extend(uploaded_files)
    if folder_files:
        all_files.extend(folder_files)

    if "parsed_uploads" not in st.session_state:
        st.session_state["parsed_uploads"] = {}

    for file in all_files:
        st.markdown("---")
        ext = Path(file.name).suffix.lower().strip(".")
        
        if ext == "json":
            handle_json_upload(conn, file)
            continue

        try:
            excel_file = pd.ExcelFile(file) if ext in ["xlsx", "xls", "ods"] else None
            sheets = excel_file.sheet_names if excel_file else [None]
        except Exception as e:
            st.error(f"Failed to read {file.name}: {e}")
            continue

        for sheet in sheets:
            label = f"Sheet: `{sheet}`" if sheet else f"File: `{file.name}`"
            key_prefix = f"{file.name}_{sheet}"
            st.markdown(f"#### Processing {label}")

            header_row = st.number_input("Header is in row", 0, 100, 0, key=f"hdr_{key_prefix}")
            cache_key = (file.name, sheet, header_row)

            # Parse/calculate only if not already in cache!
            if cache_key not in st.session_state["parsed_uploads"]:
                try:
                    file.seek(0)
                    df = pd.read_excel(file, sheet_name=sheet, header=header_row) if excel_file else pd.read_csv(file, header=header_row, on_bad_lines='warn')
                    df = dataframe_utils.sanitize_df(df)

                    if file_processors.should_apply_upx_json_path(file.name):
                        df = file_processors.process_upx_data_dictionary(df.copy())

                    # --- Strikethrough detection (only ONCE here) ---
                    if ext == "xlsx":
                        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as tmp:
                            file.seek(0)
                            tmp.write(file.read())
                            tmp.flush()
                            strikes = file_processors.get_strikethrough_flags_xlsx(tmp.name, sheet, header_row)
                            flag_col = []
                            for i, row in enumerate(df.values):
                                strike_flags = strikes[i] if strikes and i < len(strikes) else ["none"] * df.shape[1]
                                flag = file_processors.classify_strikethrough_row(strike_flags, row)
                                flag_col.append(flag)
                            df["strikethrough_flag"] = flag_col

                    st.session_state["parsed_uploads"][cache_key] = df
                except Exception as e:
                    st.error(f"Error parsing file with header at row {header_row}: {e}")
                    continue

            df = st.session_state["parsed_uploads"][cache_key]
            st.dataframe(df.head(20), use_container_width=True)

            cols = df.columns.tolist()
            keep_cols = st.multiselect("Columns to keep", cols, default=cols, key=f"keep_{key_prefix}")

            rename_df = pd.DataFrame({"Current Name": keep_cols, "New Name": keep_cols})
            edited_rename_df = st.data_editor(rename_df, key=f"rename_{key_prefix}", use_container_width=True)
            rename_map = {row["Current Name"]: row["New Name"] for _, row in edited_rename_df.iterrows()}

            df_final = df[keep_cols].rename(columns=rename_map)

            default_tbl_name = f"{Path(file.name).stem}_{sheet or ''}".replace(" ", "_").replace("-", "_")
            tbl_name = st.text_input("Save as table name", default_tbl_name, key=f"tbl_{key_prefix}").strip()
            
            if not re.match(config.TABLE_NAME_REGEX, tbl_name):
                st.error("Invalid table name.")
                continue

            all_tables = db_utils.list_tables(conn, 0)
            overwrite = tbl_name in all_tables and st.checkbox(
                "Overwrite existing table?", key=f"ovw_{key_prefix}"
            )

            embed_cols = st.multiselect(
                "Optional: Select columns to embed as vectors (semantic search, similarity, etc):",
                options=list(df_final.columns),
                key=f"embedcols_{tbl_name}"
            )

            # # if embed_cols:
            # #     with st.spinner("Generating embeddings and saving to ChromaDB..."):
            # #         file_processors.embed_and_save_to_chroma(chroma_client, df, tbl_name, embed_cols, embedding_fn)
            # #         st.success(f"Embeddings saved in ChromaDB for table: {tbl_name}")
            
            
            if st.button("💾 Save as New Version with Embeddings", key=f"save_{key_prefix}"):
                if tbl_name in all_tables and not overwrite:
                    st.error(f"Table '{tbl_name}' exists. Enable overwrite to proceed.")
                else:
                    # Step 1: Save to SQLite
                    snap = db_utils.persist_version(
                        conn, df_final, tbl_name, header_row, keep_cols, rename_map
                    )
                    # Success toast for SQLite
                    st.success(f"SQLite data saved to version `{snap}` 🎉")

                    # Step 2: Save embeddings to ChromaDB (if columns selected)
                    if embed_cols:
                        with st.spinner("Generating embeddings and saving to ChromaDB..."):
                            file_processors.embed_and_save_to_chroma(chroma_client, df_final, tbl_name, embed_cols, embedding_fn)
                        # Success toast for ChromaDB
                        st.session_state.toast_message = f"Embeddings successfully saved in ChromaDB for table `{tbl_name}` 🚀"

                    # Refresh table listing
                    db_utils.list_tables.clear()
                    st.session_state.toast_message = f"All data and embeddings saved successfully for `{tbl_name}`."
                    st.rerun()

