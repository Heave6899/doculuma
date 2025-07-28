# app.py
import streamlit as st
import pandas as pd
from pathlib import Path
from streamlit_option_menu import option_menu
# Import modularized components
import config
import db_utils
import dialogs
from _pages import chat_with_data, dynamic_pipeline, homepage, pipeline_steps, settings, uni_query, upload, sql_console, version_history, json_explorer, table_editor, visualization
from integrations import unqork_integration

def render_sidebar(conn):
    """Renders the sidebar for table browsing and management."""
    st.sidebar.header("🔍 Browse & Manage")
    
    db_mtime = Path(config.DB_PATH).stat().st_mtime if Path(config.DB_PATH).exists() else 0
    all_tables = db_utils.list_tables(conn, db_mtime)

    try:
        versions_df = pd.read_sql_query("SELECT snapshot_table FROM versions", conn)
        snapshot_tables = versions_df["snapshot_table"].tolist()
    except Exception:
        snapshot_tables = []
        
    base_tables = sorted([t for t in all_tables if t not in snapshot_tables and t != "versions"])

    # --- Base Tables Expander ---
    with st.sidebar.expander("Base Tables", expanded=True):
        preview_tbl = st.selectbox("Select a base table", [""] + base_tables, key="preview_base")
        if preview_tbl:
            n_rows = st.slider("Rows to preview", 1, config.MAX_PREVIEW_ROWS, 10, key=f"rows_{preview_tbl}")
            df_preview = pd.read_sql_query(f"SELECT * FROM `{preview_tbl}` LIMIT {n_rows}", conn)
            st.dataframe(df_preview, use_container_width=True)
            if st.button("🗑️ Delete Table", key=f"del_base_{preview_tbl}"):
                dialogs.confirm_delete_table(conn, preview_tbl)

    # --- Snapshots Expander ---
    with st.sidebar.expander("Snapshots", expanded=False):
        preview_snap = st.selectbox("Select a snapshot", [""] + snapshot_tables, key="preview_snap")
        if preview_snap:
            base_table_guess = preview_snap.split('_v')[0]
            n_rows_snap = st.slider("Rows to preview", 1, config.MAX_PREVIEW_ROWS, 10, key=f"rows_snap_{preview_snap}")
            df_snap = pd.read_sql_query(f"SELECT * FROM `{preview_snap}` LIMIT {n_rows_snap}", conn)
            st.dataframe(df_snap, use_container_width=True)
            
            col1, col2 = st.columns(2)
            if col1.button("🌱 Restore", key=f"restore_snap_{preview_snap}", use_container_width=True):
                db_utils.restore_snapshot(conn, preview_snap, base_table_guess)
                st.rerun()
            if col2.button("🗑️ Delete", key=f"del_snap_{preview_snap}", use_container_width=True):
                dialogs.confirm_delete_snapshot(conn, preview_snap)

def main():
    """Main function to run the Streamlit application."""
    st.set_page_config(layout="wide", page_title="DocuLuma")
    pages = ["Homepage", "Upload & Version", "SQL Console","Uni-Query","Dynamic Pipeline", "Chat with Data","Version History", "JSON Explorer", "Table Editor", "Data Visualization","Settings"]
    if st.session_state.get("integrations", {}).get("unqork", False):
        pages.append("Integrations > Unqork")
    if "navigate_to" in st.session_state:
        # Set the state of the navbar's key
        st.session_state.main_nav = st.session_state.navigate_to
        # Clean up the temporary key
        del st.session_state.navigate_to
        
    # Handler for persistent toast notifications
    if "toast_message" in st.session_state:
        st.toast(st.session_state.toast_message, icon="✅")
        del st.session_state.toast_message

    # --- TOP NAVBAR ---
    # The 'key' parameter is what makes the selection persistent across reruns
    selected_page = option_menu(
        menu_title=None,
        options=pages,
        icons=["house-door-fill","1-circle-fill", "terminal-fill","bi-terminal-split", "diagram-3-fill", "chat-dots-fill", "clock-history", "braces", "pencil-square","bar-chart-line-fill","gear-fill"],
        orientation="horizontal",
        key="main_nav",
        styles={
            "container": {
                "padding": "0 !important", 
                "background-color": "#0E1117", # Match Streamlit's dark theme
                "position": "sticky",
                "top": "0",
                "z-index": "1000"
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#333"
            },
            "nav-link-selected": {"background-color": "#02ab21"},
        }
    )
    
    conn = db_utils.get_connection(str(config.DB_PATH))
    render_sidebar(conn) # The sidebar no longer handles navigation, only browsing

    # --- Render the selected page ---
    if selected_page == "Homepage":
        homepage.render(conn)
    elif selected_page == "Upload & Version":
        upload.render(conn)
    elif selected_page == "SQL Console":
        sql_console.render(conn)
    elif selected_page == "Dynamic Pipeline":  # <-- ADDED HERE
        dynamic_pipeline.main()
    elif selected_page == "Version History":
        version_history.render(conn)
    elif selected_page == "JSON Explorer":
        json_explorer.render(conn)
    elif selected_page == "Table Editor":
        table_editor.render(conn)
    elif selected_page == "Data Visualization": # <-- ADDED
        visualization.render(conn)
    elif selected_page == "Settings": # Added page
        settings.render(conn)
    elif selected_page == "Integrations > Unqork" and st.session_state.get("integrations", {}).get("unqork", False):
        unqork_integration.render(conn)
    elif selected_page == "Uni-Query":
        uni_query.render(conn)
    elif selected_page == "Chat with Data":
        chat_with_data.render(conn)

if __name__ == "__main__":
    main()