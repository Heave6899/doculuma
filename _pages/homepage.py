import streamlit as st
import pandas as pd
import humanize
from datetime import datetime
import os

import config
import db_utils

chroma_client = db_utils.get_chroma_client()
def render(conn):
    """Renders the Homepage Dashboard."""
    st.header("🏠 Homepage Dashboard")
    st.markdown("A summary of your versioned data assets.")

    # --- Key Metrics ---
    st.markdown("---")
    try:
        # Fetch stats in a single query for efficiency
        stats = conn.execute(
            "SELECT COUNT(DISTINCT base_table), COUNT(snapshot_table), MAX(timestamp) FROM versions"
        ).fetchone()
        
        num_tables = stats[0] or 0
        num_versions = stats[1] or 0
        last_activity_ts = stats[2] or None
        
        if last_activity_ts:
            last_activity = humanize.naturaltime(datetime.now() - datetime.strptime(last_activity_ts, "%Y%m%d%H%M%S%f"))
        else:
            last_activity = "N/A"
            
        db_size_bytes = os.path.getsize(config.DB_PATH)
        db_size = humanize.naturalsize(db_size_bytes)

    except Exception:
        num_tables, num_versions, last_activity, db_size = 0, 0, "N/A", "0 Bytes"

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Base Tables", f"{num_tables}")
    c2.metric("Total Versions", f"{num_versions}")
    c3.metric("Last Update", f"{last_activity}")
    c4.metric("DB Size", f"{db_size}")
    collections = chroma_client.list_collections()
    num_collections = len(collections)
    total_embeddings = sum(col.count() for col in collections)
    c5.metric("ChromaDB Collections", f"{num_collections}")
    c6.metric("Total Embeddings", f"{total_embeddings:,}")
    # --- Quick Actions ---
    st.markdown("---")
    st.subheader("🚀 Quick Actions")
    c1, c2, c3 = st.columns(3)
    
    if c1.button("➕ Upload & Version a File", use_container_width=True, type="primary"):
        st.session_state.navigate_to = "Upload & Version"
        st.rerun()
        
    if c2.button("🔍 Run a SQL Query", use_container_width=True):
        st.session_state.navigate_to = "SQL Console"
        st.rerun()
        
    if c3.button("📝 Edit a Table", use_container_width=True):
        st.session_state.navigate_to = "Table Editor"
        st.rerun()
        
    # --- Recent Activity ---
    st.markdown("---")
    st.subheader("⏱️ Recent Activity")
    try:
        recent_df = pd.read_sql_query(
            "SELECT timestamp, base_table, snapshot_table FROM versions ORDER BY timestamp DESC LIMIT 5",
            conn,
            dtype={"timestamp": "str"}
        )
        if recent_df.empty:
            st.info("No activity yet. Upload a file to get started.")
        else:
            # Humanize the timestamp
            recent_df['timestamp'] = recent_df['timestamp'].apply(
                lambda ts: humanize.naturaltime(datetime.now() - datetime.strptime(ts, "%Y%m%d%H%M%S%f"))
            )
            recent_df.rename(columns={"timestamp": "When", "base_table": "Base Table", "snapshot_table": "Snapshot Created"}, inplace=True)
            st.dataframe(recent_df, use_container_width=True)
            
    except Exception as e:
        st.error(f"Could not load recent activity. Error: {e}")