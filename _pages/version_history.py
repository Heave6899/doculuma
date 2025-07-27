import streamlit as st
import pandas as pd
import humanize
from datetime import datetime
import json
import config
import db_utils
import dialogs
import dataframe_utils

def render(conn):
    """Renders the Version History page with schema comparison."""
    st.header("3️⃣ Version History")
    
    try:
        # Fetch all metadata needed for comparison and display
        df_hist = pd.read_sql_query(
            "SELECT version_id, base_table, snapshot_table, timestamp, columns, rename_map FROM versions ORDER BY base_table, timestamp DESC",
            conn,
            dtype={"timestamp": "str"}
        )
    except Exception:
        st.warning("Could not load version history. Has a table been versioned yet?")
        return

    if df_hist.empty:
        st.info("No version history yet.")
        return

    for base_tbl in sorted(df_hist['base_table'].unique()):
        # Filter to versions for the current base table
        tbl_versions = df_hist[df_hist.base_table == base_tbl].reset_index()
        
        with st.expander(f"📂 **{base_tbl}** ({len(tbl_versions)} versions)", expanded=True):
            
            # --- NEW: Schema Comparison Section ---
            st.markdown("---")
            st.subheader("🧬 Compare Schemas")
            
            # Prevent error if there's only one version
            if len(tbl_versions) > 1:
                col1, col2 = st.columns(2)
                # Create a dictionary mapping snapshot_table to its index for easy lookup
                version_options = {row.snapshot_table: index for index, row in tbl_versions.iterrows()}
                
                with col1:
                    base_choice = st.selectbox("Base Version", options=version_options.keys(), key=f"base_{base_tbl}")
                with col2:
                    # Default the second box to the next item in the list
                    compare_choice = st.selectbox("Compare With", options=version_options.keys(), index=1, key=f"compare_{base_tbl}")

                if st.button("Compare Schemas", key=f"diff_btn_{base_tbl}"):
                    if base_choice == compare_choice:
                        st.warning("Please select two different versions to compare.")
                    else:
                        # Get metadata for the chosen versions using the lookup dictionary
                        base_meta = tbl_versions.iloc[version_options[base_choice]]
                        compare_meta = tbl_versions.iloc[version_options[compare_choice]]
                        
                        # Call the new diff function from dataframe_utils
                        diff = dataframe_utils.get_schema_diff(
                            json.loads(base_meta['columns']), json.loads(base_meta['rename_map']),
                            json.loads(compare_meta['columns']), json.loads(compare_meta['rename_map'])
                        )
                        
                        st.markdown("##### Schema Changes")
                        res_col1, res_col2 = st.columns(2)
                        with res_col1:
                            st.markdown(f"**Columns Added ({len(diff['added'])}):**")
                            if diff['added']:
                                st.dataframe(pd.DataFrame(diff['added'], columns=["Column Name"]), use_container_width=True, hide_index=True)
                            else:
                                st.info("No columns added.")
                        with res_col2:
                            st.markdown(f"**Columns Removed ({len(diff['removed'])}):**")
                            if diff['removed']:
                                st.dataframe(pd.DataFrame(diff['removed'], columns=["Column Name"]), use_container_width=True, hide_index=True)
                            else:
                                st.info("No columns removed.")
            else:
                st.info("You need at least two versions of this table to compare schemas.")
            
            # --- End of Schema Comparison Section ---

            st.markdown("---")
            st.subheader("Individual Snapshots")
            
            h_cols = st.columns((4, 3, 2))
            h_cols[0].markdown("**Snapshot Table**")
            h_cols[1].markdown("**Timestamp (Relative)**")
            h_cols[2].markdown("**Actions**")
            
            for index, row in tbl_versions.iterrows():
                cols = st.columns((4, 3, 2))
                cols[0].code(row.snapshot_table)
                try:
                    ts = datetime.strptime(row.timestamp, "%Y%m%d%H%M%S%f")
                    cols[1].write(humanize.naturaltime(datetime.now() - ts))
                except (ValueError, TypeError):
                    cols[1].write(str(row.timestamp))

                action_col = cols[2]
                if action_col.button("🌱 Restore", key=f"restore_{row.snapshot_table}", use_container_width=True):
                    db_utils.restore_snapshot(conn, row.snapshot_table, base_tbl)
                    st.session_state.toast_message = f"Restored `{base_tbl}` from snapshot."
                    st.rerun()

                if action_col.button("🗑️ Delete", key=f"delete_{row.snapshot_table}", use_container_width=True):
                    dialogs.confirm_delete_snapshot(conn, row.snapshot_table)

                if st.toggle("Show metadata", key=f"meta_{row.snapshot_table}"):
                    st.write("**Columns used:**", json.loads(row['columns']))
                    st.write("**Column rename map:**", json.loads(row['rename_map']))