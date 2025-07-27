# dialogs.py
import streamlit as st
import db_utils

@st.dialog("Delete Table", width="small")
def confirm_delete_table(conn, tbl_to_delete: str):
    """Shows a confirmation dialog before deleting a table and its versions."""
    st.write(
        f"You are about to permanently delete the table `{tbl_to_delete}` and all of its associated snapshots."
    )
    st.warning("This action cannot be undone.")
    if st.button("Confirm Deletion"):
        db_utils.drop_table(conn, tbl_to_delete)
        st.rerun()

@st.dialog("Delete Snapshot", width="small")
def confirm_delete_snapshot(conn, snap_to_delete: str):
    """Shows a confirmation dialog before deleting a single snapshot."""
    st.write(
        f"You are about to permanently delete the snapshot `{snap_to_delete}`."
    )
    st.warning("This action cannot be undone.")
    if st.button("Confirm Deletion"):
        db_utils.drop_snapshot(conn, snap_to_delete)
        st.rerun()