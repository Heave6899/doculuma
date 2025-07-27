# db_utils.py
import json
from pathlib import Path
import sqlite3
from datetime import datetime
from typing import Dict, List
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
import pandas as pd
import streamlit as st
import re
import config
import math 
import numpy as np
import sqlite_vec
import requests
from huggingface_hub import configure_http_backend
import chromadb
from chromadb.utils import embedding_functions

from file_processors import ensure_stable_id

CHROMA_PERSIST_DIR = Path("./chroma_db_data")
CHROMA_PERSIST_DIR.mkdir(exist_ok=True)

def backend_factory() -> requests.Session:
    session = requests.Session()
    session.verify = False
    return session

configure_http_backend(backend_factory=backend_factory)
# Load a pre-trained model (small & fast, good for field names)
@st.cache_resource
def get_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        normalize_embeddings=True
    )

def cosine_similarity_json(json1, json2):
    if not json1 or not json2:
        return 0.0
    v1 = np.array(json.loads(json1))
    v2 = np.array(json.loads(json2))
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    return float(dot / (norm1 * norm2)) if norm1 and norm2 else 0.0

def reverse_string(s): return s[::-1]

def semantic_match(json_path, input_key_dot):
    """
    Returns a semantic similarity score (0–100) between input_key_dot and json_path.
    """
    if not json_path or not input_key_dot:
        return 0
    return fuzz.token_set_ratio(str(json_path), str(input_key_dot))

def max_substr_match(json_path, input_key_dot):
    """
    Returns the length of the matching substring if input_key_dot is found in json_path,
    else returns 0.
    """
    if not json_path or not input_key_dot:
        return 0
    if input_key_dot in json_path:
        return len(input_key_dot)
    return 0

def segments_in_order(key, path):
    try:
        if key is None or path is None:
            print(f"[SKIP: Null] key={key}, path={path}")
            return 0
        if isinstance(key, float) and math.isnan(key):
            print(f"[SKIP: NaN] key={key}, path={path}")
            return 0
        if isinstance(path, float) and math.isnan(path):
            print(f"[SKIP: NaN] key={key}, path={path}")
            return 0
        key = str(key)
        path = str(path)
        key_no_underscore = key.replace('_', '')
        segments = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?![a-z])', key_no_underscore)
        start = 0
        for segment in segments:
            if not segment:
                continue
            idx = path.find(segment, start)
            if idx == -1:
                # print(f"[NO MATCH] key={key}, path={path} -- missing segment '{segment}'")
                return 0
            start = idx + len(segment)
        print(f"[MATCH] key={key}, path={path} -- segments: {segments}")
        return 1
    except Exception as e:
        print(f"[ERROR] key={key}, path={path}, exc={e}")
        return 0

@st.cache_resource
def get_chroma_client():
    return chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
# Use st.cache_resource for caching the database connection
@st.cache_resource
def get_connection(db_path: str) -> sqlite3.Connection:
    """Initialize and return a cached SQLite connection, creating tables if they don't exist."""
    try:
        conn = sqlite3.connect(db_path, check_same_thread=False)
        conn.enable_load_extension(True)
        conn.execute('SELECT load_extension("extensions/fuzzy")')
        conn.execute('SELECT load_extension("extensions/regexp")')
        conn.create_function('segments_in_order', 2, segments_in_order)
        conn.create_function("max_substr_match", 2, max_substr_match)
        conn.create_function("semantic_match", 2, max_substr_match)
        conn.create_function("REVERSE", 1, reverse_string)
        conn.create_function("cosine_sim", 2, cosine_similarity_json)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        # Use a single function to ensure all tables are created
        _initialize_tables(conn)
        return conn
    except Exception as e:
        config.logger.error(f"Failed to initialize database: {e}")
        st.error(f"Database initialization error: {e}")
        raise

def _initialize_tables(conn: sqlite3.Connection):
    """Creates the necessary 'versions' and 'saved_queries' tables."""
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS versions (
                version_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                base_table     TEXT NOT NULL,
                snapshot_table TEXT NOT NULL,
                timestamp      TEXT NOT NULL,
                header_row     INTEGER,
                columns        TEXT,
                rename_map     TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                base_table TEXT NOT NULL,
                sql_text TEXT NOT NULL,
                saved_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS saved_pipelines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE,
                definition TEXT NOT NULL,
                saved_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
        """)
@st.cache_data
def list_tables(_conn: sqlite3.Connection, last_mod_time: float) -> List[str]:
    """Return a sorted list of all tables/views, cached based on DB modification time."""
    cur = _conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type IN ('table','view');")
    return sorted(r[0] for r in cur.fetchall())

def persist_version(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    base_tbl: str,
    header_row: int,
    keep_cols: List[str],
    rename_map: Dict[str, str],
) -> str:
    """Saves a new snapshot, overwrites the base table, and returns the snapshot table name."""
    df = ensure_stable_id(df)
    ts = datetime.now().strftime("%Y%m%d%H%M%S%f")
    snap_tbl = f"{base_tbl}_v{ts}"
    try:
        with conn:
            conn.execute(
                "INSERT INTO versions (base_table, snapshot_table, timestamp, header_row, columns, rename_map) VALUES (?, ?, ?, ?, ?, ?)",
                (base_tbl, snap_tbl, ts, header_row, json.dumps(keep_cols), json.dumps(rename_map)),
            )
            df.to_sql(snap_tbl, conn, if_exists="replace", index=False)
            df.to_sql(base_tbl, conn, if_exists="replace", index=False)
        prune_snapshots(conn, base_tbl)
        config.logger.info(f"Persisted {base_tbl} -> {snap_tbl}")
        return snap_tbl
    except Exception as e:
        config.logger.error(f"Error persisting version for {base_tbl}: {e}")
        st.error(f"Failed to save version: {e}")
        raise

def prune_snapshots(conn: sqlite3.Connection, base_tbl: str):
    """Keeps only the most recent 'PRUNE_KEEP' snapshots for a base table."""
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT snapshot_table FROM versions WHERE base_table = ? ORDER BY timestamp DESC LIMIT -1 OFFSET ?",
            (base_tbl, config.PRUNE_KEEP),
        )
        old_snaps = [r[0] for r in cur.fetchall()]
        for snap in old_snaps:
            with conn:
                conn.execute(f"DROP TABLE IF EXISTS `{snap}`;")
                conn.execute("DELETE FROM versions WHERE snapshot_table = ?", (snap,))
            config.logger.info(f"Pruned old snapshot {snap}")
    except Exception as e:
        config.logger.warning(f"Error pruning snapshots for {base_tbl}: {e}")

def restore_snapshot(conn: sqlite3.Connection, snapshot_table: str, base_table: str):
    """Restores a snapshot to become the new version of the base table."""
    try:
        df_restore = pd.read_sql_query(f"SELECT * FROM `{snapshot_table}`", conn)
        meta = conn.execute(
            "SELECT header_row, columns, rename_map FROM versions WHERE snapshot_table = ? ORDER BY timestamp DESC LIMIT 1",
            (snapshot_table,)
        ).fetchone()
        
        header_row, keep_cols, rename_map = (
            (meta[0], json.loads(meta[1]), json.loads(meta[2])) if meta
            else (0, df_restore.columns.tolist(), {c: c for c in df_restore.columns})
        )
        
        new_snap = persist_version(conn, df_restore, base_table, header_row, keep_cols, rename_map)
        config.logger.info(f"Restored {base_table} from {snapshot_table} (new snapshot: {new_snap})")
        list_tables.clear() # Invalidate cache
        st.session_state.toast_message = f"Restored `{base_table}` from `{snapshot_table}`."
    except Exception as e:
        config.logger.error(f"Failed to restore {base_table} from {snapshot_table}: {e}")
        st.error(f"Restore failed: {e}")

def drop_table(conn: sqlite3.Connection, base_tbl: str):
    """Drops the base table and all its associated snapshots and version history."""
    try:
        with conn:
            snapshots_to_drop = conn.execute("SELECT snapshot_table FROM versions WHERE base_table = ?", (base_tbl,)).fetchall()
            for (snap,) in snapshots_to_drop:
                conn.execute(f"DROP TABLE IF EXISTS `{snap}`;")
            conn.execute("DELETE FROM versions WHERE base_table = ?", (base_tbl,))
            conn.execute(f"DROP TABLE IF EXISTS `{base_tbl}`;")
        
        chroma_client = get_chroma_client()
        collection_name = f"{config.CHROMA_COLLECTION_PREFIX}_{base_tbl}"
        try:
            chroma_client.delete_collection(collection_name)
            config.logger.info(f"Deleted ChromaDB collection `{collection_name}`")
        except Exception as e:
            config.logger.error(f"Error deleting ChromaDB collection `{collection_name}`: {e}")
            
        list_tables.clear()
        st.session_state.toast_message = f"Deleted table `{base_tbl}` and all its versions."
    except Exception as e:
        st.error(f"Error deleting `{base_tbl}`: {e}")

def drop_snapshot(conn: sqlite3.Connection, snap_tbl: str):
    """Drops a single snapshot table and its entry from the versions table."""
    try:
        with conn:
            conn.execute(f"DROP TABLE IF EXISTS `{snap_tbl}`;")
            conn.execute("DELETE FROM versions WHERE snapshot_table = ?", (snap_tbl,))
        list_tables.clear()
        st.session_state.toast_message = f"Deleted snapshot `{snap_tbl}`."
    except Exception as e:
        st.error(f"Error deleting snapshot `{snap_tbl}`: {e}")

def semantic_search_chroma(chroma_client, query, collection_prefix, top_k=100):
    collection_list = chroma_client.list_collections()
    matched_results = {}

    for collection in collection_list:
        if collection.name.startswith(collection_prefix):
            try:
                results = collection.query(
                    query_texts=[query],
                    n_results=top_k
                )
                matches = []
                for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                    similarity = 1 - dist  # ChromaDB distance to similarity
                    matches.append({
                        "text": doc,
                        "metadata": meta,
                        "similarity": similarity
                    })
                if matches:
                    table_name = collection.name.replace(f"{collection_prefix}_", "")
                    matched_results[table_name] = matches
            except Exception as e:
                config.logger.error(f"Error querying ChromaDB collection '{collection.name}': {e}")
                continue  # Ensure robustness even if one collection fails
    return matched_results      

def delete_embeddings_from_chroma(chroma_client, table_name, row_ids):
    collection_name = f"{config.CHROMA_COLLECTION_PREFIX}_{table_name}"
    try:
        collection = chroma_client.get_collection(collection_name)
        ids_to_delete = [f"{table_name}_{row_id}" for row_id in row_ids]
        collection.delete(ids=ids_to_delete)
        config.logger.info(f"Deleted embeddings for rows {row_ids} from {collection_name}")
    except Exception as e:
        config.logger.error(f"Error deleting embeddings from ChromaDB: {e}")