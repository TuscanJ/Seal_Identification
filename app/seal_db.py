import os
import sqlite3
from pathlib import Path

import streamlit as st

#Seal photo database. Defaults to seals.db next to this file; on the server, set SEAL_DB_PATH to the shared copy.
SEAL_DB_PATH = Path(os.environ.get("SEAL_DB_PATH", Path(__file__).parent / "seals.db")).expanduser()

SIDE_NAMES = {"LS": "Left side", "F": "Front", "RS": "Right side"}

@st.cache_data
def load_seal_photos(seal_id):
    """
    Desc: Gets every photo of one seal from the database
    arguments: seal_id- seal's ID as stored in the database, e.g. "0101"
    returns: list of (image bytes, side, date taken), in photo number order
    """
    con = sqlite3.connect(f"file:{SEAL_DB_PATH}?mode=ro", uri=True)
    try:
        return con.execute(
            "SELECT image, side, date_taken FROM photos WHERE seal_id = ? ORDER BY photo_num",
            (seal_id,),
        ).fetchall()
    finally:
        con.close()
