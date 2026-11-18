import hashlib
import io
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import streamlit as st
from PIL import Image

#Seal photo database. On the server, set SEAL_DB_PATH to the shared copy. Otherwise uses the full seals.db next to
#this file if you have it, or the small seals_dev.db from git (3 photos per seal).
_FULL_DB = Path(__file__).parent / "seals.db"
_DEV_DB = Path(__file__).parent / "seals_dev.db"
SEAL_DB_PATH = Path(os.environ.get("SEAL_DB_PATH") or (_FULL_DB if _FULL_DB.exists() else _DEV_DB)).expanduser()
USING_DEV_DB = SEAL_DB_PATH.resolve() == _DEV_DB.resolve()

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

def exif_date_taken(image):
    """
    Desc: Reads when a photo was taken from its EXIF data
    arguments: image- PIL image as opened from the uploaded file (before cropping/converting)
    returns: date as "YYYY-MM-DD HH:MM:SS", or None if the photo has no usable date
    """
    exif = image.getexif()
    raw = exif.get_ifd(0x8769).get(36867) or exif.get(306)  #DateTimeOriginal, then DateTime
    try:
        return datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S").strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None

def guess_side(filename):
    """
    Desc: Guesses which side of the seal a photo shows from its filename
    arguments: filename- uploaded file's name, e.g. "0101_20211118_14.27left 1.jpeg"
    returns: "LS", "RS", "F", or None if the name doesn't say
    """
    name = filename.lower()
    if "left" in name:
        return "LS"
    if "right" in name:
        return "RS"
    if "front" in name:
        return "F"
    return None

def add_seal_photo(seal_id, image, side, date_taken, source_file):
    """
    Desc: Saves a labelled photo to the database as the seal's next photo
    arguments: seal_id- seal's ID as stored in the database, e.g. "0101"
               image- PIL image to save (the confirmed crop)
               side- "LS", "F", "RS", or None if unknown
               date_taken- "YYYY-MM-DD HH:MM:SS", or None if unknown
               source_file- name of the uploaded file
    returns: the new photo number, or None if this exact image is already in the database
    """
    buffer = io.BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=95)
    image_bytes = buffer.getvalue()
    sha256 = hashlib.sha256(image_bytes).hexdigest()

    #Same format the import used, e.g. "0039_20211129_1216"
    encounter = None
    if date_taken:
        taken = datetime.strptime(date_taken, "%Y-%m-%d %H:%M:%S")
        encounter = f"{seal_id}_{taken:%Y%m%d_%H%M}"

    #mode=rw so a wrong SEAL_DB_PATH errors instead of creating an empty database
    con = sqlite3.connect(f"file:{SEAL_DB_PATH}?mode=rw", uri=True, timeout=30, isolation_level=None)
    try:
        #Lock before reading the next photo number so two users saving at once can't get the same one
        con.execute("BEGIN IMMEDIATE")
        if con.execute("SELECT 1 FROM photos WHERE sha256 = ?", (sha256,)).fetchone():
            con.execute("ROLLBACK")
            return None
        con.execute("INSERT OR IGNORE INTO seals (seal_id) VALUES (?)", (seal_id,))
        photo_num = con.execute(
            "SELECT COALESCE(MAX(photo_num), 0) + 1 FROM photos WHERE seal_id = ?", (seal_id,)
        ).fetchone()[0]
        con.execute(
            "INSERT INTO photos (seal_id, photo_num, date_taken, side, image, source, source_file, encounter, sha256) "
            "VALUES (?, ?, ?, ?, ?, 'app_upload', ?, ?, ?)",
            (seal_id, photo_num, date_taken, side, image_bytes, source_file, encounter, sha256),
        )
        con.execute("COMMIT")
    except BaseException:
        if con.in_transaction:
            con.execute("ROLLBACK")
        raise
    finally:
        con.close()

    #The comparison view caches each seal's photos, so drop the cache to show the new one
    load_seal_photos.clear()
    return photo_num
