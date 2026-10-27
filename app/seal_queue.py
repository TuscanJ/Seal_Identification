import json
import os
import uuid
from datetime import datetime
from pathlib import Path

#Shared queue location. Defaults to a folder next to this file; on the server, set SEAL_QUEUE_DIR to the shared research folder.
QUEUE_DIR = Path(os.environ.get("SEAL_QUEUE_DIR", Path(__file__).parent / "training_queue")).expanduser()

#Let everyone in the research group read, move and delete queued files, not just whoever created them.
os.umask(0o002)

def load_queue():
    """
    Desc: Reads every entry in the training queue
    arguments: None
    returns: list of queue entries, oldest first (empty if the queue folder does not exist)
    """
    if not QUEUE_DIR.exists():
        return []
    entries = []
    for p in QUEUE_DIR.glob("*/*.json"):
        try:
            entries.append(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            #Another user may be moving or writing this entry right now, skip it this time
            continue
    return sorted(entries, key=lambda e: e["queued_at"])

def save_entry(entry):
    """
    Desc: Writes one queue entry next to its image
    arguments: entry- queue entry from queue_for_training
    returns: None
    """
    entry_path = QUEUE_DIR / entry["seal_name"] / f"{entry['id']}.json"
    #Write to a temp file and rename so other users never read a half-written entry
    tmp_path = entry_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(entry, f, indent=4)
    os.replace(tmp_path, entry_path)

def save_image(image, image_path):
    """
    Desc: Saves a queued image
    arguments: image- PIL image to save
               image_path- path relative to QUEUE_DIR
    returns: None
    """
    full_path = QUEUE_DIR / image_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(full_path)

def check_seal_name(seal_name):
    """
    Desc: Stops a seal name from writing outside QUEUE_DIR
    arguments: seal_name- name given to the seal
    returns: None, raises ValueError if the name is not a single folder name
    """
    if not seal_name or seal_name in (".", "..") or "/" in seal_name or "\\" in seal_name:
        raise ValueError(f"Invalid seal name: {seal_name!r}")

def queue_for_training(cropped_image, seal_name, source_filename, seal_class="new"):
    """
    Desc: Saves the image and adds the seal to the training queue
    arguments: cropped_image- confirmed crop of the uploaded image
               seal_name- name given to the seal
               source_filename- name of the uploaded file
               seal_class- "new" for a new seal, "old" for a new image of a known seal
    returns: the queue entry that was added
    """
    check_seal_name(seal_name)
    entry_id = uuid.uuid4().hex
    entry = {
        "id": entry_id,
        "seal_name": seal_name,
        "class": seal_class,
        "image_path": f"{seal_name}/{entry_id}.jpg",
        "source_filename": source_filename,
        "queued_at": datetime.now().isoformat(),
    }
    save_image(cropped_image, entry["image_path"])
    save_entry(entry)
    return entry
