import json
import os
import uuid
from pathlib import Path

#Shared queue location. Defaults to a folder next to this file; on the server, set SEAL_QUEUE_DIR to the shared research folder.
QUEUE_DIR = Path(os.environ.get("SEAL_QUEUE_DIR", Path(__file__).parent / "training_queue")).expanduser()

#Let everyone in the research group read, move and delete queued files, not just whoever created them.
os.umask(0o002)

def load_queue():
    """
    Desc: Reads every entry in the training queue
    arguments: None
    returns: list of queue entries (empty if the queue folder does not exist)
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
    return entries

def save_entry(entry):
    """
    Desc: Writes one queue entry next to its image
    arguments: entry- queue entry from queue_for_training
    returns: None
    """
    entry_path = QUEUE_DIR / entry["seal_number"] / f"{entry['id']}.json"
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

def check_seal_number(seal_number):
    """
    Desc: Makes sure a seal number is only digits, so it can't write outside QUEUE_DIR
    arguments: seal_number- seal's subject number, as a string
    returns: None, raises ValueError if the number is not only digits
    """
    if not seal_number.isdigit():
        raise ValueError(f"Invalid seal number: {seal_number!r}")

def queue_for_training(cropped_image, seal_number, source_filename, seal_class="new"):
    """
    Desc: Saves the image and adds the seal to the training queue
    arguments: cropped_image- confirmed crop of the uploaded image
               seal_number- seal's subject number, as a string
               source_filename- name of the uploaded file
               seal_class- "new" for a new seal, "old" for a new image of a known seal
    returns: the queue entry that was added
    """
    check_seal_number(seal_number)
    entry_id = uuid.uuid4().hex
    entry = {
        "id": entry_id,
        "seal_number": seal_number,
        "class": seal_class,
        "image_path": f"{seal_number}/{entry_id}.jpg",
        "source_filename": source_filename,
    }
    save_image(cropped_image, entry["image_path"])
    save_entry(entry)
    return entry

def next_seal_number(known_seals):
    """
    Desc: Finds the next unused seal number, counting known seals and seals already in the queue
    arguments: known_seals- seal numbers the model already knows, as strings
    returns: next available seal number, as a string
    """
    queued = [p.name for p in QUEUE_DIR.iterdir() if p.is_dir()] if QUEUE_DIR.exists() else []
    numbers = [int(name) for name in list(known_seals) + queued if name.isdigit()]
    return str(max(numbers, default=0) + 1)

def queue_new_seal(cropped_image, source_filename, known_seals):
    """
    Desc: Gives a new seal the next available number and queues its image
    arguments: cropped_image- confirmed crop of the uploaded image
               source_filename- name of the uploaded file
               known_seals- seal numbers the model already knows, as strings
    returns: the queue entry that was added
    """
    number = int(next_seal_number(known_seals))
    #Claim the number by creating its folder, so two users adding new seals at once can't get the same one
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            (QUEUE_DIR / str(number)).mkdir()
            break
        except FileExistsError:
            number += 1
    return queue_for_training(cropped_image, str(number), source_filename, seal_class="new")
