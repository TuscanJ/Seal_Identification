import json
import os
import uuid
from datetime import datetime
from pathlib import Path

QUEUE_DIR = Path("app/training_queue")
QUEUE_MANIFEST = QUEUE_DIR / "queue.json"

def load_queue():
    """
    Desc: Reads the training queue manifest
    arguments: None
    returns: list of queue entries (empty if no manifest yet)
    """
    if not QUEUE_MANIFEST.exists():
        return []
    with open(QUEUE_MANIFEST, encoding="utf-8") as f:
        return json.load(f)

def save_queue(queue):
    """
    Desc: Writes the training queue manifest
    arguments: queue- list of queue entries
    returns: None
    """
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_MANIFEST, "w", encoding="utf-8") as f:
        json.dump(queue, f, indent=4)


#Probably don't need this for now.
#    def next_available_id():
#       """
#        Desc: Gets the id a new seal should be given
#        arguments: None
#        returns: highest id in use (seal_imgs folders + queue) plus one
#        """
#        #TODO
#        return 0

def queue_for_training(cropped_image, seal_name, source_filename, seal_class="new"):
    """
    Desc: Saves the image and adds the seal to the training queue
    arguments: cropped_image- confirmed crop of the uploaded image
               seal_name- name given to the seal
               source_filename- name of the uploaded file
               seal_class- "new" for a new seal, "old" for a new image of a known seal
    returns: the queue entry that was added
    """
    entry = {
        "id": uuid.uuid4().hex,
        "seal_name": seal_name,
        "class": seal_class,
        "image_path": None,  #TODO: QUEUE_DIR/{id}/{source_filename}
        "source_filename": source_filename,
        "queued_at": datetime.now().isoformat(),
    }
    #TODO: save cropped_image to entry["image_path"]
    #TODO: queue = load_queue(); queue.append(entry); save_queue(queue)
    return entry
