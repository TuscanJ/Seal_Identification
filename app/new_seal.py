from pathlib import Path

#Change this later?
import streamlit as st
from PIL import Image

from seal_queue import QUEUE_DIR, load_queue, next_seal_number, queue_new_seal

SEAL_IMGS_DIR = Path("app/seal_imgs") #Change this in Sprint 2 + add option for windows and linux
NEW_SEAL_CONF_THRESHOLD = 0.75

STAGE_CONFIRM = "confirm"
STAGE_QUEUED = "queued"

def init_new_seal_state():
    """
    Desc: Sets up session state used by the new seal flow
    arguments: None
    returns: None
    """
    if "new_seal_stage" not in st.session_state:
        st.session_state.new_seal_stage = None
    if "new_seal_entry" not in st.session_state:
        st.session_state.new_seal_entry = None
    if "show_queued_seals" not in st.session_state:
        st.session_state.show_queued_seals = False

def reset_new_seal():
    """
    Desc: Closes the new seal flow (call when changing image or picking a prediction)
    arguments: None
    returns: None
    """
    st.session_state.new_seal_stage = None
    st.session_state.new_seal_entry = None

def toggle_queued_seals():
    """Toggle the queued seals list open or closed."""
    st.session_state.show_queued_seals = not st.session_state.show_queued_seals

def render_queued_seals():
    """Display the queued seal entries and their images."""
    entries = load_queue()
    st.markdown("### 📋 Queued Seals")

    if not entries:
        st.info("There are no seals in the training queue yet.")
        return

    for entry in entries:
        seal_name = entry.get("seal_name", "Unknown")
        image_relative_path = entry.get("image_path", "")
        image_path = QUEUE_DIR / image_relative_path

        st.markdown(f"**Seal {seal_name}** · {entry.get('class', 'unknown')}")
        st.caption(
            f"Queued: {entry.get('queued_at', 'Unknown date')} · "
            f"Source file: {entry.get('source_filename', 'Unknown file')}"
        )

        # Ensure an entry cannot use a crafted image path outside the queue folder.
        try:
            resolved_queue_dir = QUEUE_DIR.resolve()
            resolved_image_path = image_path.resolve()
            resolved_image_path.relative_to(resolved_queue_dir)
        except (OSError, ValueError):
            st.warning("The queued image path is invalid.")
            continue

        if resolved_image_path.is_file():
            st.image(str(resolved_image_path), caption=f"Seal {seal_name}")
        else:
            st.warning("The image for this queue entry could not be found.")

def render_new_seal_flow(cropped_image, source_filename, known_seals):
    """
    Desc: Draws whichever step of the new seal flow is active
    arguments: cropped_image- confirmed crop of the uploaded image
               source_filename- name of the uploaded file
               known_seals- seal numbers the model already knows, as strings
    returns: None
    """
    stage = st.session_state.new_seal_stage

    if stage == STAGE_CONFIRM:
        st.markdown("### 🆕 New Seal")
        st.write(f"The next available subject number is **{next_seal_number(known_seals)}**. Are you sure this is a new seal?")
        col1, col2 = st.columns(2)
        with col1:
            confirmed = st.button("✅ Yes, add as a new seal")
        with col2:
            st.button("❌ Cancel", on_click=reset_new_seal)
        if confirmed:
            st.session_state.new_seal_entry = queue_new_seal(cropped_image, source_filename, known_seals)
            st.session_state.new_seal_stage = STAGE_QUEUED
            st.rerun()

    elif stage == STAGE_QUEUED:
        entry = st.session_state.new_seal_entry
        st.success(f"Image queued for training as new seal {entry['seal_name']}")


# New seal button

def new_seal_popup(confidence_score):
    """
    Desc: Decides whether the top prediction is weak enough to suggest a new seal
    arguments: confidence_score- top prediction score, 0 to 1
    returns: True if the new seal button should be highlighted
    """
    #TODO: return confidence_score < NEW_SEAL_CONF_THRESHOLD
    return False

def start_new_seal():
    """
    Desc: Runs when the new seal button is clicked
    arguments: None
    returns: None
    """
    st.session_state.new_seal_stage = STAGE_CONFIRM
    st.session_state.compare = 0  # close the side by side comparison

def render_new_seal_button(predictions):
    """
    Desc: Draws the new seal button under the prediction buttons
    arguments: predictions- top 3 predictions from predict_image
    returns: None
    """
    button_label = (
        "Hide Queued Seals"
        if st.session_state.show_queued_seals
        else "Show Queued Seals"
    )
    st.button(button_label, on_click=toggle_queued_seals)

    if st.session_state.show_queued_seals:
        render_queued_seals()

    #TODO: use new_seal_popup(predictions[0]['score']) to highlight the button
    st.button("🆕 New Seal", on_click=start_new_seal)
