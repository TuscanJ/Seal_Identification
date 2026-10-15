from pathlib import Path

#Change this later?
import streamlit as st
from PIL import Image

from seal_queue import queue_for_training

SEAL_IMGS_DIR = Path("app/seal_imgs") #Change this in Sprint 2 + add option for windows and linux
NEW_SEAL_CONF_THRESHOLD = 0.75

STAGE_COMPARE = "compare"
STAGE_NAME = "name"
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

def reset_new_seal():
    """
    Desc: Closes the new seal flow (call when changing image or picking a prediction)
    arguments: None
    returns: None
    """
    st.session_state.new_seal_stage = None
    st.session_state.new_seal_entry = None

def render_new_seal_flow(cropped_image, predictions, source_filename, current_index):
    """
    Desc: Draws whichever step of the new seal flow is active (not called from app.py yet)
    arguments: cropped_image- confirmed crop of the uploaded image
               predictions- top 3 predictions from predict_image
               source_filename- name of the uploaded file
               current_index- index of the uploaded image, used for widget keys
    returns: None
    """
    stage = st.session_state.new_seal_stage

    if stage == STAGE_COMPARE:
        render_new_seal_compare(cropped_image, predictions)

    elif stage == STAGE_NAME:
        seal_name = render_name_bar(current_index)
        if seal_name:
            st.session_state.new_seal_entry = queue_for_training(cropped_image, seal_name, source_filename)
            st.session_state.new_seal_stage = STAGE_QUEUED
            st.rerun()

    elif stage == STAGE_QUEUED:
        entry = st.session_state.new_seal_entry
        st.success(f"Seal {entry['seal_name']} queued for training")


# Part 1: new seal button

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
    #TODO: open the new seal flow (set new_seal_stage, set compare back to 0)
    pass

def render_new_seal_button(predictions):
    """
    Desc: Draws the new seal button under the prediction buttons
    arguments: predictions- top 3 predictions from predict_image
    returns: None
    """
    #TODO: use new_seal_popup(predictions[0]['score']) to highlight the button
    st.button("🆕 New Seal", on_click=start_new_seal)


# Part 2: most likely seals next to the uploaded image

def get_candidate_seals(cropped_image, predictions):
    """
    Desc: Finds the known seals that look most like the uploaded image
    arguments: cropped_image- confirmed crop of the uploaded image
               predictions- top 3 predictions from predict_image
    returns: list of (seal label, path to a reference image)
    """
    #TODO
    return []

def confirm_new_seal():
    st.session_state.new_seal_stage = STAGE_NAME

def render_new_seal_compare(cropped_image, predictions):
    """
    Desc: Shows the most likely seals next to the uploaded image so the user can confirm it is new
    arguments: cropped_image- confirmed crop of the uploaded image
               predictions- top 3 predictions from predict_image
    returns: None
    """
    #TODO: show cropped_image next to get_candidate_seals(), with confirm (confirm_new_seal) and cancel (reset_new_seal) buttons
    pass


# Part 3: name bar

def validate_seal_name(seal_name):
    """
    Desc: Checks a new seal name before it is queued
    arguments: seal_name- name typed by the user
    returns: error message, or None if the name is usable
    """
    #TODO: reject empty names
    #TODO: reject names already in SEAL_IMGS_DIR or already in the queue
    #TODO: reject characters that are unsafe in a folder name
    return None

def render_name_bar(current_index):
    """
    Desc: Draws the name input for a confirmed new seal
    arguments: current_index- index of the uploaded image, used for widget keys
    returns: the submitted name, or None if nothing valid was submitted this run
    """
    #TODO: text input + submit button, run validate_seal_name, return the name
    return None

