import streamlit as st
import gdown
from PIL import Image
from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
import torch
from streamlit_cropper import st_cropper
from transformers import OwlViTProcessor, OwlViTForObjectDetection
import atexit
import tempfile
import shutil
import zipfile
from pathlib import Path
import os
import sqlite3

from new_seal import init_new_seal_state, render_new_seal_button, render_new_seal_flow, reset_new_seal
from seal_db import SIDE_NAMES, USING_DEV_DB, add_seal_photo, exif_date_taken, guess_side, load_seal_photos
from seal_queue import queue_for_training, queued_seal_images, queued_seal_numbers

os.environ["STREAMLIT_WATCHER_TYPE"] = "none"


# Setup constants
if not os.path.exists("VIT"):
    gdown.download_folder("https://drive.google.com/drive/folders/1rPSmx28DGFKN-lAN2gTosglsZigJoSzZ?usp=drive_link", output="VIT", quiet=False, use_cookies=False)

MODEL_DIR = "VIT"
PROCESSOR_DIR = "VIT/saved_processor"

OWL_PROCESSOR = "VIT/owlvit_processor"
OWL_MODEL = "VIT/owlvit_model"

if "saved_crops_dir" not in st.session_state:
    st.session_state.saved_crops_dir = tempfile.mkdtemp()
if "saved_files" not in st.session_state:
    st.session_state.saved_files = []


@st.cache_resource
def load_owlvit():
    """
    Desc: Loads OwlVIT
    arguments: None
    returns: processor, model
    """
    processor = OwlViTProcessor.from_pretrained(OWL_PROCESSOR)
    model = OwlViTForObjectDetection.from_pretrained(OWL_MODEL)
    return processor, model

@st.cache_data
def get_recommended_crop(image: Image.Image, text="harbor seal's head"):
    """
    Desc: Uses OwlVIT to get coordinates of a box around seals head
    arguments: image- image to predict on
               text- what OwlVIT should look for
    returns: coordinates for OwlVIT recommended box
    """
    processor, model = load_owlvit()
    inputs = processor(text=[[text]], images=[image], return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    target_size = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, threshold=0.1, target_sizes=target_size)
    boxes = results[0]['boxes']
    if boxes.nelement() == 0:
        return None
    best_box = boxes[torch.argmax(results[0]['scores'])].tolist()
    return [round(x, 2) for x in best_box]  # x1, y1, x2, y2

@st.cache_resource
def load_model_and_processor():
    """
    Desc: Loads the model and processor for ViTForImageClassification and label/id configuration
    arguments: None
    returns: model-VitForImageClassification model
             processor- VirForImageClassification processor
             id2label- label/id pairs for the model
    """
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    processor = ViTImageProcessor.from_pretrained(PROCESSOR_DIR)

    id2label = model.config.id2label

    return model, processor, id2label

@st.cache_resource
def create_classifier(_model, _processor):
    """
    Desc: creates classifier object
    arguments: model- model to use for classification
               processor- processor to use for classification
    returns: classifier- image classification predictor
    """
    classifier = pipeline(
        "image-classification",
        model=_model,
        feature_extractor=_processor,
        device=0 if torch.cuda.is_available() else -1
    )
    return classifier

@st.cache_data
def predict_image(_classifier, image):
    """
    Desc: predicts an image using the given classifier
    arguments: _classifier- classifier to be used on image
               image- image to be classified
    returns: top 10 most likely classes for image
    """
    return _classifier(image, top_k=10)

if 'compare' not in st.session_state:
    st.session_state.compare = 0
if 'selected_img' not in st.session_state:
    st.session_state.selected_img = 0
if 'selected_queued' not in st.session_state:
    st.session_state.selected_queued = None
if 'show_all_queued' not in st.session_state:
    st.session_state.show_all_queued = False

init_new_seal_state()

def reset_selected_img():
    st.session_state.selected_img = 0

def compare(i):
    st.session_state.compare = i
    st.session_state.selected_queued = None  #Only one comparison open at a time
    reset_selected_img()
    reset_new_seal()

def show_queued_seal(seal_number):
    """
    Desc: Runs when a queued seal is clicked: shows its pictures and selects it as the label
    arguments: seal_number- queued seal's number, as a string
    returns: None
    """
    st.session_state.selected_queued = seal_number
    st.session_state.selected_label = seal_number  #Changes "Select the correct seal label" to this seal
    st.session_state.compare = 0
    reset_new_seal()

@st.dialog("Confirm label")
def confirm_save(image, seal_id, is_queued, source_file, date_taken, default_side):
    """
    Desc: Popup asking the user to double check the label before saving to the database or the training queue
    arguments: image- confirmed crop to save
               seal_id- label chosen in "Select the correct seal label"
               is_queued- True if seal_id is a queued new seal (saves to the training queue instead)
               source_file- name of the uploaded file
               date_taken- date from the photo's EXIF, or None
               default_side- side guessed from the filename, or None
    returns: None
    """
    st.image(image, use_container_width=True)
    if is_queued:
        st.warning(f"Are you sure this is queued seal **{seal_id}**? It will be added to the training queue "
                   "with that seal's other pictures.")
    else:
        st.warning(f"Are you sure this is seal **{seal_id}**? It will be added to the database "
                   "and used as a reference photo for this seal.")
    st.caption(f"File: {source_file} · Taken: {date_taken or 'unknown'}")

    if not is_queued:
        side_options = [None, "LS", "F", "RS"]
        side = st.selectbox("Side of the seal", side_options, index=side_options.index(default_side),
                            format_func=lambda s: SIDE_NAMES.get(s, "Unknown"))

    col1, col2 = st.columns(2)
    with col1:
        confirmed = st.button("✅ Yes, the label is correct")
    with col2:
        cancelled = st.button("❌ Cancel")

    if confirmed and is_queued:
        try:
            queue_for_training(image, seal_id, source_file, seal_class="new")
        except OSError as e:
            st.error(f"Could not add to the training queue: {e}")
            return
        st.session_state.db_save_message = ("success", f"Added to the training queue as seal {seal_id}.")
        st.rerun()
    elif confirmed:
        try:
            photo_num = add_seal_photo(seal_id, image, side, date_taken, source_file)
        except sqlite3.Error as e:
            st.error(f"Could not save to the database: {e}")
            return
        if photo_num is None:
            st.session_state.db_save_message = ("info", "This image is already in the database.")
        else:
            st.session_state.db_save_message = ("success", f"Saved as photo {photo_num} of seal {seal_id}.")
        st.rerun()
    if cancelled:
        st.rerun()

# Page setup
st.title("Seal Individual Identifier 🦭")
st.write("Upload a single image or a folder of images of seals. Navigate through them and see predictions!")
if USING_DEV_DB:
    st.caption("Using the developer database (3 photos per seal). Set SEAL_DB_PATH to use the full one.")

# File uploader allows multiple files
uploaded_files = st.file_uploader("Choose a folder of images...", type=["jpg", "jpeg", "png"], accept_multiple_files="directory")
uploaded_files = sorted(uploaded_files, key=lambda f: f.name)

# Load model and classifier
model, processor, id2label = load_model_and_processor()
classifier = create_classifier(model, processor)
known_seals = [str(int(label)) for label in id2label.values() if label.isdigit()]

# Number of seals to show (defualts to 3)
if "seals_to_show" not in st.session_state:
    st.session_state.seals_to_show = 3
def show_more_button():
    if st.session_state.seals_to_show == 3:
        st.button("Show More Seals 🔽", on_click=lambda: st.session_state.update(seals_to_show=10))
    else:
        st.button("Show Fewer Seals 🔼", on_click=lambda: st.session_state.update(seals_to_show=3))

def render_queued_seals_box(queued):
    """
    Desc: Draws the queued seals box, and the pictures of the queued seal that was clicked
    arguments: queued- queued seal numbers from queued_seal_numbers
    returns: None
    """
    st.markdown("### 📋 Queued Seals")
    if not queued:
        st.caption("No new seals are waiting in the training queue.")
        return

    shown = queued if st.session_state.show_all_queued else queued[:3]
    with st.container(border=True):
        for row_start in range(0, len(shown), 3):
            for col, seal_number in zip(st.columns(3), shown[row_start:row_start + 3]):
                with col:
                    st.button(f"**{seal_number}**", key=f"queued-{seal_number}",
                              on_click=show_queued_seal, args=(seal_number,), width="stretch")

    if len(queued) > 3:
        if st.session_state.show_all_queued:
            st.button("Show Fewer Queued Seals 🔼", on_click=lambda: st.session_state.update(show_all_queued=False))
        else:
            st.button(f"Show More Queued Seals 🔽 ({len(queued)})", on_click=lambda: st.session_state.update(show_all_queued=True))

    if st.session_state.selected_queued:
        seal_number = st.session_state.selected_queued
        images = queued_seal_images(seal_number)
        st.markdown(f"#### Queued seal {seal_number} · {len(images)} picture(s)")
        for row_start in range(0, len(images), 3):
            for col, (path, source_filename) in zip(st.columns(3), images[row_start:row_start + 3]):
                with col:
                    st.image(str(path), caption=source_filename, use_container_width=True)


if uploaded_files:
    #Start from the first picture whenever a different set of pictures is uploaded
    if st.session_state.get("uploaded_names") != [f.name for f in uploaded_files]:
        st.session_state.uploaded_names = [f.name for f in uploaded_files]
        st.session_state.img_index = 0
        st.session_state.crop_confirmed = False
        st.session_state.pop("selected_label", None)

    image_files = uploaded_files
    total_images = len(image_files)
    current_index = st.session_state.img_index
    current_image_file = image_files[current_index]
    raw_image = Image.open(current_image_file)
    date_taken = exif_date_taken(raw_image)  #Read before convert(), which drops EXIF
    original_image = raw_image.convert("RGB")

    st.markdown(f"### Image {current_index + 1} of {total_images}")
    st.text(f"File: {uploaded_files[current_index].name}")

    #Grabs OwlVIT crop
    best_box = get_recommended_crop(original_image)

    #If OwlVIT finds a crop then map it to coordinates for streamlit cropper
    if best_box:
        x1, y1, x2, y2 = map(int, best_box)
        default_coords = (x1, x2, y1, y2)
    else:
        default_coords = None

    st.markdown("### ✂️ Crop your image")
    cropped_image = st_cropper(
        original_image,
        realtime_update=True,
        default_coords=default_coords, #Gives cropper the OwlVIT crop as the default box
        box_color="blue",
        aspect_ratio=None,
        return_type="image",
        key=f"cropper-{current_index}",
        should_resize_image=True
    )

    #Preview of cropped image
    st.image(cropped_image, caption="Cropped Preview", use_container_width=True)

    #Button to confirm img crop
    if st.button("✅ Confirm Crop"):
        st.session_state.crop_confirmed = True

    #Predict class of cropped img
    if st.session_state.crop_confirmed:
        with st.spinner("Predicting..."):
            predictions = predict_image(classifier, cropped_image)

        st.markdown("### 🔍 Top Predictions:")
        st.text("Formatted as: {Seal individual} · {confidence}%")

        #Thin gray box around the predictions, laid out left to right, 3 per row
        with st.container(border=True):
            for row_start in range(0, st.session_state.seals_to_show, 3):
                cols = st.columns(3)
                for col, i in zip(cols, range(row_start, min(row_start + 3, st.session_state.seals_to_show))):
                    with col:
                        conf_score = predictions[i]['score'] * 100
                        st.button(f"**{predictions[i]['label']}** · {conf_score:.0f}%",
                                  on_click=compare, args=(i+1,), width="stretch")

        queued = queued_seal_numbers(known_seals)

        show_more_button()

        render_queued_seals_box(queued)

        render_new_seal_button(predictions)
        render_new_seal_flow(cropped_image, uploaded_files[current_index].name, known_seals)

        if st.session_state.compare > 0:

            selected_label = predictions[st.session_state.compare - 1]['label']
            photos = load_seal_photos(selected_label)
            image1 = cropped_image

            def next_image():
                st.session_state.selected_img += 1
            def prev_image():
                st.session_state.selected_img -= 1

            col1, col2, col3, col4 = st.columns(4)
            with col3:
                st.button("Previous Image", on_click=prev_image)
            with col4:
                st.button("Next Image", on_click=next_image)

            # Display images side by side
            col1, col2 = st.columns(2)

            with col1:
                st.image(image1, caption="Unidentified Seal", use_container_width=True)

            with col2:
                if photos:
                    photo_index = st.session_state.selected_img % len(photos)
                    image_bytes, side, date_taken = photos[photo_index]
                    st.image(
                        image_bytes,
                        caption=f"Seal {selected_label} · {SIDE_NAMES.get(side, 'Side unknown')} · "
                                f"{date_taken[:10] if date_taken else 'Date unknown'} · "
                                f"Photo {photo_index + 1} of {len(photos)}",
                        use_container_width=True,
                    )
                else:
                    st.warning(f"No database photos for seal {selected_label}.")

        prediction_labels = [predictions[i]['label'] for i in range(st.session_state.seals_to_show)]
        scores = {p['label']: p['score'] for p in predictions}
        label_options = prediction_labels + queued

        #If the selected label is no longer listed (e.g. after Show Fewer Seals), go back to the top prediction
        if st.session_state.get("selected_label") not in label_options:
            st.session_state.pop("selected_label", None)
        selected_label = st.selectbox(
            "Select the correct seal label:", label_options, key="selected_label",
            format_func=lambda label: f"{label} (queued)" if label in queued else f"{label} ({scores[label]*100:.0f}%)",
        )
        is_queued = selected_label in queued

        if st.button("📋 Add to Training Queue" if is_queued else "🗄️ Save to Database"):
            confirm_save(cropped_image, selected_label, is_queued, current_image_file.name,
                         date_taken, guess_side(current_image_file.name))
        if "db_save_message" in st.session_state:
            kind, message = st.session_state.pop("db_save_message")
            getattr(st, kind)(message)

        default_filename = f"{selected_label}.jpg"

        # Navigation buttons
        def go_previous():
            st.session_state.img_index = max(0, st.session_state.img_index - 1)
            st.session_state.crop_confirmed = False
            st.session_state.pop("selected_label", None)
            compare(0)

        def go_next():
            st.session_state.img_index = min(total_images - 1, st.session_state.img_index + 1)
            st.session_state.crop_confirmed = False
            st.session_state.pop("selected_label", None)
            compare(0)

        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("⬅️ Previous", on_click=go_previous, disabled=current_index == 0)
        with col2:
            st.button("Next ➡️", on_click=go_next, disabled=current_index == total_images - 1)


        filename = st.text_input("Enter filename to save this image:", default_filename)
        st.markdown("### 💾 Add Image to Queue")

        if st.button("💾 Add Image"):
            #TODO - Change this to add image to a queue (local or shared?)
            save_path = os.path.join(st.session_state.saved_crops_dir, filename)
            cropped_image.save(save_path)
            st.success(f"Image saved as {filename}")
            if filename not in st.session_state.saved_files:
                st.session_state.saved_files.append(filename)

if st.session_state.saved_files:
    st.markdown("### 📦 Download All Saved Cropped Images")

    zip_path = os.path.join(st.session_state.saved_crops_dir, "seal_images.zip")

    # Create zip only if not already created
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for fname in st.session_state.saved_files:
            fpath = os.path.join(st.session_state.saved_crops_dir, fname)
            zipf.write(fpath, arcname=fname)

    with open(zip_path, "rb") as f:
        st.download_button(
            label="⬇️ Download All Cropped Images",
            data=f,
            file_name="cropped_images.zip",
            mime="application/zip"
        )
