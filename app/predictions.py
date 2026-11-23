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
import io
from collections import Counter

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

@st.dialog("Confirm labels")
def confirm_save_all(files):
    """
    Desc: Popup asking the user to double check every label, then saves them all to the database or training queue
    arguments: files- uploaded image files
    returns: None
    """
    labels = st.session_state.labels
    n_queue = sum(info["queued"] for info in labels.values())
    st.warning(f"Are you sure these {len(labels)} photos have the correct labels? {len(labels) - n_queue} will be "
               f"added to the database and {n_queue} will go to the training queue.")
    counts = Counter(f"{info['label']} (queued)" if info["queued"] else info["label"] for info in labels.values())
    st.markdown("\n".join(f"- **{label}** × {count}" for label, count in sorted(counts.items())))

    col1, col2 = st.columns(2)
    with col1:
        confirmed = st.button("✅ Yes, save them all")
    with col2:
        cancelled = st.button("❌ Cancel")

    if confirmed:
        results = {"saved": 0, "duplicates": 0, "queued": 0, "failed": []}
        bar = st.progress(0.0, text="Saving...")
        for n, (i, info) in enumerate(sorted(labels.items()), 1):
            crop, name = st.session_state.crops[i], files[i].name
            try:
                if info["queued"]:
                    queue_for_training(crop, info["label"], name, seal_class="new")
                    results["queued"] += 1
                elif add_seal_photo(info["label"], crop, info["side"], st.session_state.photo_dates.get(i), name) is None:
                    results["duplicates"] += 1
                else:
                    results["saved"] += 1
            except (sqlite3.Error, OSError) as e:
                results["failed"].append(f"{name}: {e}")
            bar.progress(n / len(labels), text=f"Saving {n} of {len(labels)}...")
        st.session_state.save_results = results
        st.rerun()
    if cancelled:
        st.rerun()

# Page setup
st.title("Seal Individual Identifier 🦭")
st.write("Upload a single image or a folder of images of seals. Crop every image, label them, then review and save them all at once.")
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


STEPS = {"crop": "① Crop", "label": "② Label", "review": "③ Review & Save"}

def reset_batch():
    """
    Desc: Clears all work on the uploaded folder (call when a different folder is uploaded)
    arguments: None
    returns: None
    """
    st.session_state.step = "crop"
    st.session_state.img_index = 0
    st.session_state.crops = {}            #image index -> confirmed crop (PIL image)
    st.session_state.crop_boxes = {}       #image index -> confirmed box, in st_cropper's default_coords format
    st.session_state.photo_dates = {}      #image index -> date from EXIF, or None
    st.session_state.predictions = {}      #image index -> predict_image results for that crop
    st.session_state.labels = {}           #image index -> {"label", "queued", "side"} once the label is confirmed
    st.session_state.new_seal_images = {}  #image index -> seal number, for images already queued with New Seal
    st.session_state.save_results = None
    st.session_state.pop("selected_label", None)
    compare(0)

def go_to(step, index):
    """
    Desc: Moves to a step and image, closing anything left open on the previous image
    arguments: step- "crop", "label" or "review"
               index- image index to show
    returns: None
    """
    st.session_state.step = step
    st.session_state.img_index = index
    st.session_state.pop("selected_label", None)
    compare(0)

def confirm_crop(i, crop, box, total):
    """
    Desc: Saves the crop for image i and moves to the next image
    arguments: i- image index
               crop- cropped PIL image
               box- st_cropper box {"left", "top", "width", "height"}
               total- number of uploaded images
    returns: None
    """
    st.session_state.crops[i] = crop
    st.session_state.crop_boxes[i] = (box["left"], box["left"] + box["width"], box["top"], box["top"] + box["height"])
    st.session_state.predictions.pop(i, None)  #A new crop needs a new prediction
    go_to("crop", min(i + 1, total - 1))

def render_crop_step(files):
    """
    Desc: Step 1, the user confirms a crop for every uploaded image
    arguments: files- uploaded image files, sorted by name
    returns: None
    """
    total = len(files)
    i = st.session_state.img_index
    raw_image = Image.open(files[i])
    st.session_state.photo_dates[i] = exif_date_taken(raw_image)  #Read before convert(), which drops EXIF
    original_image = raw_image.convert("RGB")

    done = len(st.session_state.crops)
    st.progress(done / total, text=f"Cropped {done} of {total}")
    st.markdown(f"### ✂️ Crop image {i + 1} of {total}")
    st.text(f"File: {files[i].name}")

    #Start from the box confirmed earlier if there is one, otherwise OwlVIT's suggestion
    default_coords = st.session_state.crop_boxes.get(i)
    if default_coords is None:
        best_box = get_recommended_crop(original_image)
        if best_box:
            x1, y1, x2, y2 = map(int, best_box)
            default_coords = (x1, x2, y1, y2)

    cropped_image, box = st_cropper(original_image, realtime_update=True, default_coords=default_coords,
                                    box_color="blue", aspect_ratio=None, return_type="both",
                                    key=f"cropper-{i}", should_resize_image=True)
    st.image(cropped_image, caption="Cropped Preview", use_container_width=True)
    if i in st.session_state.crops:
        st.caption("✅ Crop confirmed. Confirm again to replace it.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("⬅️ Previous", on_click=go_to, args=("crop", i - 1), disabled=i == 0)
    with col2:
        st.button("✅ Confirm Crop & Next", type="primary", on_click=confirm_crop, args=(i, cropped_image, box, total))
    with col3:
        st.button("Next ➡️", on_click=go_to, args=("crop", i + 1),
                  disabled=i == total - 1 or i not in st.session_state.crops)

    st.button("Continue to Labeling ➡️", on_click=go_to, args=("label", 0), disabled=done < total)
    if done < total:
        st.caption(f"Confirm a crop for every image to continue ({total - done} left).")

def confirm_label(i, label, queued, side, total):
    """
    Desc: Saves the label for image i and moves to the next image
    arguments: i- image index
               label- chosen seal label
               queued- True if label is a queued new seal
               side- "LS", "F", "RS" or None
               total- number of uploaded images
    returns: None
    """
    st.session_state.labels[i] = {"label": label, "queued": queued, "side": None if queued else side}
    go_to("label", min(i + 1, total - 1))

def render_label_step(files, known_seals):
    """
    Desc: Step 2, the user confirms a label for every cropped image
    arguments: files- uploaded image files, sorted by name
               known_seals- seal numbers the model already knows, as strings
    returns: None
    """
    total = len(files)
    i = st.session_state.img_index
    crop = st.session_state.crops[i]

    #Predict every crop up front, so moving between images is instant
    missing = [j for j in range(total) if j not in st.session_state.predictions]
    if missing:
        bar = st.progress(0.0, text="Predicting...")
        for n, j in enumerate(missing, 1):
            st.session_state.predictions[j] = predict_image(classifier, st.session_state.crops[j])
            bar.progress(n / len(missing), text=f"Predicting {n} of {len(missing)}...")
        bar.empty()
    predictions = st.session_state.predictions[i]

    #New Seal already put this image in the training queue, so it counts as done and isn't saved again in step 3
    if st.session_state.new_seal_entry:
        st.session_state.new_seal_images[i] = st.session_state.new_seal_entry["seal_number"]
        st.session_state.labels.pop(i, None)

    done = len(st.session_state.labels) + len(st.session_state.new_seal_images)
    finished = i in st.session_state.labels or i in st.session_state.new_seal_images
    st.progress(done / total, text=f"Labeled {done} of {total}")
    st.markdown(f"### 🔍 Label image {i + 1} of {total}")
    st.text(f"File: {files[i].name}")
    st.image(crop, caption="Your crop", width=300)
    st.button("⬅️ Back to Cropping", on_click=go_to, args=("crop", i))

    st.markdown("### 🔍 Top Predictions:")
    st.text("Formatted as: {Seal individual} · {confidence}%")

    #Thin gray box around the predictions, laid out left to right, 3 per row
    with st.container(border=True):
        for row_start in range(0, st.session_state.seals_to_show, 3):
            cols = st.columns(3)
            for col, k in zip(cols, range(row_start, min(row_start + 3, st.session_state.seals_to_show))):
                with col:
                    conf_score = predictions[k]['score'] * 100
                    st.button(f"**{predictions[k]['label']}** · {conf_score:.0f}%",
                              on_click=compare, args=(k+1,), width="stretch")

    queued = queued_seal_numbers(known_seals)

    show_more_button()

    render_queued_seals_box(queued)

    render_new_seal_button(predictions)
    render_new_seal_flow(crop, files[i].name, known_seals)

    if st.session_state.compare > 0:

        selected_label = predictions[st.session_state.compare - 1]['label']
        photos = load_seal_photos(selected_label)
        image1 = crop

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
                image_bytes, photo_side, photo_date = photos[photo_index]
                st.image(
                    image_bytes,
                    caption=f"Seal {selected_label} · {SIDE_NAMES.get(photo_side, 'Side unknown')} · "
                            f"{photo_date[:10] if photo_date else 'Date unknown'} · "
                            f"Photo {photo_index + 1} of {len(photos)}",
                    use_container_width=True,
                )
            else:
                st.warning(f"No database photos for seal {selected_label}.")

    if i in st.session_state.new_seal_images:
        st.success(f"Queued as new seal {st.session_state.new_seal_images[i]}. It won't be saved again in step 3.")
    else:
        prediction_labels = [predictions[k]['label'] for k in range(st.session_state.seals_to_show)]
        scores = {p['label']: p['score'] for p in predictions}
        label_options = prediction_labels + queued

        #Coming back to a labeled image shows the label that was confirmed for it
        saved = st.session_state.labels.get(i)
        if saved and "selected_label" not in st.session_state:
            st.session_state.selected_label = saved["label"]
        #If the selected label is no longer listed (e.g. after Show Fewer Seals), go back to the top prediction
        if st.session_state.get("selected_label") not in label_options:
            st.session_state.pop("selected_label", None)
        selected_label = st.selectbox(
            "Select the correct seal label:", label_options, key="selected_label",
            format_func=lambda label: f"{label} (queued)" if label in queued else f"{label} ({scores[label]*100:.0f}%)",
        )
        is_queued = selected_label in queued

        side_options = [None, "LS", "F", "RS"]
        default_side = saved["side"] if saved else guess_side(files[i].name)
        side = st.selectbox("Side of the seal", side_options, index=side_options.index(default_side),
                            key=f"side-{i}", disabled=is_queued,
                            format_func=lambda s: SIDE_NAMES.get(s, "Unknown"))
        if saved:
            st.caption(f"✅ Label confirmed: {saved['label']}. Confirm again to replace it.")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.button("⬅️ Previous", key="label-prev", on_click=go_to, args=("label", i - 1), disabled=i == 0)
    with col2:
        if i not in st.session_state.new_seal_images:
            st.button("✅ Confirm Label & Next", type="primary", on_click=confirm_label,
                      args=(i, selected_label, is_queued, side, total))
    with col3:
        st.button("Next ➡️", key="label-next", on_click=go_to, args=("label", i + 1),
                  disabled=i == total - 1 or not finished)

    st.button("Continue to Review ➡️", on_click=go_to, args=("review", 0), disabled=done < total)
    if done < total:
        st.caption(f"Confirm a label for every image to continue ({total - done} left).")

def render_review_grid(title, indexes, files, describe, editable):
    """
    Desc: Draws a box of crops for step 3, 3 per row, each with its destination
    arguments: title- heading for the group
               indexes- image indexes in this group
               files- uploaded image files
               describe- function giving the caption line for an image index
               editable- True to show an Edit button that goes back to that image's label
    returns: None
    """
    if not indexes:
        return
    st.markdown(f"#### {title} ({len(indexes)})")
    with st.container(border=True):
        for row_start in range(0, len(indexes), 3):
            for col, i in zip(st.columns(3), indexes[row_start:row_start + 3]):
                with col:
                    st.image(st.session_state.crops[i], caption=f"{describe(i)} · {files[i].name}", use_container_width=True)
                    if editable:
                        st.button("✏️ Edit", key=f"edit-{i}", on_click=go_to, args=("label", i),
                                  disabled=st.session_state.save_results is not None)

def render_review_step(files):
    """
    Desc: Step 3, shows where every image will go and saves them all at once
    arguments: files- uploaded image files, sorted by name
    returns: None
    """
    labels, new_seals = st.session_state.labels, st.session_state.new_seal_images
    results = st.session_state.save_results
    st.markdown("### 🗂️ Review & Save")

    render_review_grid("→ Database", [i for i in sorted(labels) if not labels[i]["queued"]], files,
                       lambda i: f"Seal {labels[i]['label']} · {SIDE_NAMES.get(labels[i]['side'], 'Side unknown')}", True)
    render_review_grid("→ Training Queue", [i for i in sorted(labels) if labels[i]["queued"]], files,
                       lambda i: f"Queued seal {labels[i]['label']}", True)
    render_review_grid("Already queued with New Seal", sorted(new_seals), files,
                       lambda i: f"New seal {new_seals[i]}", False)

    if results:
        message = f"Saved {results['saved']} to the database and added {results['queued']} to the training queue."
        if results["duplicates"]:
            message += f" {results['duplicates']} were already in the database."
        st.success(message)
        for failure in results["failed"]:
            st.error(f"Could not save {failure}")
        st.caption("Upload another folder to start again.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.button("⬅️ Back to Labeling", on_click=go_to, args=("label", 0))
        with col2:
            if st.button(f"💾 Save All {len(labels)}", type="primary", disabled=not labels):
                confirm_save_all(files)

    #Every confirmed crop, named by its label, e.g. 0101_IMG_3132.jpg
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zipf:
        for i, crop in sorted(st.session_state.crops.items()):
            prefix = labels[i]["label"] if i in labels else new_seals.get(i, "unlabeled")
            image_bytes = io.BytesIO()
            crop.save(image_bytes, format="JPEG")
            zipf.writestr(f"{prefix}_{Path(files[i].name).stem}.jpg", image_bytes.getvalue())
    st.download_button("⬇️ Download All Crops (zip)", buffer.getvalue(), file_name="cropped_images.zip", mime="application/zip")


if uploaded_files:
    #Start over whenever a different set of pictures is uploaded
    if st.session_state.get("uploaded_names") != [f.name for f in uploaded_files]:
        st.session_state.uploaded_names = [f.name for f in uploaded_files]
        reset_batch()

    st.markdown(" → ".join(f"**{name}**" if key == st.session_state.step else name for key, name in STEPS.items()))
    if st.session_state.step == "crop":
        render_crop_step(uploaded_files)
    elif st.session_state.step == "label":
        render_label_step(uploaded_files, known_seals)
    else:
        render_review_step(uploaded_files)
