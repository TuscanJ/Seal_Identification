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

def predict_image(classifier, image):
    """
    Desc: predicts an image using the given classifier
    arguments: classifier- classifier to be used on image
               image- image to be classified
    returns: top 3 most likely classes for image
    """
    return classifier(image, top_k=3)

if 'compare' not in st.session_state:
    st.session_state.compare = 0
if 'selected_img' not in st.session_state:
    st.session_state.selected_img = 0

def reset_selected_img():
    st.session_state.selected_img = 0

def compare1():
    st.session_state.compare = 1
def compare2():
    st.session_state.compare = 2
def compare3():
    st.session_state.compare = 3

def compare(i):
    st.session_state.compare = i
    reset_selected_img()

# Page setup
st.set_page_config(page_title="Seal Identifier 🦭", layout="centered")
st.title("Seal Individual Identifier 🦭")
st.write("Upload a single image or a folder of images of seals. Navigate through them and see predictions!")

# File uploader allows multiple files
uploaded_files = st.file_uploader("Choose image(s)...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Load model and classifier
model, processor, id2label = load_model_and_processor()
classifier = create_classifier(model, processor)

if uploaded_files:
    if "img_index" not in st.session_state:
        st.session_state.img_index = 0
    if "crop_confirmed" not in st.session_state:
        st.session_state.crop_confirmed = False

    image_files = uploaded_files
    total_images = len(image_files)
    current_index = st.session_state.img_index
    current_image_file = image_files[current_index]
    original_image = Image.open(current_image_file).convert("RGB")

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
        st.text("Formatted as: {Seal individual} with confidence {score of confidence}")

        conf_score = predictions[0]['score'] * 100
        st.button(f"- **{predictions[0]['label']}** with confidence **{conf_score:.0f}**%", on_click=compare1)

        conf_score = predictions[1]['score'] * 100
        st.button(f"- **{predictions[1]['label']}** with confidence **{conf_score:.0f}**%", on_click=compare2)

        conf_score = predictions[2]['score'] * 100
        st.button(f"- **{predictions[2]['label']}** with confidence **{conf_score:.0f}**%", on_click=compare3)

        if st.session_state.compare > 0:

            selected_label = str(int(predictions[st.session_state.compare-1]['label']))
            train_folder = Path(f"app/seal_imgs/{selected_label}/train")
            dev_folder = Path(f"app/seal_imgs/{selected_label}/dev")

            train_files = list(train_folder.glob("*"))
            dev_files = list(dev_folder.glob("*"))

            all_files = train_files + dev_files

            image1 = cropped_image
            image2 = Image.open(all_files[st.session_state.selected_img])

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
                st.image(image2, caption=f"Seal {selected_label}", use_container_width=True)

        prediction_options = [
            f"{pred['label']} ({pred['score']*100:.0f}%)"
            for pred in predictions
        ]
        prediction_labels = [pred['label'] for pred in predictions]

        selected_option = st.selectbox("Select the correct seal label:", prediction_options)
        selected_label = prediction_labels[prediction_options.index(selected_option)]

        default_filename = f"{selected_label}.jpg"

        # Navigation buttons
        def go_previous():
            st.session_state.img_index = max(0, st.session_state.img_index - 1)
            st.session_state.crop_confirmed = False

        def go_next():
            st.session_state.img_index = min(total_images - 1, st.session_state.img_index + 1)
            st.session_state.crop_confirmed = False

        col1, col2 = st.columns([1, 1])
        with col1:
            st.button("⬅️ Previous", on_click=go_previous, disabled=current_index == 0)
        with col2:
            st.button("Next ➡️", on_click=go_next, disabled=current_index == total_images - 1)

                
        filename = st.text_input("Enter filename to save this image:", default_filename)
        st.markdown("### 💾 Save Cropped Image")
        if st.button("💾 Save Image"):
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



