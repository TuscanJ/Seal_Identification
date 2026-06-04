from transformers import pipeline, ViTImageProcessor, ViTForImageClassification
import argparse
from PIL import Image
import torch
import pandas as pd
import os


def load_model_and_processor(MODEL_DIR):
    model = ViTForImageClassification.from_pretrained(MODEL_DIR)
    processor = ViTImageProcessor.from_pretrained(MODEL_DIR)

    id2label = model.config.id2label

    return model, processor, id2label

def create_classifier(model, processor):
    classifier = pipeline(
        "image-classification",
        model = model,
        feature_extractor= processor,
        device = 0 if torch.cuda.is_available() else -1
    )
    return classifier

def predict_seal(seal_path, classifier, k):
    image = Image.open(seal_path).convert("RGB")
    return classifier(image, top_k=k)

def predictions(IMG_DIR, classifier, k, output_file):
    img_lst = os.listdir(IMG_DIR)
    img_paths = [os.path.join(IMG_DIR, file) for file in img_lst]

    rows = []

    for path in img_paths:
        preds = predict_seal(path, classifier, k)
        row = {'filename': os.path.basename(path)}
        for i, pred in enumerate(preds, start=1):
            row[f'label_{i}'] = pred['label']
            row[f'score_{i}'] = round(pred['score'], 4)
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(f'{output_file}.csv', index=False)
    print("Predictions saved to predictions.csv")
        
    


def main(argv):
    args = parse_all_args()

    MODEL_DIR = args.model_dir
    IMG_DIR = args.img_dir

    model, processor, id2label = load_model_and_processor(MODEL_DIR)
    classifier = create_classifier(model, processor)
    predictions(IMG_DIR, classifier, args.top_k, args.output_file)
    print(f"Predictions have been successfully saved to {args.output_file}.csv")

    

def parse_all_args():
    """
    The parse_all_args function parses the commandline arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-model_dir", type=str, \
            help="Path to your saved model directory [default: ./model]", default="./model")
    parser.add_argument("-img_dir", type=str, \
            help="Path to un-classified images [default: ./images]", default="./images")
    parser.add_argument("-top_k", type=int, \
            help="Top k number of predictions for each seal will be returned [default: 1]", default=1)
    parser.add_argument("-output_file", type=str, \
        help="Desired name of outputted csv file [default: predictions]", default="predictions")
    

    
    return parser.parse_args()

if __name__ == "__main__":
    import sys
    main(sys.argv)