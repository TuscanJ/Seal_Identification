import argparse
import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from transformers import AutoImageProcessor, ResNetForImageClassification, TrainingArguments, Trainer, AutoModelForImageClassification, ViTImageProcessor, ViTForImageClassification
import evaluate
import numpy as np
from dotenv import load_dotenv
import os
import wandb
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import environ

from hf_dataset import HFDataset

ROOT_DIR = '../storage/date_split'


accuracy_metric = evaluate.load("accuracy")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

classes = []
for seal_folder in os.listdir("../storage/date_split"):
    classes.append(int(seal_folder))
classes.sort()

def save_classification_report(true_labels, predictions, model, train_dataset, file_path="class_metrics.csv"):
    true_labels = [str(label) for label in true_labels]
    predictions = [str(label) for label in predictions]

    # Generate a classification report
    report_dict = classification_report(true_labels, predictions, output_dict=True)

    report_df = pd.DataFrame(report_dict).transpose()

    report_df = report_df.reset_index().rename(columns={"index": "seal"})

    # Ensure the "seal" column remains a string to preserve leading zeros
    report_df["seal"] = report_df["seal"].astype(str)

    train_counts = {}
    for sample in train_dataset:
        label_idx = sample["label"]
        label_str = model.config.id2label[label_idx]
        label_str = label_str.zfill(4)
        train_counts[label_str] = train_counts.get(label_str, 0) + 1

    print(train_counts)
    
    train_counts["accuracy"] = 0
    train_counts["macro avg"] = 0
    train_counts["weighted avg"] = 0
    report_df["num_training_examples"] = report_df["seal"].map(train_counts).fillna(0).astype(int)

    report_df.to_csv(file_path, index=False)

    # Plot the table as an image
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("tight")
    ax.axis("off")

    # Create the table
    table = ax.table(cellText=report_df.values, 
                     colLabels=report_df.columns, 
                     cellLoc="center", 
                     loc="center", 
                     colWidths=[0.15] + [0.2] * (len(report_df.columns) - 1)
)

    # Save as image
    plt.savefig("class_metrics.png", bbox_inches="tight", dpi=300)
    print(f"Classification report saved as class_metrics.png and {file_path}")


def compute_metrics(eval_pred, k=5):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)  # Get highest probability class

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")

    top_k_pred = np.argsort(logits, axis=-1)[:, -k:]
    wandb.init(project="huggingface", name="confusion-matrix")
    for i in range(len(predictions)):
        if int(predictions[i]) > 52:
            predictions[i] = 52
    
    actual_labels = [classes[i] for i in labels]
    actual_predictions = [classes[i] for i in predictions]
    # Compute confusion matrix
    cm = confusion_matrix(actual_labels, actual_predictions, normalize="true")  # Normalize per class

    # Define class names (trim if too long)
    class_names = [f"Seal {i}" for i in classes]

    # Create heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=class_names, yticklabels=class_names)

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Normalized Confusion Matrix")
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)

    # Save figure
    plt.savefig("confusion_matrix.png")
    wandb.log({"confusion_matrix": wandb.Image("confusion_matrix.png")})

    top_2_pred = np.argsort(logits, axis=-1)[:, -2:]
    top_5_pred = np.argsort(logits, axis=-1)[:, -5:]
    top_10_pred = np.argsort(logits, axis=-1)[:, -10:]
    correct = 0
    t2_correct = 0
    t5_correct = 0
    t10_correct = 0
    for i in range(len(labels)):
        if labels[i] in top_2_pred[i]:
            t2_correct += 1
        if labels[i] in top_5_pred[i]:
            t5_correct += 1
        if labels[i] in top_10_pred[i]:
            t10_correct += 1
    # top_k_accuracy = correct / len(labels)
    t2_acc = t2_correct / len(labels)
    t5_acc = t5_correct / len(labels)
    t10_acc = t10_correct / len(labels)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1': f1,
            'top_k_accuracy': {'2': t2_acc, '5': t5_acc, '10': t10_acc}}

def main(argv):

    load_dotenv()
    wandb.login()

    # Path to your root data directory
    your_root_dir = ROOT_DIR

    # Initialize one dataset per split
    train_dataset = HFDataset(your_root_dir, 'train',  processor=processor, transform = transform)
    val_dataset = HFDataset(your_root_dir, 'dev',  processor=processor, transform = transform)
    test_dataset = HFDataset(your_root_dir, 'test', processor=processor)


    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')

    # Build mapping from indices to your original class labels
    class_labels = sorted(set(train_dataset.classes))  # ensures sorted unique labels
    num_labels = len(class_labels)
    id2label = {i: label for i, label in enumerate(class_labels)}
    label2id = {label: i for i, label in enumerate(class_labels)}

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224', num_labels=num_labels, ignore_mismatched_sizes=True)


    # Update model configuration so predictions include the original labels
    model.config.id2label = id2label
    model.config.label2id = label2id

    args = parse_all_args()

    if args.run_name == "":
        name = "vit"
    else:
        name = args.run_name
    training_args = TrainingArguments(
        "test-trainer",
        report_to="wandb",
        run_name=name,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        num_train_epochs=args.epochs
    )

    trainer = Trainer(
        model,
        training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )
    trainer.train()

    metrics = trainer.evaluate()
    print(metrics)

    if args.save:
        print('saving model . . .')
        trainer.save_model("./vit")
        processor.save_pretrained("./vit_processor")
    
    # Run predictions on the dev set using your trained trainer
    predictions_output = trainer.predict(val_dataset)
    logits, labels, _ = predictions_output

    # Convert logits to predicted labels
    preds = np.argmax(logits, axis=-1)

    original_preds = [id2label[p] for p in preds]
    original_labels = [id2label[l] for l in labels]

    # Compute the confusion matrix
    cm = confusion_matrix(original_labels, original_preds)
    # Save per class metrics
    save_classification_report(original_labels, original_preds, model, train_dataset, file_path="metrics.csv")

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix on Dev Set")
    plt.tight_layout()

    # Save the figure to a file
    plt.savefig("conf_matrices/confusion_matrix.png")
    plt.close()



def parse_all_args():
    """
    The parse_all_args function parses the commandline arguments
    """

    parser = argparse.ArgumentParser()

    parser.add_argument("-opt", type=str,\
            help='The optimizer: "adadelta", "adagrad", "adam", "adamw","rmsprop", "sgd" (string) [default: "adam"]', choices={"adadelta", "adagrad", "adam", "adamw","rmsprop", "sgd"}, default="adam")
    parser.add_argument("-lr",type=float,\
            help="The learning rate (float) [default: 0.001]",default=0.001)
    parser.add_argument("-mb", type=int,\
            help="The minibatch size (int) [default: 16]", default=16)
    parser.add_argument("-report_freq", type=int,\
            help="Dev performance is reported every report_freq updates(int) [default: 128]", default=128)
    parser.add_argument("-epochs", type=int,\
            help="The number of training epochs (int) [default: 20]", default=20)
    parser.add_argument("-save", action='store_true',\
        help="Flag for saving model params after training is done [default: False]")
    parser.add_argument("-confmatrix", action='store_true',\
        help="Create a Confusion Matrix for the validation set [default: False]")
    parser.add_argument("-run_name", type=str,\
            help="Run name to report to wandb (str) [default: model name]", default="")
    
    return parser.parse_args()

if __name__ == "__main__":
    main(sys.argv)