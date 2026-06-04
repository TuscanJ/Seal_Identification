import os
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import AutoFeatureExtractor
import pandas as pd

DATA_FOLDER = "../storage/date_split"

classes = []
for seal_folder in os.listdir(DATA_FOLDER):
    classes.append(int(seal_folder))
classes.sort()

class HFDataset(Dataset):
    # This init function is run once, when a dataset object is initialized
    def __init__(self, img_dir, phase, transform = None, mask_dir=None, processor=None):
        
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        if mask_dir is not None:
            self.mask_paths = []
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(self.img_dir)) 
        self.processor = processor
        self.transform = transform

        for seal in os.listdir(self.img_dir):
            # Example: {img dir}/{seal}/{train, dev or test}
            for img_path in os.listdir(os.path.join(self.img_dir, seal, phase)):
                self.image_paths.append(os.path.join(self.img_dir, seal, phase, img_path))
                if mask_dir is not None:
                    self.mask_paths.append(os.path.join(self.mask_dir, seal, phase, img_path))
                self.labels.append(classes.index(int(seal)))

    # Function that returns the number of datapoints in the dataset
    def __len__(self):
        return len(self.labels)

    # Function that is called to grab 1 datapoint from the dataset
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if self.mask_dir is not None:
            mask_path = self.mask_paths[idx]
            mask = torch.load(mask_path)
        else: 
            mask = torch.ones([224, 224])
        label = self.labels[idx]
        if(img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))):
            #standardize all photos to RGB so they are equal tensor shapes
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            encoding = self.processor(images=image, return_tensors="pt", do_normalize=False, do_resize=False, do_rescale=False)
            pixel_values = encoding["pixel_values"].squeeze(0)
            return {"image" : image, "label": label, "pixel_values" : pixel_values}
        else:
            print(img_path)
            return None
        
class HFMinMaxDataset(Dataset):
    def __init__(self, img_dir, phase, transform=None, mask_dir=None, processor=None, min_per_class=0, max_per_class=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        if mask_dir is not None:
            self.mask_paths = []
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(self.img_dir)) 
        self.processor = processor
        self.transform = transform

        class_counts = {}
        class_paths = {}

        # First, gather and count images per class
        for seal in os.listdir(self.img_dir):
            seal_dir = os.path.join(self.img_dir, seal, phase)
            if not os.path.isdir(seal_dir): continue

            img_list = [os.path.join(seal_dir, f) for f in os.listdir(seal_dir) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff'))]

            if len(img_list) < min_per_class:
                continue  # skip underrepresented seals

            if max_per_class:
                img_list = img_list[:max_per_class]  # cap the number

            class_paths[seal] = img_list
            class_counts[seal] = len(img_list)

        # Now build the dataset
        for seal, img_list in class_paths.items():
            for img_path in img_list:
                self.image_paths.append(img_path)
                if mask_dir is not None:
                    self.mask_paths.append(os.path.join(self.mask_dir, seal, phase, os.path.basename(img_path)))
                self.labels.append(classes.index(int(seal)))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        if self.mask_dir is not None:
            mask_path = self.mask_paths[idx]
            mask = torch.load(mask_path)
        else: 
            mask = torch.ones([224, 224])
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')
        if self.transform:  
            image = self.transform(image)
        encoding = self.processor(images=image, return_tensors="pt", do_normalize=False, do_resize=False, do_rescale=False)
        pixel_values = encoding["pixel_values"].squeeze(0)
        return {"image": image, "label": label, "pixel_values": pixel_values}

