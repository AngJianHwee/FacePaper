'''
 -*- coding: utf-8 -*-
 @Author: Ang Jian Hwee <angjianhwee@gmail.com>
 @Date:   2025-05-06 18:25:22
 @Last Modified by:   Ang Jian Hwee <angjianhwee@gmail.com>
 @Last Modified time: 2025-05-06 18:25:27
'''
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

def get_face_dataloaders(data_dir='./data/face', batch_size=64, num_workers=4, task='gender', resize=224):
    """
    Load face dataset from CSV for a specific task.
    
    Args:
        data_dir (str): Directory containing images and CSV.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoader.
        task (str): Task to train on ('gender', 'age_10', 'age_5', 'disease').
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load CSV and split (assuming no explicit val split, we'll create one)
    csv_path = os.path.join(data_dir, 'face_images_path_with_meta_jpg_exist_only.csv')
    if not os.path.exists(csv_path):
        csv_full_path = os.path.abspath(csv_path)
        raise FileNotFoundError(f"CSV not found at: {csv_full_path}")
    
    df = pd.read_csv(csv_path)
    train_df = df[df['split'] == 'train']
    test_df = df[df['split'] == 'test']
    
    # Create a validation split from train (e.g., 10%)
    val_split = 0.1
    train_size = int((1 - val_split) * len(train_df))
    
    # add shuffle here with random seed 3407
    train_df = train_df.sample(frac=1, random_state=3407).reset_index(drop=True)
    val_df = train_df.iloc[train_size:]
    train_df = train_df.iloc[:train_size]
    print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

    

    if resize != 224:
        # Define transformations
        train_transform = transforms.Compose([
            transforms.Resize((resize + int(0.1*resize), resize + int(0.1*resize))),  # Resize to 110% of target size
            transforms.RandomCrop((resize,resize)),            # Crop to target size with augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((resize,resize)),                # Resize directly to target size
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    elif resize == 224:
        # Define transformations
        train_transform = transforms.Compose([
            transforms.Resize((256,256)),                # Resize to 256x256 first
            transforms.RandomCrop((224,224)),            # Crop to 224x224 with augmentation
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        test_transform = transforms.Compose([
            transforms.Resize((224,224)),                # Resize directly to 224x224
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        raise ValueError("Resize not specified")

    # Custom Dataset class
    class CustomDataset(Dataset):
        def __init__(self, dataframe, task, image_folder, transform=None):
            self.dataframe = dataframe
            self.task = task
            self.transform = transform
            self.image_folder = image_folder
            
            valid_tasks = ['gender', 'age_10', 'age_5', 'disease']
            if task not in valid_tasks:
                raise ValueError(f"Task must be one of {valid_tasks}, got {task}")

            if self.task == 'gender':
                self.label_col = 'gender'
            elif self.task == 'age_10':
                self.label_col = 'age_div_10_round'
            elif self.task == 'age_5':
                self.label_col = 'age_div_5_round'
            elif self.task == 'disease':
                self.label_col = 'disease'
            
            unique_labels = sorted(self.dataframe[self.label_col].unique())
            self.label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            print(f"Task: {self.task}, Number of classes: {len(self.label_to_idx)}")

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            img_filename = self.dataframe.iloc[idx]['dest_filename']
            img_path = os.path.join(self.image_folder, img_filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            label = self.label_to_idx[self.dataframe.iloc[idx][self.label_col]]
            return image, label

    # Create datasets
    train_dataset = CustomDataset(train_df, task, data_dir, train_transform)
    val_dataset = CustomDataset(val_df, task, data_dir, train_transform)
    test_dataset = CustomDataset(test_df, task, data_dir, test_transform)

    print(f"Train dataset size: {len(train_dataset)}, transform: {train_transform}")
    print(f"Validation dataset size: {len(val_dataset)}, transform: {train_transform}")
    print(f"Test dataset size: {len(test_dataset)}, transform: {test_transform}")
    
    print(f"Train dataset label mapping: {train_dataset.label_to_idx}")
    print(f"Validation dataset label mapping: {val_dataset.label_to_idx}")
    print(f"Test dataset label mapping: {test_dataset.label_to_idx}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # try iter one batch and print size
    try:
        for images, labels in train_loader:
            print(f"Train batch size: {images.size()}, Labels: {labels.size()}")
            break
        for images, labels in val_loader:
            print(f"Validation batch size: {images.size()}, Labels: {labels.size()}")
            break
        for images, labels in test_loader:
            print(f"Test batch size: {images.size()}, Labels: {labels.size()}")
            break
    except Exception as e:
        print(f"Error during DataLoader iteration: {e}")

    # reset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, df