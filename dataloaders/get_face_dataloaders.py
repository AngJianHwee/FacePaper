import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import torchvision.transforms as transforms

def get_face_dataloaders(data_dir='./data/face', batch_size=64, num_workers=4, task='all', resize=224):
    """
    Load face dataset from CSV for multiple tasks.
    
    Args:
        data_dir (str): Directory containing images and CSV.
        batch_size (int): Batch size for DataLoaders.
        num_workers (int): Number of workers for DataLoader.
        task (str): Task to train on ('all', 'gender', 'age_10', 'age_5', 'disease').
        resize (int): Size to resize images to.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, df)
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
            self.transform = transform
            self.image_folder = image_folder
            
            # Initialize label mappings for all tasks
            self.label_mappings = {}
            
            # Gender labels
            gender_labels = sorted(self.dataframe['gender'].unique())
            self.label_mappings['gender'] = {label: idx for idx, label in enumerate(gender_labels)}
            
            # Age labels (both 5 and 10 year divisions)
            age_10_labels = sorted(self.dataframe['age_div_10_round'].unique())
            self.label_mappings['age_10'] = {label: idx for idx, label in enumerate(age_10_labels)}
            
            age_5_labels = sorted(self.dataframe['age_div_5_round'].unique())
            self.label_mappings['age_5'] = {label: idx for idx, label in enumerate(age_5_labels)}
            
            # Disease labels
            disease_labels = sorted(self.dataframe['disease'].unique())
            self.label_mappings['disease'] = {label: idx for idx, label in enumerate(disease_labels)}
            
            print(f"Number of classes - Gender: {len(self.label_mappings['gender'])}, "
                  f"Age (10): {len(self.label_mappings['age_10'])}, "
                  f"Age (5): {len(self.label_mappings['age_5'])}, "
                  f"Disease: {len(self.label_mappings['disease'])}")

        def __len__(self):
            return len(self.dataframe)

        def __getitem__(self, idx):
            row = self.dataframe.iloc[idx]
            img_filename = row['dest_filename']
            img_path = os.path.join(self.image_folder, img_filename)
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found at: {img_path}")
            
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get all labels
            labels = {
                'gender': self.label_mappings['gender'][row['gender']],
                'age_10': self.label_mappings['age_10'][row['age_div_10_round']],
                'age_5': self.label_mappings['age_5'][row['age_div_5_round']],
                'disease': self.label_mappings['disease'][row['disease']]
            }
            
            # Add metadata
            metadata = {
                'filename': img_filename,
                'subject': row['subject']
            }
            
            return image, labels, metadata

    # Create datasets
    train_dataset = CustomDataset(train_df, task, data_dir, train_transform)
    val_dataset = CustomDataset(val_df, task, data_dir, train_transform)
    test_dataset = CustomDataset(test_df, task, data_dir, test_transform)

    print(f"Train dataset size: {len(train_dataset)}, transform: {train_transform}")
    print(f"Validation dataset size: {len(val_dataset)}, transform: {train_transform}")
    print(f"Test dataset size: {len(test_dataset)}, transform: {test_transform}")
    
    print(f"Train dataset label mapping: {train_dataset.label_mappings}")
    print(f"Validation dataset label mapping: {val_dataset.label_mappings}")
    print(f"Test dataset label mapping: {test_dataset.label_mappings}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # try iter one batch and print size
    try:
        for images, labels, metadata in train_loader:
            print(f"Train batch size: {images.size()}")
            print("Labels shape:")
            for task, label in labels.items():
                print(f"{task}: {label.size()}")
            print("Metadata sample:", {k: v[0] for k, v in metadata.items()})
            break
        for images, labels, metadata in val_loader:
            print(f"Validation batch size: {images.size()}")
            print("Labels shape:")
            for task, label in labels.items():
                print(f"{task}: {label.size()}")
            print("Metadata sample:", {k: v[0] for k, v in metadata.items()})
            break
        for images, labels, metadata in test_loader:
            print(f"Test batch size: {images.size()}")
            print("Labels shape:")
            for task, label in labels.items():
                print(f"{task}: {label.size()}")
            print("Metadata sample:", {k: v[0] for k, v in metadata.items()})
            break
    except Exception as e:
        print(f"Error during DataLoader iteration: {e}")

    # reset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, df