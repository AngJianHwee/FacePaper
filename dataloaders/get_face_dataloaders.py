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


# Columns:
#     subject
#     disease
#     disease_chinese
#     gender
#     age
#     age_div_10
#     age_div_10_round
#     age_div_5
#     age_div_5_round
#     path
#     split
#     dest_filename
#     dest_path

# Example:
#     L00005565
#     Hyperlipidemia
#     高脂血症
#     M
#     36.0
#     3.6
#     4.0
#     7.2
#     7.0
#     D:\所有整理数据\四诊分类完整数据\第10批完整版20140805\L00005565\20140723_T_TJ00271121_陈航平_dev1\face\calibrate_face_1.bmp
#     test
#     L00005565__Hyperlipidemia__calibrate_face_1.jpg
#     C:/Users/megah/Dropbox/Prompt/self_attention_face/L00005565__Hyperlipidemia__calibrate_face_1.jpg


def get_face_dataloaders_subj_id(data_dir='./data/face', batch_size=64, num_workers=4, task='all', resize=224):
    """
    """
    # Load CSV and split (assuming no explicit val split, we'll create one)
    csv_path = os.path.join(data_dir, 'face_images_path_with_meta_jpg_exist_only.csv')
    if not os.path.exists(csv_path):
        csv_full_path = os.path.abspath(csv_path)
        raise FileNotFoundError(f"CSV not found at: {csv_full_path}")
    
    df = pd.read_csv(csv_path)

    # first I need to group them by subject
        # Columns:
    #     subject
    #     disease
    #     disease_chinese
    #     gender
    #     age
    #     age_div_10
    #     age_div_10_round
    #     age_div_5
    #     age_div_5_round
    #     path
    #     split
    #     dest_filename
    #     dest_path

    # Example:
    #     L00005565
    #     Hyperlipidemia
    #     高脂血症
    #     M
    #     36.0
    #     3.6
    #     4.0
    #     7.2
    #     7.0
    #     D:\所有整理数据\四诊分类完整数据\第10批完整版20140805\L00005565\20140723_T_TJ00271121_陈航平_dev1\face\calibrate_face_1.bmp
    #     test
    #     L00005565__Hyperlipidemia__calibrate_face_1.jpg
    #     C:/Users/megah/Dropbox/Prompt/self_attention_face/L00005565__Hyperlipidemia__calibrate_face_1.jpg



    # group by subject, show count as a count dict
    grouped_df = df.groupby('subject').size().reset_index(name='count')
    # should show how many subject has 3 images, how many has 4 images, etc.
    count_dict = grouped_df['count'].value_counts().to_dict()
    print("Count of subjects by number of images:")
    for count, num_subjects in sorted(count_dict.items()):
        print(f"{count} images: {num_subjects} subjects")
    #     Count of subjects by number of images:
    #     2 images: 1 subjects
    #     3 images: 597 subjects
    #     4 images: 223 subjects
    #     5 images: 69 subjects
    #     6 images: 22 subjects
    #     7 images: 20 subjects
    #     8 images: 7 subjects
    #     9 images: 2 subjects
    #     11 images: 1 subjects


    # Now, for those with 5, 6, 7, 8, 9, 11 images, I want to drop them until they have 4 images
    def drop_images_to_four(df):
        # Group by subject and count images
        grouped = df.groupby('subject').size().reset_index(name='count')
        
        # Filter subjects with more than 4 images
        subjects_to_drop = grouped[grouped['count'] > 4]['subject'].tolist()
        
        # Drop images for those subjects until they have 4
        for subject in subjects_to_drop:
            subject_images = df[df['subject'] == subject]
            if len(subject_images) > 4:
                # Randomly drop images until only 4 remain
                df = df.drop(subject_images.sample(len(subject_images) - 4).index)
        
        return df

    # Apply the function to drop images
    df_reduced = drop_images_to_four(df)
    # Check the new counts
    grouped_reduced_df = df_reduced.groupby('subject').size().reset_index(name='count')
    count_reduced_dict = grouped_reduced_df['count'].value_counts().to_dict()
    print("Count of subjects after reducing to 4 images:")
    for count, num_subjects in sorted(count_reduced_dict.items()):
        print(f"{count} images: {num_subjects} subjects")
    #     Count of subjects after reducing to 4 images:
    #     2 images: 1 subjects
    #     3 images: 597 subjects
    #     4 images: 344 subjects

    # Next step, drop the one with 2 images
    def drop_two_images(df):
        # Identify subjects with exactly 2 images
        subjects_with_two = df.groupby('subject').filter(lambda x: len(x) == 2)['subject'].unique()
        
        # Drop these subjects from the DataFrame
        df = df[~df['subject'].isin(subjects_with_two)]
        
        return df

    # Apply the function to drop subjects with 2 images
    df_final = drop_two_images(df_reduced)
    # Check the final counts
    grouped_final_df = df_final.groupby('subject').size().reset_index(name='count')
    count_final_dict = grouped_final_df['count'].value_counts().to_dict()
    print("Count of subjects after dropping those with 2 images:")
    for count, num_subjects in sorted(count_final_dict.items()):
        print(f"{count} images: {num_subjects} subjects")
    #     Count of subjects after dropping those with 2 images:
    #     3 images: 597 subjects
    #     4 images: 344 subjects

    # Perfect! Now I want to return the train and test df
    # but select it as follows, for both 3 and 4, select 1 into test, remaining into train
    train_df = pd.DataFrame(columns=df_final.columns)
    test_df = pd.DataFrame(columns=df_final.columns)
    for subject in df_final['subject'].unique():
        subject_df = df_final[df_final['subject'] == subject]
        if len(subject_df) == 3:
            # Select 1 for test, 2 for train
            test_df = pd.concat([test_df, subject_df.sample(1)])
            train_df = pd.concat([train_df, subject_df.drop(subject_df.sample(1).index)])
        elif len(subject_df) == 4:
            # Select 1 for test, 3 for train
            test_df = pd.concat([test_df, subject_df.sample(1)])
            train_df = pd.concat([train_df, subject_df.drop(subject_df.sample(1).index)])
        else:
            raise ValueError("Unexpected number of images per subject")
    

    # train_df, test_df


    
    # Create a validation split from train (e.g., 10%)
    val_split = 0
    train_size = int((1 - val_split) * len(train_df))
    
    # add shuffle here with random seed 3407
    train_df = train_df.sample(frac=1, random_state=3407).reset_index(drop=True)
    # val_df = train_df.iloc[train_size:]
    train_df = train_df.iloc[:train_size]
    # print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")
    print(f"Train size: {len(train_df)}, Test size: {len(test_df)}")

    

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

            # one more for subject
            subject_labels = sorted(self.dataframe['subject'].unique())
            self.label_mappings['subject'] = {label: idx for idx, label in enumerate(subject_labels)}
            
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
                'subject': self.label_mappings['subject'][row['subject']],
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
    # val_dataset = CustomDataset(val_df, task, data_dir, train_transform)
    test_dataset = CustomDataset(test_df, task, data_dir, test_transform)

    print(f"Train dataset size: {len(train_dataset)}, transform: {train_transform}")
    # print(f"Validation dataset size: {len(val_dataset)}, transform: {train_transform}")
    print(f"Test dataset size: {len(test_dataset)}, transform: {test_transform}")
    
    print(f"Train dataset label mapping: {train_dataset.label_mappings}")
    # print(f"Validation dataset label mapping: {val_dataset.label_mappings}")
    print(f"Test dataset label mapping: {test_dataset.label_mappings}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
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
        # for images, labels, metadata in val_loader:
        #     print(f"Validation batch size: {images.size()}")
        #     print("Labels shape:")
        #     for task, label in labels.items():
        #         print(f"{task}: {label.size()}")
        #     print("Metadata sample:", {k: v[0] for k, v in metadata.items()})
        #     break
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
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset, test_dataset, df