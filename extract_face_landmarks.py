import os
import pandas as pd
import cv2
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm
import pkg_resources
import spiga.demo.analyze.track.get_tracker as tr
import spiga.demo.analyze.extract.spiga_processor as pr_spiga
from spiga.demo.analyze.analyzer import VideoAnalyzer

def load_spiga_model(spiga_dataset='wflw', tracker='RetinaSort'):
    """Load SPIGA model and tracker once."""
    print("📥🧠 Loading SPIGA model and tracker...")

    # Initialize face tracker
    faces_tracker = tr.get_tracker(tracker)
    print(f"🕵️‍♂️🤖 Tracker '{tracker}' initialized.")

    # Initialize processors
    processor = pr_spiga.SPIGAProcessor(dataset=spiga_dataset)
    print(f"🛠️⚙️ SPIGA processor set up for dataset: {spiga_dataset}")

    # Create analyzer instance
    analyzer = VideoAnalyzer(faces_tracker, processor=processor)
    print("✅🧠 Model and tracker loaded successfully!")

    return analyzer

def process_image_with_landmarks(analyzer, image_path):
    """Process a single image and extract facial landmarks."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"❌🚫 Error: Image not found or invalid path: {image_path}")
        return None

    # Set input shape dynamically
    analyzer.tracker.detector.set_input_shape(image.shape[1], image.shape[0])

    # Process image
    processed_image = image.copy()
    analyzer.process_frame(processed_image)

    # Extract landmarks
    landmarks = []
    for obj in analyzer.tracked_obj:
        landmarks.append(obj.get_attributes('landmarks'))

    return landmarks

def extract_landmarks_from_dataset(data_dir='./data/face', batch_size=32):
    """
    Extract facial landmarks from all images in the dataset.
    
    Args:
        data_dir (str): Directory containing images and CSV
        batch_size (int): Number of images to process in each batch
    
    Returns:
        pd.DataFrame: Original dataframe with added landmark columns
    """
    # Load CSV
    csv_path = os.path.join(data_dir, 'face_images_path_with_meta_jpg_exist_only.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Load SPIGA model
    analyzer = load_spiga_model()
    
    # Initialize lists to store landmarks
    all_landmarks = []
    processed_files = []
    
    # Process images in batches
    for idx in tqdm(range(0, len(df), batch_size), desc="Processing images"):
        batch_df = df.iloc[idx:idx + batch_size]
        
        for _, row in batch_df.iterrows():
            img_filename = row['dest_filename']
            img_path = os.path.join(data_dir, img_filename)
            
            if not os.path.exists(img_path):
                print(f"Warning: Image not found at {img_path}")
                all_landmarks.append(None)
                processed_files.append(img_filename)
                continue
            
            # Process image and get landmarks
            landmarks = process_image_with_landmarks(analyzer, img_path)
            
            if landmarks and len(landmarks) > 0:
                # Take the first face's landmarks if multiple faces detected
                landmarks = landmarks[0]
                # Convert landmarks to a flat list of coordinates
                flat_landmarks = [coord for point in landmarks for coord in point]
                all_landmarks.append(flat_landmarks)
            else:
                all_landmarks.append(None)
            
            processed_files.append(img_filename)
    
    # Add landmarks to dataframe
    df['landmarks'] = all_landmarks
    
    # Save results
    output_csv = os.path.join(data_dir, 'face_images_with_landmarks.csv')
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    
    return df

if __name__ == '__main__':
    # Example usage
    df = extract_landmarks_from_dataset()
    print(f"Processed {len(df)} images")
    print(f"Number of images with landmarks: {df['landmarks'].notna().sum()}") 