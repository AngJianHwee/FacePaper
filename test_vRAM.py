import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.get_face_dataloaders import get_face_dataloaders
from models.ours.AttentionMobileNetShallow_s_three_task import AttentionMobileNetShallow_s_three_task
from models.cait_32x32 import cait_tiny
from models.efficientnet_32x32 import EfficientNetB0
from models.resnet_32x32 import ResNet18
from models.resnext_32x32 import ResNeXt29_2x64d
from models.swin_32x32 import swin_tiny
from models.vitsmall_32x32 import ViTSmall
import time
import numpy as np
from tqdm import tqdm
import os
import json
import logging
from datetime import datetime

def setup_logging():
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f'vram_usage_{timestamp}.log'
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # This will also print to console
        ]
    )
    return log_file

def get_gpu_memory_usage():
    """Get current GPU memory usage in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024**2  # Convert to MB
    return 0

def get_gpu_memory_reserved():
    """Get current GPU memory reserved in MB"""
    if torch.cuda.is_available():
        return torch.cuda.memory_reserved() / 1024**2  # Convert to MB
    return 0

def measure_model_memory(model, device, model_name=None, task=None):
    """Measure the memory footprint of a loaded model"""
    context = f"[{model_name}][{task}]" if model_name and task else ""
    logging.info(f"{context} Measuring model memory footprint")
    
    # Clear GPU cache before measurement
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = get_gpu_memory_usage()
        logging.debug(f"{context} Initial GPU memory: {initial_memory:.2f} MB")
    
    # Move model to device
    model = model.to(device)
    
    # Get model memory usage
    if device.type == 'cuda':
        model_memory = get_gpu_memory_usage() - initial_memory
        model_reserved = get_gpu_memory_reserved() - initial_memory
        
        logging.info(f"{context} Model memory usage: {model_memory:.2f} MB")
        logging.info(f"{context} Model memory reserved: {model_reserved:.2f} MB")
        
        return {
            'model_memory_mb': model_memory,
            'model_reserved_mb': model_reserved
        }
    return {
        'model_memory_mb': 0,
        'model_reserved_mb': 0
    }

def load_model(model_name, task, device):
    """Load the appropriate model based on model name and task"""
    logging.info(f"[{model_name}][{task}] Loading model")
    
    # Define number of classes for each task
    n_classes = {
        'age_5': 4,    # 4 age classes
        'disease': 11,  # 11 disease classes
        'gender': 2    # 2 gender classes
    }
    
    logging.debug(f"[{model_name}][{task}] Number of classes: {n_classes[task]}")
    
    if model_name == 'ours':
        model = AttentionMobileNetShallow_s_three_task(
            input_channels=3,
            n_classes_task1=n_classes['age_5'],     # age
            n_classes_task2=n_classes['gender'],    # gender
            n_classes_task3=n_classes['disease'],   # disease
            input_size=32,
            use_attention=True
        )
    elif model_name == 'cait_tiny':
        model = cait_tiny(num_classes=n_classes[task])
    elif model_name == 'efficientnet':
        model = EfficientNetB0(num_classes=n_classes[task])
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=n_classes[task])
    elif model_name == 'resnext':
        model = ResNeXt29_2x64d(num_classes=n_classes[task])
    elif model_name == 'swin_tiny':
        model = swin_tiny(num_classes=n_classes[task])
    elif model_name == 'vitsmall':
        model = ViTSmall(num_classes=n_classes[task])
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    logging.info(f"[{model_name}][{task}] Successfully loaded model")
    return model

def main():
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting model memory measurement script")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    if device.type == 'cuda':
        logging.info(f'GPU: {torch.cuda.get_device_name(0)}')
        logging.info(f'Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2:.2f} MB')
    
    # Define models and tasks to test
    models = ['ours', 'cait_tiny', 'efficientnet', 'resnet18', 'resnext', 'swin_tiny', 'vitsmall']
    tasks = ['age_5', 'gender', 'disease']
    logging.info(f"Testing models: {models}")
    logging.info(f"Testing tasks: {tasks}")
    
    # Create results directory
    results_dir = 'model_memory_results'
    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Created results directory: {results_dir}")
    
    # Store all results
    all_results = {}
    
    # Test each model and task combination
    for model_name in models:
        logging.info(f"\n{'='*50}")
        logging.info(f"Testing model: {model_name}")
        logging.info(f"{'='*50}")
        model_results = {}
        
        for task in tasks:
            logging.info(f"\n{'-'*30}")
            logging.info(f"Testing task: {task}")
            logging.info(f"{'-'*30}")
            
            # Load model
            model = load_model(model_name, task, device)
            
            # Measure model memory
            memory_results = measure_model_memory(model, device, model_name=model_name, task=task)
            
            # Store results
            model_results[task] = {
                'model_memory_mb': memory_results['model_memory_mb'],
                'model_reserved_mb': memory_results['model_reserved_mb']
            }
            
            logging.info(f"\n[{model_name}][{task}] Results summary:")
            logging.info(f"[{model_name}][{task}] Model memory: {memory_results['model_memory_mb']:.2f} MB")
            logging.info(f"[{model_name}][{task}] Model reserved: {memory_results['model_reserved_mb']:.2f} MB")
        
        all_results[model_name] = model_results
    
    # Save results to JSON file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'model_memory_{timestamp}.json')
    
    logging.info(f"\nSaving results to: {results_file}")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logging.info(f"Results saved successfully")
    logging.info(f"Log file saved to: {log_file}")
    
    # Print summary table
    logging.info("\nSummary of Model Memory Usage (MB):")
    logging.info("\nModel\t\tAge\t\tGender\t\tDisease")
    logging.info("-" * 60)
    
    for model_name in models:
        row = f"{model_name:<15}"
        for task in tasks:
            memory = all_results[model_name][task]['model_memory_mb']
            row += f"{memory:>8.2f} MB\t"
        logging.info(row)
    
    logging.info("\nScript execution completed successfully")

if __name__ == '__main__':
    main() 