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
import GPUtil

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

def get_gpu_memory_used():
    """Get current GPU memory used according to nvidia-smi in MB"""
    if torch.cuda.is_available():
        gpu = GPUtil.getGPUs()[0]
        return gpu.memoryUsed  # Already in MB
    return 0

def calculate_model_size(model):
    """Calculate the actual size of model parameters and buffers in MB"""
    total_params = sum(p.numel() for p in model.parameters())
    total_buffers = sum(b.numel() for b in model.buffers())
    
    # Assuming float32 (4 bytes per parameter)
    param_size = total_params * 4 / (1024**2)  # Convert to MB
    buffer_size = total_buffers * 4 / (1024**2)  # Convert to MB
    
    return {
        'total_params': total_params,
        'total_buffers': total_buffers,
        'param_size_mb': param_size,
        'buffer_size_mb': buffer_size,
        'total_size_mb': param_size + buffer_size
    }

def measure_model_memory(model, device, model_name=None, task=None):
    """Measure both the actual model size and GPU memory usage"""
    context = f"[{model_name}][{task}]" if model_name and task else ""
    logging.info(f"{context} Measuring model memory footprint")
    
    # Calculate actual model size (parameters + buffers)
    model_size = calculate_model_size(model)
    logging.info(f"{context} Model size (parameters + buffers): {model_size['total_size_mb']:.2f} MB")
    logging.info(f"{context} Number of parameters: {model_size['total_params']:,}")
    logging.info(f"{context} Number of buffers: {model_size['total_buffers']:,}")
    
    # Clear GPU cache before measurement
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = get_gpu_memory_usage()
        initial_reserved = get_gpu_memory_reserved()
        initial_used = get_gpu_memory_used()
        logging.debug(f"{context} Initial GPU memory - Allocated: {initial_memory:.2f} MB, Reserved: {initial_reserved:.2f} MB, Used: {initial_used:.2f} MB")
    
    # Move model to device
    model = model.to(device)
    
    # Get GPU memory usage
    if device.type == 'cuda':
        allocated_memory = get_gpu_memory_usage() - initial_memory
        reserved_memory = get_gpu_memory_reserved() - initial_reserved
        used_memory = get_gpu_memory_used() - initial_used
        
        logging.info(f"{context} GPU Memory Usage:")
        logging.info(f"{context} - Allocated (torch.cuda.memory_allocated): {allocated_memory:.2f} MB")
        logging.info(f"{context} - Reserved (torch.cuda.memory_reserved): {reserved_memory:.2f} MB")
        logging.info(f"{context} - Used (nvidia-smi): {used_memory:.2f} MB")
        
        return {
            'model_size_mb': model_size['total_size_mb'],
            'allocated_memory_mb': allocated_memory,
            'reserved_memory_mb': reserved_memory,
            'used_memory_mb': used_memory,
            'total_params': model_size['total_params'],
            'total_buffers': model_size['total_buffers']
        }
    return {
        'model_size_mb': model_size['total_size_mb'],
        'allocated_memory_mb': 0,
        'reserved_memory_mb': 0,
        'used_memory_mb': 0,
        'total_params': model_size['total_params'],
        'total_buffers': model_size['total_buffers']
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
            model_results[task] = memory_results
            
            logging.info(f"\n[{model_name}][{task}] Results summary:")
            logging.info(f"[{model_name}][{task}] Model size (parameters + buffers): {memory_results['model_size_mb']:.2f} MB")
            logging.info(f"[{model_name}][{task}] GPU Memory Usage:")
            logging.info(f"[{model_name}][{task}] - Allocated: {memory_results['allocated_memory_mb']:.2f} MB")
            logging.info(f"[{model_name}][{task}] - Reserved: {memory_results['reserved_memory_mb']:.2f} MB")
            logging.info(f"[{model_name}][{task}] - Used (nvidia-smi): {memory_results['used_memory_mb']:.2f} MB")
            
            # Clean up
            del model
            torch.cuda.empty_cache()
        
        all_results[model_name] = model_results
    
    # Save results to JSON file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'model_memory_{timestamp}.json')
    
    logging.info(f"\nSaving results to: {results_file}")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logging.info(f"Results saved successfully")
    logging.info(f"Log file saved to: {log_file}")
    
    # Print summary tables
    logging.info("\nSummary of Model Sizes (MB):")
    logging.info("\nModel\t\tAge\t\tGender\t\tDisease")
    logging.info("-" * 60)
    
    for model_name in models:
        row = f"{model_name:<15}"
        for task in tasks:
            size = all_results[model_name][task]['model_size_mb']
            row += f"{size:>8.2f} MB\t"
        logging.info(row)
    
    logging.info("\nSummary of GPU Memory Usage (MB) - nvidia-smi:")
    logging.info("\nModel\t\tAge\t\tGender\t\tDisease")
    logging.info("-" * 60)
    
    for model_name in models:
        row = f"{model_name:<15}"
        for task in tasks:
            used = all_results[model_name][task]['used_memory_mb']
            row += f"{used:>8.2f} MB\t"
        logging.info(row)
    
    logging.info("\nScript execution completed successfully")

if __name__ == '__main__':
    main() 