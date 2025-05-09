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
    
    # Clear GPU cache before loading model
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = get_gpu_memory_usage()
        logging.debug(f"[{model_name}][{task}] Initial GPU memory: {initial_memory:.2f} MB")
    
    if model_name == 'ours':
        model = AttentionMobileNetShallow_s_three_task(
            input_channels=3,
            n_classes_task1=n_classes['age_5'],     # age
            n_classes_task2=n_classes['gender'],    # gender
            n_classes_task3=n_classes['disease'],   # disease
            input_size=32,
            use_attention=True
        ).to(device)
    elif model_name == 'cait_tiny':
        model = cait_tiny(num_classes=n_classes[task]).to(device)
    elif model_name == 'efficientnet':
        model = EfficientNetB0(num_classes=n_classes[task]).to(device)
    elif model_name == 'resnet18':
        model = ResNet18(num_classes=n_classes[task]).to(device)
    elif model_name == 'resnext':
        model = ResNeXt29_2x64d(num_classes=n_classes[task]).to(device)
    elif model_name == 'swin_tiny':
        model = swin_tiny(num_classes=n_classes[task]).to(device)
    elif model_name == 'vitsmall':
        model = ViTSmall(num_classes=n_classes[task]).to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    if device.type == 'cuda':
        model_memory = get_gpu_memory_usage() - initial_memory
        logging.info(f"[{model_name}][{task}] Model memory usage: {model_memory:.2f} MB")
    
    logging.info(f"[{model_name}][{task}] Successfully loaded model")
    return model

def measure_vram_usage(model, test_loader, device, num_runs=100, model_name=None, task=None):
    """Measure vRAM usage for a model"""
    model.eval()
    context = f"[{model_name}][{task}]" if model_name and task else ""
    logging.info(f"{context} Starting vRAM usage measurement with {num_runs} runs")
    
    # Clear GPU cache before measurement
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = get_gpu_memory_usage()
        logging.debug(f"{context} Initial GPU memory: {initial_memory:.2f} MB")
    
    vram_usage = []
    max_vram_usage = []
    
    with torch.no_grad():
        for i, (images, _, _) in enumerate(test_loader):
            if i >= num_runs:
                break
                
            images = images.to(device)
            
            # Measure memory before forward pass
            if device.type == 'cuda':
                before_memory = get_gpu_memory_usage()
            
            # Forward pass
            _ = model(images)
            
            # Measure memory after forward pass
            if device.type == 'cuda':
                after_memory = get_gpu_memory_usage()
                current_usage = after_memory - before_memory
                vram_usage.append(current_usage)
                max_vram_usage.append(get_gpu_memory_reserved())
            
            if (i + 1) % 10 == 0:
                logging.debug(f"{context} Completed {i + 1}/{num_runs} runs. Current vRAM usage: {current_usage:.2f} MB")
    
    # Calculate statistics
    mean_usage = np.mean(vram_usage)
    std_usage = np.std(vram_usage)
    min_usage = np.min(vram_usage)
    max_usage = np.max(vram_usage)
    max_reserved = np.max(max_vram_usage)
    
    logging.info(f"{context} vRAM usage statistics:")
    logging.info(f"{context} Mean usage: {mean_usage:.2f} MB")
    logging.info(f"{context} Std usage: {std_usage:.2f} MB")
    logging.info(f"{context} Min usage: {min_usage:.2f} MB")
    logging.info(f"{context} Max usage: {max_usage:.2f} MB")
    logging.info(f"{context} Max reserved: {max_reserved:.2f} MB")
    logging.debug(f"{context} Raw vRAM usage: {vram_usage}")
    
    return {
        'mean_mb': mean_usage,
        'std_mb': std_usage,
        'min_mb': min_usage,
        'max_mb': max_usage,
        'max_reserved_mb': max_reserved,
        'all_usage': vram_usage
    }

def measure_batch_vram_usage(model, test_loader, device, batch_size=32, num_runs=100, model_name=None, task=None):
    """Measure vRAM usage for a model with different batch sizes"""
    model.eval()
    context = f"[{model_name}][{task}]" if model_name and task else ""
    logging.info(f"{context} Starting batch vRAM usage measurement with batch size {batch_size}")
    
    # Create a new dataloader with the specified batch size
    batch_loader = DataLoader(
        test_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    logging.debug(f"{context} Created dataloader with batch size {batch_size}")
    
    # Clear GPU cache before measurement
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = get_gpu_memory_usage()
        logging.debug(f"{context} Initial GPU memory: {initial_memory:.2f} MB")
    
    vram_usage = []
    max_vram_usage = []
    
    with torch.no_grad():
        for i, (images, _, _) in enumerate(batch_loader):
            if i >= num_runs:
                break
                
            images = images.to(device)
            
            # Measure memory before forward pass
            if device.type == 'cuda':
                before_memory = get_gpu_memory_usage()
            
            # Forward pass
            _ = model(images)
            
            # Measure memory after forward pass
            if device.type == 'cuda':
                after_memory = get_gpu_memory_usage()
                current_usage = after_memory - before_memory
                vram_usage.append(current_usage)
                max_vram_usage.append(get_gpu_memory_reserved())
            
            if (i + 1) % 5 == 0:
                logging.debug(f"{context} Completed {i + 1}/{num_runs} runs. Current batch vRAM usage: {current_usage:.2f} MB")
    
    # Calculate statistics
    mean_usage = np.mean(vram_usage)
    std_usage = np.std(vram_usage)
    min_usage = np.min(vram_usage)
    max_usage = np.max(vram_usage)
    max_reserved = np.max(max_vram_usage)
    
    # Calculate per-image usage
    mean_usage_per_image = mean_usage / batch_size
    std_usage_per_image = std_usage / batch_size
    
    logging.info(f"{context} Batch vRAM usage statistics for batch size {batch_size}:")
    logging.info(f"{context} Mean batch usage: {mean_usage:.2f} MB")
    logging.info(f"{context} Mean per image: {mean_usage_per_image:.2f} MB")
    logging.info(f"{context} Std batch usage: {std_usage:.2f} MB")
    logging.info(f"{context} Min batch usage: {min_usage:.2f} MB")
    logging.info(f"{context} Max batch usage: {max_usage:.2f} MB")
    logging.info(f"{context} Max reserved: {max_reserved:.2f} MB")
    logging.debug(f"{context} Raw batch vRAM usage: {vram_usage}")
    
    return {
        'mean_batch_mb': mean_usage,
        'std_batch_mb': std_usage,
        'min_batch_mb': min_usage,
        'max_batch_mb': max_usage,
        'max_reserved_mb': max_reserved,
        'mean_per_image_mb': mean_usage_per_image,
        'std_per_image_mb': std_usage_per_image,
        'all_usage': vram_usage
    }

def main():
    # Setup logging
    log_file = setup_logging()
    logging.info("Starting vRAM usage measurement script")
    
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
    results_dir = 'vram_usage_results'
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
            
            # Get dataloader
            logging.info(f"[{model_name}][{task}] Loading dataset...")
            _, _, test_loader, train_dataset, _, _, _ = get_face_dataloaders(
                data_dir='./data/face',
                batch_size=1,  # Use batch size 1 for per-image measurement
                num_workers=4,
                task=task,
                resize=32
            )
            logging.info(f"[{model_name}][{task}] Dataset loaded. Test set size: {len(test_loader.dataset)}")
            
            # Load model
            model = load_model(model_name, task, device)
            
            # Measure per-image vRAM usage
            logging.info(f"\n[{model_name}][{task}] Measuring per-image vRAM usage...")
            vram_results = measure_vram_usage(model, test_loader, device, model_name=model_name, task=task)
            
            # Measure batch vRAM usage
            batch_sizes = [8, 16, 32]
            batch_results = {}
            
            for batch_size in batch_sizes:
                logging.info(f"\n[{model_name}][{task}] Testing batch size: {batch_size}")
                batch_vram = measure_batch_vram_usage(model, test_loader, device, batch_size, model_name=model_name, task=task)
                batch_results[batch_size] = {
                    'mean_batch_mb': batch_vram['mean_batch_mb'],
                    'std_batch_mb': batch_vram['std_batch_mb'],
                    'mean_per_image_mb': batch_vram['mean_per_image_mb'],
                    'std_per_image_mb': batch_vram['std_per_image_mb'],
                    'max_reserved_mb': batch_vram['max_reserved_mb']
                }
            
            # Store results
            model_results[task] = {
                'per_image': {
                    'mean_mb': vram_results['mean_mb'],
                    'std_mb': vram_results['std_mb'],
                    'min_mb': vram_results['min_mb'],
                    'max_mb': vram_results['max_mb'],
                    'max_reserved_mb': vram_results['max_reserved_mb']
                },
                'batch_results': batch_results
            }
            
            logging.info(f"\n[{model_name}][{task}] Results summary:")
            logging.info(f"[{model_name}][{task}] Per-image vRAM usage:")
            logging.info(f"[{model_name}][{task}] Mean: {vram_results['mean_mb']:.2f} MB")
            logging.info(f"[{model_name}][{task}] Std deviation: {vram_results['std_mb']:.2f} MB")
            logging.info(f"[{model_name}][{task}] Min: {vram_results['min_mb']:.2f} MB")
            logging.info(f"[{model_name}][{task}] Max: {vram_results['max_mb']:.2f} MB")
            logging.info(f"[{model_name}][{task}] Max reserved: {vram_results['max_reserved_mb']:.2f} MB")
            
            logging.info(f"\n[{model_name}][{task}] Batch vRAM usage:")
            for batch_size, results in batch_results.items():
                logging.info(f"\n[{model_name}][{task}] Batch size {batch_size}:")
                logging.info(f"[{model_name}][{task}] Mean batch usage: {results['mean_batch_mb']:.2f} MB")
                logging.info(f"[{model_name}][{task}] Mean per image: {results['mean_per_image_mb']:.2f} MB")
                logging.info(f"[{model_name}][{task}] Max reserved: {results['max_reserved_mb']:.2f} MB")
        
        all_results[model_name] = model_results
    
    # Save results to JSON file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'vram_usage_{timestamp}.json')
    
    logging.info(f"\nSaving results to: {results_file}")
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    logging.info(f"Results saved successfully")
    logging.info(f"Log file saved to: {log_file}")
    
    # Print summary tables
    logging.info("\nSummary of Per-Image vRAM Usage (mean MB per image):")
    logging.info("\nModel\t\tAge\t\tGender\t\tDisease")
    logging.info("-" * 60)
    
    for model_name in models:
        row = f"{model_name:<15}"
        for task in tasks:
            mean_usage = all_results[model_name][task]['per_image']['mean_mb']
            row += f"{mean_usage:>8.2f} MB\t"
        logging.info(row)
    
    # Print batch size summary
    logging.info("\nSummary of Batch vRAM Usage (mean MB per image):")
    for batch_size in batch_sizes:
        logging.info(f"\nBatch Size: {batch_size}")
        logging.info("\nModel\t\tAge\t\tGender\t\tDisease")
        logging.info("-" * 60)
        
        for model_name in models:
            row = f"{model_name:<15}"
            for task in tasks:
                mean_usage = all_results[model_name][task]['batch_results'][batch_size]['mean_per_image_mb']
                row += f"{mean_usage:>8.2f} MB\t"
            logging.info(row)
    
    logging.info("\nScript execution completed successfully")

if __name__ == '__main__':
    main() 