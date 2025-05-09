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

def load_model(model_name, task, device):
    """Load the appropriate model based on model name and task"""
    # Define number of classes for each task
    n_classes = {
        'age_5': 4,    # 4 age classes
        'disease': 11,  # 11 disease classes
        'gender': 2    # 2 gender classes
    }
    
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
    
    return model

def measure_inference_time(model, test_loader, device, num_warmup=10, num_runs=100):
    """Measure inference time for a model"""
    model.eval()
    
    # Warmup runs
    print("Performing warmup runs...")
    with torch.no_grad():
        for i, (images, _, _) in enumerate(test_loader):
            if i >= num_warmup:
                break
            images = images.to(device)
            _ = model(images)
    
    # Actual timing runs
    print("Measuring inference time...")
    inference_times = []
    
    with torch.no_grad():
        for i, (images, _, _) in enumerate(test_loader):
            if i >= num_runs:
                break
                
            images = images.to(device)
            
            # Synchronize GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Measure time
            start_time = time.time()
            _ = model(images)
            
            # Synchronize GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    return {
        'mean': mean_time,
        'std': std_time,
        'min': min_time,
        'max': max_time,
        'all_times': inference_times
    }

def measure_batch_inference_time(model, test_loader, device, batch_size=32, num_warmup=10, num_runs=100):
    """Measure inference time for a model with different batch sizes"""
    model.eval()
    
    # Create a new dataloader with the specified batch size
    batch_loader = DataLoader(
        test_loader.dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Warmup runs
    print(f"Performing warmup runs with batch size {batch_size}...")
    with torch.no_grad():
        for i, (images, _, _) in enumerate(batch_loader):
            if i >= num_warmup:
                break
            images = images.to(device)
            _ = model(images)
    
    # Actual timing runs
    print(f"Measuring batch inference time with batch size {batch_size}...")
    inference_times = []
    
    with torch.no_grad():
        for i, (images, _, _) in enumerate(batch_loader):
            if i >= num_runs:
                break
                
            images = images.to(device)
            
            # Synchronize GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            # Measure time
            start_time = time.time()
            _ = model(images)
            
            # Synchronize GPU
            if device.type == 'cuda':
                torch.cuda.synchronize()
            
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000)  # Convert to milliseconds
    
    # Calculate statistics
    mean_time = np.mean(inference_times)
    std_time = np.std(inference_times)
    min_time = np.min(inference_times)
    max_time = np.max(inference_times)
    
    # Calculate per-image time
    mean_time_per_image = mean_time / batch_size
    std_time_per_image = std_time / batch_size
    
    return {
        'mean_batch_ms': mean_time,
        'std_batch_ms': std_time,
        'min_batch_ms': min_time,
        'max_batch_ms': max_time,
        'mean_per_image_ms': mean_time_per_image,
        'std_per_image_ms': std_time_per_image,
        'all_times': inference_times
    }

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Define models and tasks to test
    models = ['ours', 'cait_tiny', 'efficientnet', 'resnet18', 'resnext', 'swin_tiny', 'vitsmall']
    tasks = ['age_5', 'gender', 'disease']
    
    # Create results directory
    results_dir = 'inference_time_results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Store all results
    all_results = {}
    
    # Test each model and task combination
    for model_name in models:
        print(f"\nTesting model: {model_name}")
        model_results = {}
        
        for task in tasks:
            print(f"\nTesting task: {task}")
            
            # Get dataloader
            _, _, test_loader, train_dataset, _, _, _ = get_face_dataloaders(
                data_dir='./data/face',
                batch_size=1,  # Use batch size 1 for per-image timing
                num_workers=4,
                task=task,
                resize=32
            )
            
            # Load model
            model = load_model(model_name, task, device)
            
            # Measure per-image inference time
            timing_results = measure_inference_time(model, test_loader, device)
            
            # Measure batch inference time
            batch_sizes = [8, 16, 32, 64]
            batch_results = {}
            
            for batch_size in batch_sizes:
                print(f"\nTesting batch size: {batch_size}")
                batch_timing = measure_batch_inference_time(model, test_loader, device, batch_size)
                batch_results[batch_size] = {
                    'mean_batch_ms': batch_timing['mean_batch_ms'],
                    'std_batch_ms': batch_timing['std_batch_ms'],
                    'mean_per_image_ms': batch_timing['mean_per_image_ms'],
                    'std_per_image_ms': batch_timing['std_per_image_ms']
                }
            
            # Store results
            model_results[task] = {
                'per_image': {
                    'mean_ms': timing_results['mean'],
                    'std_ms': timing_results['std'],
                    'min_ms': timing_results['min'],
                    'max_ms': timing_results['max']
                },
                'batch_results': batch_results
            }
            
            print(f"\nResults for {model_name} on {task}:")
            print("Per-image inference time:")
            print(f"Mean: {timing_results['mean']:.2f} ms")
            print(f"Std deviation: {timing_results['std']:.2f} ms")
            print(f"Min: {timing_results['min']:.2f} ms")
            print(f"Max: {timing_results['max']:.2f} ms")
            
            print("\nBatch inference times:")
            for batch_size, results in batch_results.items():
                print(f"\nBatch size {batch_size}:")
                print(f"Mean batch time: {results['mean_batch_ms']:.2f} ms")
                print(f"Mean per image: {results['mean_per_image_ms']:.2f} ms")
        
        all_results[model_name] = model_results
    
    # Save results to JSON file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = os.path.join(results_dir, f'inference_times_{timestamp}.json')
    
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nResults saved to: {results_file}")
    
    # Print summary tables
    print("\nSummary of Per-Image Inference Times (mean ms per image):")
    print("\nModel\t\tAge\t\tGender\t\tDisease")
    print("-" * 60)
    
    for model_name in models:
        row = f"{model_name:<15}"
        for task in tasks:
            mean_time = all_results[model_name][task]['per_image']['mean_ms']
            row += f"{mean_time:>8.2f} ms\t"
        print(row)
    
    # Print batch size summary
    print("\nSummary of Batch Inference Times (mean ms per image):")
    for batch_size in batch_sizes:
        print(f"\nBatch Size: {batch_size}")
        print("\nModel\t\tAge\t\tGender\t\tDisease")
        print("-" * 60)
        
        for model_name in models:
            row = f"{model_name:<15}"
            for task in tasks:
                mean_time = all_results[model_name][task]['batch_results'][batch_size]['mean_per_image_ms']
                row += f"{mean_time:>8.2f} ms\t"
            print(row)

if __name__ == '__main__':
    main() 