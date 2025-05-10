import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.get_face_dataloaders import get_face_dataloaders
from models.efficientnet_32x32 import EfficientNetB0
from utils.optimizers import get_optimizer
from utils.loss import CrossEntropyLoss
import logging
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import datetime
import uuid
import os

# Task-specific configurations
TASK_CONFIGS = {
    'age_5': {
        'num_epochs': 80,  # Example: more epochs for age prediction
    },
    'disease': {
        'num_epochs': 80,   # Example: standard epochs for disease
    },
    'gender': {
        'num_epochs': 80,   # Example: fewer epochs for gender
    }
}

def get_unique_suffix():
    # Get current timestamp and UUID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Get first 8 characters of UUID
    return f"{timestamp}_{unique_id}"

def train_epoch(model, train_loader, optimizer, criterion, device, current_epoch, total_epochs, task):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for images, labels, _ in tqdm(train_loader, desc=f'Training {task}'):
        images = images.to(device)
        labels = labels[task].to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    # Print loss and accuracy
    logging.info(f"\n[{current_epoch}/{total_epochs}] Training {task}:")
    logging.info(f"[{current_epoch}/{total_epochs}] Loss: {avg_loss:.4f}")
    logging.info(f"[{current_epoch}/{total_epochs}] Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, current_epoch, total_epochs, task):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc=f'Validation {task}'):
            images = images.to(device)
            labels = labels[task].to(device)
            
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    
    # Print loss and accuracy
    logging.info(f"\n[{current_epoch}/{total_epochs}] Validation {task}:")
    logging.info(f"[{current_epoch}/{total_epochs}] Loss: {avg_loss:.4f}")
    logging.info(f"[{current_epoch}/{total_epochs}] Accuracy: {accuracy:.2f}%")
    
    return avg_loss, accuracy

def test(model, test_loader, criterion, device, current_epoch=None, total_epochs=None, task=None):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # For detailed analysis
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc=f'Testing {task}'):
            images = images.to(device)
            labels = labels[task].to(device)
            
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            
            # Store predictions and targets for detailed analysis
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total
    
    # Print loss and accuracy
    prefix = f"[{current_epoch}/{total_epochs}]" if current_epoch is not None else ""
    logging.info(f"\n{prefix} Test {task}:")
    logging.info(f"{prefix} Loss: {avg_loss:.4f}")
    logging.info(f"{prefix} Accuracy: {accuracy:.2f}%")
    
    # Calculate confusion matrix
    confusion_mat = confusion_matrix(all_targets, all_predictions)
    
    return avg_loss, accuracy, confusion_mat

def train_task(task, run_suffix, device):
    # Create task-specific directory for models and logs
    task_dir = f'task_{task}_{run_suffix}'
    os.makedirs(task_dir, exist_ok=True)
    
    # Set up logging for this task
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(task_dir, f'training_{task}_EfficientNetB0.log'))
        ]
    )
    
    print(f"\nStarting training for task: {task}")
    logging.info(f"Starting training for task: {task}")
    
    # Get task-specific configuration
    task_config = TASK_CONFIGS[task]
    num_epochs = task_config['num_epochs']
    
    # Get dataloaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, _ = get_face_dataloaders(
        data_dir='./data/face',
        batch_size=32,
        num_workers=4,
        task=task,
        resize=32
    )
    
    # Get number of classes for the task
    n_classes = len(train_dataset.label_mappings[task])
    
    # Initialize EfficientNet model
    model = EfficientNetB0(num_classes=n_classes).to(device)
    print(f"Model: EfficientNetB0 for {task}")
    print(f"Number of epochs: {num_epochs}")
    
    # Initialize optimizer and criterion
    optimizer = get_optimizer(model.parameters(), name='adam', lr=0.001, weight_decay=1e-4)
    criterion = CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch, i in enumerate(range(1, num_epochs + 1), 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, i, num_epochs, task)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, i, num_epochs, task)
        
        # Test after each epoch
        test_loss, test_acc, _ = test(model, test_loader, criterion, device, i, num_epochs, task)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'\n[{i}/{num_epochs}] Epoch completed in {epoch_time:.2f}s')
        print(f'[{i}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'[{i}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'[{i}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        logging.info(f'[{i}/{num_epochs}] Epoch completed in {epoch_time:.2f}s')
        logging.info(f'[{i}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        logging.info(f'[{i}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        logging.info(f'[{i}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_model_path = os.path.join(task_dir, f'best_model_val_EfficientNetB0.pth')
            torch.save(model.state_dict(), val_model_path)
            print(f'[{i}/{num_epochs}] Saved best validation model to {val_model_path}')
            logging.info(f'[{i}/{num_epochs}] Saved best validation model to {val_model_path}')
        
        # Save best model based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            test_model_path = os.path.join(task_dir, f'best_model_test_EfficientNetB0.pth')
            torch.save(model.state_dict(), test_model_path)
            print(f'[{i}/{num_epochs}] Saved best test model to {test_model_path}')
            logging.info(f'[{i}/{num_epochs}] Saved best test model to {test_model_path}')
    
    print(f"\nTraining completed for {task}. Testing both best models...")
    
    # Test best validation model
    print(f"\nTesting best validation model for {task}...")
    val_model_path = os.path.join(task_dir, 'best_model_val_EfficientNetB0.pth')
    model.load_state_dict(torch.load(val_model_path))
    val_test_loss, val_test_acc, val_confusion_mat = test(model, test_loader, criterion, device, task=task)
    
    print(f'\nBest Validation Model Test Results for {task}:')
    print(f'Test Loss: {val_test_loss:.4f}')
    print(f'Test Accuracy: {val_test_acc:.2f}%')
    
    logging.info(f'\nBest Validation Model Test Results for {task}:')
    logging.info(f'Test Loss: {val_test_loss:.4f}')
    logging.info(f'Test Accuracy: {val_test_acc:.2f}%')
    
    # Test best test model
    print(f"\nTesting best test model for {task}...")
    test_model_path = os.path.join(task_dir, 'best_model_test_EfficientNetB0.pth')
    model.load_state_dict(torch.load(test_model_path))
    test_test_loss, test_test_acc, test_confusion_mat = test(model, test_loader, criterion, device, task=task)
    
    print(f'\nBest Test Model Test Results for {task}:')
    print(f'Test Loss: {test_test_loss:.4f}')
    print(f'Test Accuracy: {test_test_acc:.2f}%')
    
    logging.info(f'\nBest Test Model Test Results for {task}:')
    logging.info(f'Test Loss: {test_test_loss:.4f}')
    logging.info(f'Test Accuracy: {test_test_acc:.2f}%')
    
    # Print confusion matrices for both models
    print(f'\nConfusion Matrix for Best Validation Model ({task}):')
    print(val_confusion_mat)
    logging.info(f'\nConfusion Matrix (Best Validation Model) for {task}:')
    logging.info(val_confusion_mat)
    
    print(f'\nConfusion Matrix for Best Test Model ({task}):')
    print(test_confusion_mat)
    logging.info(f'\nConfusion Matrix (Best Test Model) for {task}:')
    logging.info(test_confusion_mat)

def main():
    # Generate unique suffix for this run
    run_suffix = get_unique_suffix()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Print task configurations
    print("\nTask Configurations:")
    for task, config in TASK_CONFIGS.items():
        print(f"{task}: {config['num_epochs']} epochs")
    
    # Train models for each task
    tasks = ['age_5', 'disease', 'gender']
    for task in tasks:
        train_task(task, run_suffix, device)
    
    print(f"\nTraining and testing completed successfully for all tasks! (Run ID: {run_suffix})")

if __name__ == '__main__':
    main() 