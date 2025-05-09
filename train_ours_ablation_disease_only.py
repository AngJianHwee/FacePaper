import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.get_face_dataloaders import get_face_dataloaders
from models.ours.AttentionMobileNetShallow_s_single_task import AttentionMobileNetShallow_s_single_task
from utils.optimizers import get_optimizer
from utils.loss import CrossEntropyLoss
import logging
import time
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import datetime
import uuid

def get_unique_suffix():
    # Get current timestamp and UUID
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = str(uuid.uuid4())[:8]  # Get first 8 characters of UUID
    return f"{timestamp}_{unique_id}"

def train_epoch(model, train_loader, optimizer, criterion, device, current_epoch, total_epochs):
    model.train()
    total_loss = 0
    total_loss_disease = 0
    correct = {'disease': 0}
    total = {'disease': 0}
    
    for images, labels, _ in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss for disease task
        loss_disease = criterion(outputs, labels['disease'])
        
        # Total loss is just disease loss
        loss = loss_disease
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_disease += loss_disease.item()
        
        # Calculate accuracy for disease task
        _, predicted_disease = outputs.max(1)
        
        correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
        total['disease'] += labels['disease'].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    avg_loss_disease = total_loss_disease / len(train_loader)
    
    accuracy = {
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    # Print loss magnitudes
    logging.info(f"\n[{current_epoch}/{total_epochs}] Loss magnitudes:")
    logging.info(f"[{current_epoch}/{total_epochs}] Disease loss: {avg_loss_disease:.4f}")
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, current_epoch, total_epochs):
    model.eval()
    total_loss = 0
    total_loss_disease = 0
    correct = {'disease': 0}
    total = {'disease': 0}
    
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            outputs = model(images)
            
            # Calculate loss for disease task
            loss_disease = criterion(outputs, labels['disease'])
            
            # Total loss is just disease loss
            loss = loss_disease
            
            total_loss += loss.item()
            total_loss_disease += loss_disease.item()
            
            # Calculate accuracy for disease task
            _, predicted_disease = outputs.max(1)
            
            correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
            total['disease'] += labels['disease'].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    avg_loss_disease = total_loss_disease / len(val_loader)
    
    accuracy = {
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    # Print loss magnitudes
    logging.info(f"\n[{current_epoch}/{total_epochs}] Validation loss magnitudes:")
    logging.info(f"[{current_epoch}/{total_epochs}] Disease loss: {avg_loss_disease:.4f}")
    
    return avg_loss, accuracy

def test(model, test_loader, criterion, device, current_epoch=None, total_epochs=None):
    model.eval()
    total_loss = 0
    total_loss_disease = 0
    correct = {'disease': 0}
    total = {'disease': 0}
    
    # For detailed analysis
    all_predictions = {'disease': []}
    all_targets = {'disease': []}
    
    # For timing measurements
    total_inference_time = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images)
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            num_batches += 1
            
            # Calculate loss for disease task
            loss_disease = criterion(outputs, labels['disease'])
            
            # Total loss is just disease loss
            loss = loss_disease
            
            total_loss += loss.item()
            total_loss_disease += loss_disease.item()
            
            # Calculate accuracy for disease task
            _, predicted_disease = outputs.max(1)
            
            # Store predictions and targets for detailed analysis
            all_predictions['disease'].extend(predicted_disease.cpu().numpy())
            all_targets['disease'].extend(labels['disease'].cpu().numpy())
            
            correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
            total['disease'] += labels['disease'].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    avg_loss_disease = total_loss_disease / len(test_loader)
    
    # Calculate average inference time
    avg_inference_time = total_inference_time / num_batches
    
    accuracy = {
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    # Print loss magnitudes
    prefix = f"[{current_epoch}/{total_epochs}]" if current_epoch is not None else ""
    logging.info(f"\n{prefix} Test loss magnitudes:")
    logging.info(f"{prefix} Disease loss: {avg_loss_disease:.4f}")
    
    # Calculate confusion matrix
    confusion_matrices = {
        'disease': confusion_matrix(
            all_targets['disease'],
            all_predictions['disease']
        )
    }
    
    return avg_loss, accuracy, confusion_matrices, avg_inference_time

def main():
    # Generate unique suffix for this run
    run_suffix = get_unique_suffix()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Log to stdout
            logging.FileHandler(f'training_{run_suffix}.log')  # Log to file with unique name
        ]
    )
    
    print(f"Starting training process... (Run ID: {run_suffix})")
    logging.info(f"Starting training process... (Run ID: {run_suffix})")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    print(f'Using device: {device}')
    
    # Get dataloaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, _ = get_face_dataloaders(
        data_dir='./data/face',
        batch_size=32,
        num_workers=4,
        task='disease',  # Changed to only load disease task
        resize=32
    )
    
    # Get number of classes for disease task
    n_classes_disease = len(train_dataset.label_mappings['disease'])
    
    # Initialize model
    model = AttentionMobileNetShallow_s_single_task(
        input_channels=3,
        n_classes=n_classes_disease,
        input_size=32,
        use_attention=True
    ).to(device)
    print(f"Model: AttentionMobileNetShallow_s_single_task")
    
    # Initialize optimizer and criterion
    optimizer = get_optimizer(model.parameters(), name='adam', lr=0.001, weight_decay=1e-4)
    criterion = CrossEntropyLoss()
    
    # Training loop
    num_epochs = 80
    best_val_loss = float('inf')
    best_test_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch, i in enumerate(range(1, num_epochs + 1), 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, i, num_epochs)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device, i, num_epochs)
        
        # Test after each epoch
        test_loss, test_acc, test_confusion_matrices, test_avg_inference_time = test(model, test_loader, criterion, device, i, num_epochs)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'\n[{i}/{num_epochs}] Epoch completed in {epoch_time:.2f}s')
        print(f'[{i}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc - Disease: {train_acc["disease"]:.2f}%')
        print(f'[{i}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Acc - Disease: {val_acc["disease"]:.2f}%')
        print(f'[{i}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Acc - Disease: {test_acc["disease"]:.2f}%')
        
        logging.info(f'[{i}/{num_epochs}] Epoch completed in {epoch_time:.2f}s')
        logging.info(f'[{i}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc - Disease: {train_acc["disease"]:.2f}%')
        logging.info(f'[{i}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Acc - Disease: {val_acc["disease"]:.2f}%')
        logging.info(f'[{i}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Acc - Disease: {test_acc["disease"]:.2f}%')
        
        # Save best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            val_model_path = f'best_model_val_{run_suffix}.pth'
            torch.save(model.state_dict(), val_model_path)
            print(f'[{i}/{num_epochs}] Saved best validation model to {val_model_path}')
            logging.info(f'[{i}/{num_epochs}] Saved best validation model to {val_model_path}')
        
        # Save best model based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            test_model_path = f'best_model_test_{run_suffix}.pth'
            torch.save(model.state_dict(), test_model_path)
            print(f'[{i}/{num_epochs}] Saved best test model to {test_model_path}')
            logging.info(f'[{i}/{num_epochs}] Saved best test model to {test_model_path}')
    
    print("\nTraining completed. Testing both best models...")
    
    # Test best validation model
    print("\nTesting best validation model...")
    val_model_path = f'best_model_val_{run_suffix}.pth'
    model.load_state_dict(torch.load(val_model_path))
    val_test_loss, val_test_acc, val_confusion_matrices, val_avg_inference_time = test(model, test_loader, criterion, device)
    
    print('\nBest Validation Model Test Results:')
    print(f'Test Loss: {val_test_loss:.4f}')
    print(f'Test Accuracy:')
    print(f'Disease: {val_test_acc["disease"]:.2f}%')
    print(f'Average Inference Time: {val_avg_inference_time*1000:.2f} ms per batch')
    
    logging.info('\nBest Validation Model Test Results:')
    logging.info(f'Test Loss: {val_test_loss:.4f}')
    logging.info(f'Test Accuracy:')
    logging.info(f'Disease: {val_test_acc["disease"]:.2f}%')
    logging.info(f'Average Inference Time: {val_avg_inference_time*1000:.2f} ms per batch')
    
    # Test best test model
    print("\nTesting best test model...")
    test_model_path = f'best_model_test_{run_suffix}.pth'
    model.load_state_dict(torch.load(test_model_path))
    test_test_loss, test_test_acc, test_confusion_matrices, test_avg_inference_time = test(model, test_loader, criterion, device)
    
    print('\nBest Test Model Test Results:')
    print(f'Test Loss: {test_test_loss:.4f}')
    print(f'Test Accuracy:')
    print(f'Disease: {test_test_acc["disease"]:.2f}%')
    print(f'Average Inference Time: {test_avg_inference_time*1000:.2f} ms per batch')
    
    logging.info('\nBest Test Model Test Results:')
    logging.info(f'Test Loss: {test_test_loss:.4f}')
    logging.info(f'Test Accuracy:')
    logging.info(f'Disease: {test_test_acc["disease"]:.2f}%')
    logging.info(f'Average Inference Time: {test_avg_inference_time*1000:.2f} ms per batch')
    
    # Print confusion matrices for both models
    print('\nConfusion Matrices for Best Validation Model:')
    print('\nDISEASE Confusion Matrix:')
    print(val_confusion_matrices['disease'])
    logging.info('\nDISEASE Confusion Matrix (Best Validation Model):')
    logging.info(val_confusion_matrices['disease'])
    
    print('\nConfusion Matrices for Best Test Model:')
    print('\nDISEASE Confusion Matrix:')
    print(test_confusion_matrices['disease'])
    logging.info('\nDISEASE Confusion Matrix (Best Test Model):')
    logging.info(test_confusion_matrices['disease'])
    
    print(f"\nTraining and testing completed successfully! (Run ID: {run_suffix})")

if __name__ == '__main__':
    main() 