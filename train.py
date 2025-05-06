import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.get_face_dataloaders import get_face_dataloaders
from models.ours.AttentionMobileNetShallow_s_three_task import AttentionMobileNetShallow_s_three_task
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
    total_loss_age = 0
    total_loss_gender = 0 
    total_loss_disease = 0
    correct = {'age_5': 0, 'gender': 0, 'disease': 0}
    total = {'age_5': 0, 'gender': 0, 'disease': 0}
    
    for images, labels, _ in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        outputs = model(images)
        
        # Calculate loss for each task
        loss_age = criterion(outputs[0], labels['age_5'])
        loss_gender = criterion(outputs[1], labels['gender'])
        loss_disease = criterion(outputs[2], labels['disease'])
        
        # Total loss is sum of individual losses
        loss = loss_age + loss_gender + loss_disease
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_age += loss_age.item()
        total_loss_gender += loss_gender.item()
        total_loss_disease += loss_disease.item()
        
        # Calculate accuracy for each task
        _, predicted_age = outputs[0].max(1)
        _, predicted_gender = outputs[1].max(1)
        _, predicted_disease = outputs[2].max(1)
        
        correct['age_5'] += predicted_age.eq(labels['age_5']).sum().item()
        correct['gender'] += predicted_gender.eq(labels['gender']).sum().item()
        correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
        
        total['age_5'] += labels['age_5'].size(0)
        total['gender'] += labels['gender'].size(0)
        total['disease'] += labels['disease'].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    avg_loss_age = total_loss_age / len(train_loader)
    avg_loss_gender = total_loss_gender / len(train_loader)
    avg_loss_disease = total_loss_disease / len(train_loader)
    
    # Calculate loss percentages
    loss_percentages = {
        'age_5': (avg_loss_age / avg_loss) * 100,
        'gender': (avg_loss_gender / avg_loss) * 100,
        'disease': (avg_loss_disease / avg_loss) * 100
    }
    
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    # Print loss magnitudes and percentages
    logging.info(f"\n[{current_epoch}/{total_epochs}] Loss magnitudes:")
    logging.info(f"[{current_epoch}/{total_epochs}] Age loss: {avg_loss_age:.4f} ({loss_percentages['age_5']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Gender loss: {avg_loss_gender:.4f} ({loss_percentages['gender']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Disease loss: {avg_loss_disease:.4f} ({loss_percentages['disease']:.1f}%)")
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device, current_epoch, total_epochs):
    model.eval()
    total_loss = 0
    total_loss_age = 0
    total_loss_gender = 0
    total_loss_disease = 0
    correct = {'age_5': 0, 'gender': 0, 'disease': 0}
    total = {'age_5': 0, 'gender': 0, 'disease': 0}
    
    with torch.no_grad():
        for images, labels, _ in tqdm(val_loader, desc='Validation'):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            outputs = model(images)
            
            # Calculate loss for each task
            loss_age = criterion(outputs[0], labels['age_5'])
            loss_gender = criterion(outputs[1], labels['gender'])
            loss_disease = criterion(outputs[2], labels['disease'])
            
            # Total loss is sum of individual losses
            loss = loss_age + loss_gender + loss_disease
            
            total_loss += loss.item()
            total_loss_age += loss_age.item()
            total_loss_gender += loss_gender.item()
            total_loss_disease += loss_disease.item()
            
            # Calculate accuracy for each task
            _, predicted_age = outputs[0].max(1)
            _, predicted_gender = outputs[1].max(1)
            _, predicted_disease = outputs[2].max(1)
            
            correct['age_5'] += predicted_age.eq(labels['age_5']).sum().item()
            correct['gender'] += predicted_gender.eq(labels['gender']).sum().item()
            correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
            
            total['age_5'] += labels['age_5'].size(0)
            total['gender'] += labels['gender'].size(0)
            total['disease'] += labels['disease'].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(val_loader)
    avg_loss_age = total_loss_age / len(val_loader)
    avg_loss_gender = total_loss_gender / len(val_loader)
    avg_loss_disease = total_loss_disease / len(val_loader)
    
    # Calculate loss percentages
    loss_percentages = {
        'age_5': (avg_loss_age / avg_loss) * 100,
        'gender': (avg_loss_gender / avg_loss) * 100,
        'disease': (avg_loss_disease / avg_loss) * 100
    }
    
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    # Print loss magnitudes and percentages
    logging.info(f"\n[{current_epoch}/{total_epochs}] Validation loss magnitudes:")
    logging.info(f"[{current_epoch}/{total_epochs}] Age loss: {avg_loss_age:.4f} ({loss_percentages['age_5']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Gender loss: {avg_loss_gender:.4f} ({loss_percentages['gender']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Disease loss: {avg_loss_disease:.4f} ({loss_percentages['disease']:.1f}%)")
    
    return avg_loss, accuracy

def test(model, test_loader, criterion, device, current_epoch=None, total_epochs=None):
    model.eval()
    total_loss = 0
    total_loss_age = 0
    total_loss_gender = 0
    total_loss_disease = 0
    correct = {'age_5': 0, 'gender': 0, 'disease': 0}
    total = {'age_5': 0, 'gender': 0, 'disease': 0}
    
    # For detailed analysis
    all_predictions = {'age_5': [], 'gender': [], 'disease': []}
    all_targets = {'age_5': [], 'gender': [], 'disease': []}
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            outputs = model(images)
            
            # Calculate loss for each task
            loss_age = criterion(outputs[0], labels['age_5'])
            loss_gender = criterion(outputs[1], labels['gender'])
            loss_disease = criterion(outputs[2], labels['disease'])
            
            # Total loss is sum of individual losses
            loss = loss_age + loss_gender + loss_disease
            
            total_loss += loss.item()
            total_loss_age += loss_age.item()
            total_loss_gender += loss_gender.item()
            total_loss_disease += loss_disease.item()
            
            # Calculate accuracy for each task
            _, predicted_age = outputs[0].max(1)
            _, predicted_gender = outputs[1].max(1)
            _, predicted_disease = outputs[2].max(1)
            
            # Store predictions and targets for detailed analysis
            all_predictions['age_5'].extend(predicted_age.cpu().numpy())
            all_predictions['gender'].extend(predicted_gender.cpu().numpy())
            all_predictions['disease'].extend(predicted_disease.cpu().numpy())
            
            all_targets['age_5'].extend(labels['age_5'].cpu().numpy())
            all_targets['gender'].extend(labels['gender'].cpu().numpy())
            all_targets['disease'].extend(labels['disease'].cpu().numpy())
            
            correct['age_5'] += predicted_age.eq(labels['age_5']).sum().item()
            correct['gender'] += predicted_gender.eq(labels['gender']).sum().item()
            correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
            
            total['age_5'] += labels['age_5'].size(0)
            total['gender'] += labels['gender'].size(0)
            total['disease'] += labels['disease'].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    avg_loss_age = total_loss_age / len(test_loader)
    avg_loss_gender = total_loss_gender / len(test_loader)
    avg_loss_disease = total_loss_disease / len(test_loader)
    
    # Calculate loss percentages
    loss_percentages = {
        'age_5': (avg_loss_age / avg_loss) * 100,
        'gender': (avg_loss_gender / avg_loss) * 100,
        'disease': (avg_loss_disease / avg_loss) * 100
    }
    
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    # Print loss magnitudes and percentages
    prefix = f"[{current_epoch}/{total_epochs}]" if current_epoch is not None else ""
    logging.info(f"\n{prefix} Test loss magnitudes:")
    logging.info(f"{prefix} Age loss: {avg_loss_age:.4f} ({loss_percentages['age_5']:.1f}%)")
    logging.info(f"{prefix} Gender loss: {avg_loss_gender:.4f} ({loss_percentages['gender']:.1f}%)")
    logging.info(f"{prefix} Disease loss: {avg_loss_disease:.4f} ({loss_percentages['disease']:.1f}%)")
    
    # Calculate confusion matrices
    confusion_matrices = {}
    for task in ['age_5', 'gender', 'disease']:
        confusion_matrices[task] = confusion_matrix(
            all_targets[task],
            all_predictions[task]
        )
    
    return avg_loss, accuracy, confusion_matrices

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
        task='all',
        resize=32
    )
    
    # Get number of classes for each task
    n_classes_age = len(train_dataset.label_mappings['age_5'])
    n_classes_gender = len(train_dataset.label_mappings['gender'])
    n_classes_disease = len(train_dataset.label_mappings['disease'])
    
    # Initialize model
    model = AttentionMobileNetShallow_s_three_task(
        input_channels=3,
        n_classes_task1=n_classes_age,
        n_classes_task2=n_classes_gender,
        n_classes_task3=n_classes_disease,
        input_size=32,
        use_attention=True
    ).to(device)
    
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
        test_loss, test_acc, _ = test(model, test_loader, criterion, device, i, num_epochs)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'\n[{i}/{num_epochs}] Epoch completed in {epoch_time:.2f}s')
        print(f'[{i}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc - Age: {train_acc["age_5"]:.2f}%, Gender: {train_acc["gender"]:.2f}%, Disease: {train_acc["disease"]:.2f}%')
        print(f'[{i}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Acc - Age: {val_acc["age_5"]:.2f}%, Gender: {val_acc["gender"]:.2f}%, Disease: {val_acc["disease"]:.2f}%')
        print(f'[{i}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Acc - Age: {test_acc["age_5"]:.2f}%, Gender: {test_acc["gender"]:.2f}%, Disease: {test_acc["disease"]:.2f}%')
        
        logging.info(f'[{i}/{num_epochs}] Epoch completed in {epoch_time:.2f}s')
        logging.info(f'[{i}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc - Age: {train_acc["age_5"]:.2f}%, Gender: {train_acc["gender"]:.2f}%, Disease: {train_acc["disease"]:.2f}%')
        logging.info(f'[{i}/{num_epochs}] Val Loss: {val_loss:.4f}, Val Acc - Age: {val_acc["age_5"]:.2f}%, Gender: {val_acc["gender"]:.2f}%, Disease: {val_acc["disease"]:.2f}%')
        logging.info(f'[{i}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Acc - Age: {test_acc["age_5"]:.2f}%, Gender: {test_acc["gender"]:.2f}%, Disease: {test_acc["disease"]:.2f}%')
        
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
    val_test_loss, val_test_acc, val_confusion_matrices = test(model, test_loader, criterion, device)
    
    print('\nBest Validation Model Test Results:')
    print(f'Test Loss: {val_test_loss:.4f}')
    print(f'Test Accuracy:')
    print(f'Age: {val_test_acc["age_5"]:.2f}%')
    print(f'Gender: {val_test_acc["gender"]:.2f}%')
    print(f'Disease: {val_test_acc["disease"]:.2f}%')
    
    logging.info('\nBest Validation Model Test Results:')
    logging.info(f'Test Loss: {val_test_loss:.4f}')
    logging.info(f'Test Accuracy:')
    logging.info(f'Age: {val_test_acc["age_5"]:.2f}%')
    logging.info(f'Gender: {val_test_acc["gender"]:.2f}%')
    logging.info(f'Disease: {val_test_acc["disease"]:.2f}%')
    
    # Test best test model
    print("\nTesting best test model...")
    test_model_path = f'best_model_test_{run_suffix}.pth'
    model.load_state_dict(torch.load(test_model_path))
    test_test_loss, test_test_acc, test_confusion_matrices = test(model, test_loader, criterion, device)
    
    print('\nBest Test Model Test Results:')
    print(f'Test Loss: {test_test_loss:.4f}')
    print(f'Test Accuracy:')
    print(f'Age: {test_test_acc["age_5"]:.2f}%')
    print(f'Gender: {test_test_acc["gender"]:.2f}%')
    print(f'Disease: {test_test_acc["disease"]:.2f}%')
    
    logging.info('\nBest Test Model Test Results:')
    logging.info(f'Test Loss: {test_test_loss:.4f}')
    logging.info(f'Test Accuracy:')
    logging.info(f'Age: {test_test_acc["age_5"]:.2f}%')
    logging.info(f'Gender: {test_test_acc["gender"]:.2f}%')
    logging.info(f'Disease: {test_test_acc["disease"]:.2f}%')
    
    # Print confusion matrices for both models
    print('\nConfusion Matrices for Best Validation Model:')
    for task in ['age_5', 'gender', 'disease']:
        print(f'\n{task.upper()} Confusion Matrix:')
        print(val_confusion_matrices[task])
        
        logging.info(f'\n{task.upper()} Confusion Matrix (Best Validation Model):')
        logging.info(val_confusion_matrices[task])
    
    print('\nConfusion Matrices for Best Test Model:')
    for task in ['age_5', 'gender', 'disease']:
        print(f'\n{task.upper()} Confusion Matrix:')
        print(test_confusion_matrices[task])
        
        logging.info(f'\n{task.upper()} Confusion Matrix (Best Test Model):')
        logging.info(test_confusion_matrices[task])
    
    print(f"\nTraining and testing completed successfully! (Run ID: {run_suffix})")

if __name__ == '__main__':
    main() 