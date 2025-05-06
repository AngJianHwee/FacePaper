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

def train_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
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
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    return avg_loss, accuracy

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
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
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    return avg_loss, accuracy

def test(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
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
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease']
    }
    
    # Calculate confusion matrices
    confusion_matrices = {}
    for task in ['age_5', 'gender', 'disease']:
        confusion_matrices[task] = confusion_matrix(
            all_targets[task],
            all_predictions[task]
        )
    
    return avg_loss, accuracy, confusion_matrices

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    
    # Get dataloaders
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset, _ = get_face_dataloaders(
        data_dir='./data/face',
        batch_size=32,
        num_workers=4,
        task='all',
        resize=224
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
        input_size=224,
        use_attention=True
    ).to(device)
    
    # Initialize optimizer and criterion
    optimizer = get_optimizer(model.parameters(), name='adam', lr=0.001, weight_decay=1e-4)
    criterion = CrossEntropyLoss()
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        logging.info(f'Epoch {epoch+1}/{num_epochs} - {epoch_time:.2f}s')
        logging.info(f'Train Loss: {train_loss:.4f}, Train Acc - Age: {train_acc["age_5"]:.2f}%, Gender: {train_acc["gender"]:.2f}%, Disease: {train_acc["disease"]:.2f}%')
        logging.info(f'Val Loss: {val_loss:.4f}, Val Acc - Age: {val_acc["age_5"]:.2f}%, Gender: {val_acc["gender"]:.2f}%, Disease: {val_acc["disease"]:.2f}%')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logging.info('Saved best model')
    
    # Load best model for testing
    model.load_state_dict(torch.load('best_model.pth'))
    logging.info('Loaded best model for testing')
    
    # Test the model
    test_loss, test_acc, confusion_matrices = test(model, test_loader, criterion, device)
    
    # Print test results
    logging.info('\nTest Results:')
    logging.info(f'Test Loss: {test_loss:.4f}')
    logging.info(f'Test Accuracy:')
    logging.info(f'Age: {test_acc["age_5"]:.2f}%')
    logging.info(f'Gender: {test_acc["gender"]:.2f}%')
    logging.info(f'Disease: {test_acc["disease"]:.2f}%')
    
    # Print confusion matrices
    logging.info('\nConfusion Matrices:')
    for task in ['age_5', 'gender', 'disease']:
        logging.info(f'\n{task.upper()} Confusion Matrix:')
        logging.info(confusion_matrices[task])

if __name__ == '__main__':
    main() 