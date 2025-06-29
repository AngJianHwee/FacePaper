import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.get_face_dataloaders import get_face_dataloaders_subj_id
from models.ours.AttentionMobileNetShallow_s_three_task import AttentionMobileNetShallow_s_three_task
from models.ours.AttentionMobileNetShallow_s_three_task_IAT import AttentionMobileNetShallow_s_three_task_IAT
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
    return f"{timestamp}_{unique_id}_IAT_0.05_15ep"

def train_epoch(model, train_loader, optimizer, criterion, device, current_epoch, total_epochs):
    model.train()
    total_loss = 0
    total_loss_age = 0
    total_loss_gender = 0 
    total_loss_disease = 0
    total_loss_subject = 0 # Added for subject loss
    correct = {'age_5': 0, 'gender': 0, 'disease': 0, 'subject': 0} # Added for subject accuracy
    total = {'age_5': 0, 'gender': 0, 'disease': 0, 'subject': 0} # Added for subject total
    
    for images, labels, _ in tqdm(train_loader, desc='Training'):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        outputs = model(images) # outputs will be (out1, out2, out3, ID_pred)
        
        # Calculate loss for each task
        loss_age = criterion(outputs[0], labels['age_5'])
        loss_gender = criterion(outputs[1], labels['gender'])
        loss_disease = criterion(outputs[2], labels['disease'])
        loss_subject = criterion(outputs[3], labels['subject']) # Calculate subject loss
        
        # Total loss is sum of individual losses
        loss = loss_age + loss_gender + loss_disease + loss_subject # Include subject loss
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_age += loss_age.item()
        total_loss_gender += loss_gender.item()
        total_loss_disease += loss_disease.item()
        total_loss_subject += loss_subject.item() # Accumulate subject loss
        
        # Calculate accuracy for each task
        _, predicted_age = outputs[0].max(1)
        _, predicted_gender = outputs[1].max(1)
        _, predicted_disease = outputs[2].max(1)
        _, predicted_subject = outputs[3].max(1) # Predict subject
        
        correct['age_5'] += predicted_age.eq(labels['age_5']).sum().item()
        correct['gender'] += predicted_gender.eq(labels['gender']).sum().item()
        correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
        correct['subject'] += predicted_subject.eq(labels['subject']).sum().item() # Subject accuracy
        
        total['age_5'] += labels['age_5'].size(0)
        total['gender'] += labels['gender'].size(0)
        total['disease'] += labels['disease'].size(0)
        total['subject'] += labels['subject'].size(0) # Subject total
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    avg_loss_age = total_loss_age / len(train_loader)
    avg_loss_gender = total_loss_gender / len(train_loader)
    avg_loss_disease = total_loss_disease / len(train_loader)
    avg_loss_subject = total_loss_subject / len(train_loader) # Average subject loss
    
    # Calculate loss percentages
    loss_percentages = {
        'age_5': (avg_loss_age / avg_loss) * 100,
        'gender': (avg_loss_gender / avg_loss) * 100,
        'disease': (avg_loss_disease / avg_loss) * 100,
        'subject': (avg_loss_subject / avg_loss) * 100 # Subject loss percentage
    }
    
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease'],
        'subject': 100. * correct['subject'] / total['subject'] # Subject accuracy
    }
    
    # Print loss magnitudes and percentages
    logging.info(f"\n[{current_epoch}/{total_epochs}] Loss magnitudes:")
    logging.info(f"[{current_epoch}/{total_epochs}] Age loss: {avg_loss_age:.4f} ({loss_percentages['age_5']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Gender loss: {avg_loss_gender:.4f} ({loss_percentages['gender']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Disease loss: {avg_loss_disease:.4f} ({loss_percentages['disease']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Subject loss: {avg_loss_subject:.4f} ({loss_percentages['subject']:.1f}%)") # Subject loss logging
    
    return avg_loss, accuracy

# Removed validate function as get_face_dataloaders_subj_id does not return a validation set

def test(model, test_loader, criterion, device, current_epoch=None, total_epochs=None):
    model.eval()
    total_loss = 0
    total_loss_age = 0
    total_loss_gender = 0
    total_loss_disease = 0
    total_loss_subject = 0 # Added for subject loss
    correct = {'age_5': 0, 'gender': 0, 'disease': 0, 'subject': 0} # Added for subject accuracy
    total = {'age_5': 0, 'gender': 0, 'disease': 0, 'subject': 0} # Added for subject total
    
    # For detailed analysis
    all_predictions = {'age_5': [], 'gender': [], 'disease': [], 'subject': []} # Added for subject
    all_targets = {'age_5': [], 'gender': [], 'disease': [], 'subject': []} # Added for subject
    
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
            
            # Calculate loss for each task
            loss_age = criterion(outputs[0], labels['age_5'])
            loss_gender = criterion(outputs[1], labels['gender'])
            loss_disease = criterion(outputs[2], labels['disease'])
            loss_subject = criterion(outputs[3], labels['subject']) # Calculate subject loss
            
            # Total loss is sum of individual losses
            loss = loss_age + loss_gender + loss_disease + loss_subject # Include subject loss
            
            total_loss += loss.item()
            total_loss_age += loss_age.item()
            total_loss_gender += loss_gender.item()
            total_loss_disease += loss_disease.item()
            total_loss_subject += loss_subject.item() # Accumulate subject loss
            
            # Calculate accuracy for each task
            _, predicted_age = outputs[0].max(1)
            _, predicted_gender = outputs[1].max(1)
            _, predicted_disease = outputs[2].max(1)
            _, predicted_subject = outputs[3].max(1) # Predict subject
            
            # Store predictions and targets for detailed analysis
            all_predictions['age_5'].extend(predicted_age.cpu().numpy())
            all_predictions['gender'].extend(predicted_gender.cpu().numpy())
            all_predictions['disease'].extend(predicted_disease.cpu().numpy())
            all_predictions['subject'].extend(predicted_subject.cpu().numpy()) # Subject predictions
            
            all_targets['age_5'].extend(labels['age_5'].cpu().numpy())
            all_targets['gender'].extend(labels['gender'].cpu().numpy())
            all_targets['disease'].extend(labels['disease'].cpu().numpy())
            all_targets['subject'].extend(labels['subject'].cpu().numpy()) # Subject targets
            
            correct['age_5'] += predicted_age.eq(labels['age_5']).sum().item()
            correct['gender'] += predicted_gender.eq(labels['gender']).sum().item()
            correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
            correct['subject'] += predicted_subject.eq(labels['subject']).sum().item() # Subject accuracy
            
            total['age_5'] += labels['age_5'].size(0)
            total['gender'] += labels['gender'].size(0)
            total['disease'] += labels['disease'].size(0)
            total['subject'] += labels['subject'].size(0) # Subject total
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    avg_loss_age = total_loss_age / len(test_loader)
    avg_loss_gender = total_loss_gender / len(test_loader)
    avg_loss_disease = total_loss_disease / len(test_loader)
    avg_loss_subject = total_loss_subject / len(test_loader) # Average subject loss
    
    # Calculate average inference time
    avg_inference_time = total_inference_time / num_batches
    
    # Calculate loss percentages
    loss_percentages = {
        'age_5': (avg_loss_age / avg_loss) * 100,
        'gender': (avg_loss_gender / avg_loss) * 100,
        'disease': (avg_loss_disease / avg_loss) * 100,
        'subject': (avg_loss_subject / avg_loss) * 100 # Subject loss percentage
    }
    
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease'],
        'subject': 100. * correct['subject'] / total['subject'] # Subject accuracy
    }
    
    # Print loss magnitudes and percentages
    prefix = f"[{current_epoch}/{total_epochs}]" if current_epoch is not None else ""
    logging.info(f"\n{prefix} Test loss magnitudes:")
    logging.info(f"{prefix} Age loss: {avg_loss_age:.4f} ({loss_percentages['age_5']:.1f}%)")
    logging.info(f"{prefix} Gender loss: {avg_loss_gender:.4f} ({loss_percentages['gender']:.1f}%)")
    logging.info(f"{prefix} Disease loss: {avg_loss_disease:.4f} ({loss_percentages['disease']:.1f}%)")
    logging.info(f"{prefix} Subject loss: {avg_loss_subject:.4f} ({loss_percentages['subject']:.1f}%)") # Subject loss logging
    
    # Calculate confusion matrices
    confusion_matrices = {}
    for task in ['age_5', 'gender', 'disease', 'subject']: # Include subject
        confusion_matrices[task] = confusion_matrix(
            all_targets[task],
            all_predictions[task]
        )
    
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
    train_loader, test_loader, train_dataset, test_dataset, _ = get_face_dataloaders_subj_id( # Changed to get_face_dataloaders_subj_id
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
    n_classes_subject = len(train_dataset.label_mappings['subject']) # Get number of subject classes
    
    # Initialize base model
    base_model = AttentionMobileNetShallow_s_three_task(
        input_channels=3,
        n_classes_task1=n_classes_age,
        n_classes_task2=n_classes_gender,
        n_classes_task3=n_classes_disease,
        input_size=32,
        use_attention=True
    ).to(device)
    
    # Load pre-trained weights for the base model (dummy placeholder)
    pretrained_weights_path = "best_model_test_20250510_173949_3aaaee01.pth" # Placeholder
    try:
        base_model.load_state_dict(torch.load(pretrained_weights_path, map_location=device))
        print(f"Loaded pre-trained weights for base model from {pretrained_weights_path}")
        logging.info(f"Loaded pre-trained weights for base model from {pretrained_weights_path}")
    except FileNotFoundError:
        print(f"Pre-trained weights file not found at {pretrained_weights_path}. Starting training without pre-trained weights.")
        logging.warning(f"Pre-trained weights file not found at {pretrained_weights_path}. Starting training without pre-trained weights.")
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}. Starting training without pre-trained weights.")
        logging.error(f"Error loading pre-trained weights: {e}. Starting training without pre-trained weights.")

    # Initialize IAT model with the base model
    model = AttentionMobileNetShallow_s_three_task_IAT(
        existing_model=base_model,
        grad_reverse=0.05, # Example value for grad_reverse
        num_subjects=n_classes_subject
    ).to(device)
    print(f"Model: AttentionMobileNetShallow_s_three_task_IAT (Finetuning)")
    
    
    # Initialize optimizer and criterion
    optimizer = get_optimizer(model.parameters(), name='adam', lr=0.001, weight_decay=1e-4)
    criterion = CrossEntropyLoss()
    
    # Training loop
    num_epochs = 15
    # Removed best_val_loss as there is no validation set
    best_test_loss = float('inf')
    
    print(f"\nStarting training for {num_epochs} epochs...")
    
    for epoch, i in enumerate(range(1, num_epochs + 1), 1):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, i, num_epochs)
        
        # Removed Validate call
        
        # Test after each epoch
        test_loss, test_acc, test_confusion_matrices, test_avg_inference_time = test(model, test_loader, criterion, device, i, num_epochs)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'\n[{i}/{num_epochs}] Epoch completed in {epoch_time:.2f}s')
        print(f'[{i}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc - Age: {train_acc["age_5"]:.2f}%, Gender: {train_acc["gender"]:.2f}%, Disease: {train_acc["disease"]:.2f}%, Subject: {train_acc["subject"]:.2f}%') # Added subject
        # Removed Val Loss print
        print(f'[{i}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Acc - Age: {test_acc["age_5"]:.2f}%, Gender: {test_acc["gender"]:.2f}%, Disease: {test_acc["disease"]:.2f}%, Subject: {test_acc["subject"]:.2f}%') # Added subject
        
        logging.info(f'[{i}/{num_epochs}] Epoch completed in {epoch_time:.2f}s')
        logging.info(f'[{i}/{num_epochs}] Train Loss: {train_loss:.4f}, Train Acc - Age: {train_acc["age_5"]:.2f}%, Gender: {train_acc["gender"]:.2f}%, Disease: {train_acc["disease"]:.2f}%, Subject: {train_acc["subject"]:.2f}%') # Added subject
        # Removed Val Loss logging
        logging.info(f'[{i}/{num_epochs}] Test Loss: {test_loss:.4f}, Test Acc - Age: {test_acc["age_5"]:.2f}%, Gender: {test_acc["gender"]:.2f}%, Disease: {test_acc["disease"]:.2f}%, Subject: {test_acc["subject"]:.2f}%') # Added subject
        
        # Removed Save best model based on validation loss
        
        # Save best model based on test loss
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            test_model_path = f'best_model_test_{run_suffix}.pth'
            torch.save(model.state_dict(), test_model_path)
            print(f'[{i}/{num_epochs}] Saved best test model to {test_model_path}')
            logging.info(f'[{i}/{num_epochs}] Saved best test model to {test_model_path}')
    
    # print("\nTraining completed. Testing best model...") # Adjusted message
    
    # # Removed Test best validation model section
    
    # # Test best test model
    # print("\nTesting best test model...")
    # test_model_path = f'best_model_test_{run_suffix}.pth'
    # model.load_state_dict(torch.load(test_model_path))
    # test_test_loss, test_test_acc, test_confusion_matrices, test_avg_inference_time = test(model, test_loader, criterion, device)
    
    # print('\nBest Test Model Test Results:')
    # print(f'Test Loss: {test_test_loss:.4f}')
    # print(f'Test Accuracy:')
    # print(f'Age: {test_test_acc["age_5"]:.2f}%')
    # print(f'Gender: {test_test_acc["gender"]:.2f}%')
    # print(f'Disease: {test_test_acc["disease"]:.2f}%')
    # print(f'Subject: {test_test_acc["subject"]:.2f}%') # Added subject
    # print(f'Average Inference Time: {test_avg_inference_time*1000:.2f} ms per batch')
    
    # logging.info('\nBest Test Model Test Results:')
    # logging.info(f'Test Loss: {test_test_loss:.4f}')
    # logging.info(f'Test Accuracy:')
    # logging.info(f'Age: {test_test_acc["age_5"]:.2f}%')
    # logging.info(f'Gender: {test_test_acc["gender"]:.2f}%')
    # logging.info(f'Disease: {test_test_acc["disease"]:.2f}%')
    # logging.info(f'Subject: {test_test_acc["subject"]:.2f}%') # Added subject
    # logging.info(f'Average Inference Time: {test_avg_inference_time*1000:.2f} ms per batch')
    
    # # Print confusion matrices for the best test model
    # print('\nConfusion Matrices for Best Test Model:')
    # for task in ['age_5', 'gender', 'disease', 'subject']: # Include subject
    #     print(f'\n{task.upper()} Confusion Matrix:')
    #     print(test_confusion_matrices[task])
        
    #     logging.info(f'\n{task.upper()} Confusion Matrix (Best Test Model):')
    #     logging.info(test_confusion_matrices[task])
    
    print(f"\nTraining and testing completed successfully! (Run ID: {run_suffix})")

if __name__ == '__main__':
    main()
