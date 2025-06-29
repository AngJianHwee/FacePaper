import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.get_face_dataloaders import get_face_dataloaders, get_face_dataloaders_subj_id
from models.ours.AttentionMobileNetShallow_s_three_task_IAT import AttentionMobileNetShallow_s_three_task
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
        outputs = model(images) # Model returns (out_age, out_gender, out_disease) when grad_reverse=0
        
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

def train_epoch_finetune(model, train_loader, optimizer, criterion, subject_criterion, device, current_epoch, total_epochs, lambda_gr=1.0):
    print("Lambda_gr is set to:", lambda_gr)
    model.train()
    total_loss = 0
    total_loss_age = 0
    total_loss_gender = 0 
    total_loss_disease = 0
    total_loss_subject = 0
    correct = {'age_5': 0, 'gender': 0, 'disease': 0, 'subject': 0}
    total = {'age_5': 0, 'gender': 0, 'disease': 0, 'subject': 0}
    
    for images, labels, _ in tqdm(train_loader, desc='Fine-tuning Training'):
        images = images.to(device)
        labels = {k: v.to(device) for k, v in labels.items()}
        
        optimizer.zero_grad()
        outputs = model(images) # Model returns (out_age, out_gender, out_disease, out_subject) when grad_reverse!=0
        
        # Calculate loss for each task
        loss_age = criterion(outputs[0], labels['age_5'])
        loss_gender = criterion(outputs[1], labels['gender'])
        loss_disease = criterion(outputs[2], labels['disease'])
        loss_subject = subject_criterion(outputs[3], labels['subject'])
        
        # Total loss is sum of individual losses, with subject loss weighted by lambda_gr
        loss = loss_age + loss_gender + loss_disease + lambda_gr * loss_subject
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_loss_age += loss_age.item()
        total_loss_gender += loss_gender.item()
        total_loss_disease += loss_disease.item()
        total_loss_subject += loss_subject.item()
        
        # Calculate accuracy for each task
        _, predicted_age = outputs[0].max(1)
        _, predicted_gender = outputs[1].max(1)
        _, predicted_disease = outputs[2].max(1)
        _, predicted_subject = outputs[3].max(1)
        
        correct['age_5'] += predicted_age.eq(labels['age_5']).sum().item()
        correct['gender'] += predicted_gender.eq(labels['gender']).sum().item()
        correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
        correct['subject'] += predicted_subject.eq(labels['subject']).sum().item()
        
        total['age_5'] += labels['age_5'].size(0)
        total['gender'] += labels['gender'].size(0)
        total['disease'] += labels['disease'].size(0)
        total['subject'] += labels['subject'].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(train_loader)
    avg_loss_age = total_loss_age / len(train_loader)
    avg_loss_gender = total_loss_gender / len(train_loader)
    avg_loss_disease = total_loss_disease / len(train_loader)
    avg_loss_subject = total_loss_subject / len(train_loader)
    
    # Calculate loss percentages
    loss_percentages = {
        'age_5': (avg_loss_age / avg_loss) * 100,
        'gender': (avg_loss_gender / avg_loss) * 100,
        'disease': (avg_loss_disease / avg_loss) * 100,
        'subject': (avg_loss_subject / avg_loss) * 100
    }
    
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease'],
        'subject': 100. * correct['subject'] / total['subject']
    }
    
    # Print loss magnitudes and percentages
    logging.info(f"\n[{current_epoch}/{total_epochs}] Fine-tuning Loss magnitudes:")
    logging.info(f"[{current_epoch}/{total_epochs}] Age loss: {avg_loss_age:.4f} ({loss_percentages['age_5']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Gender loss: {avg_loss_gender:.4f} ({loss_percentages['gender']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Disease loss: {avg_loss_disease:.4f} ({loss_percentages['disease']:.1f}%)")
    logging.info(f"[{current_epoch}/{total_epochs}] Subject loss: {avg_loss_subject:.4f} ({loss_percentages['subject']:.1f}%)")
    
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
    
    # For timing measurements
    total_inference_time = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='Testing'):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images) # Model returns (out_age, out_gender, out_disease) when grad_reverse=0
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            num_batches += 1
            
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
    
    # Calculate average inference time
    avg_inference_time = total_inference_time / num_batches
    
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
    
    return avg_loss, accuracy, confusion_matrices, avg_inference_time

def test_finetune(model, test_loader, criterion, subject_criterion, device, current_epoch=None, total_epochs=None):
    model.eval()
    total_loss = 0
    total_loss_age = 0
    total_loss_gender = 0
    total_loss_disease = 0
    total_loss_subject = 0
    correct = {'age_5': 0, 'gender': 0, 'disease': 0, 'subject': 0}
    total = {'age_5': 0, 'gender': 0, 'disease': 0, 'subject': 0}
    
    # For detailed analysis
    all_predictions = {'age_5': [], 'gender': [], 'disease': [], 'subject': []}
    all_targets = {'age_5': [], 'gender': [], 'disease': [], 'subject': []}
    
    # For timing measurements
    total_inference_time = 0
    num_batches = 0
    
    with torch.no_grad():
        for images, labels, _ in tqdm(test_loader, desc='Fine-tuning Testing'):
            images = images.to(device)
            labels = {k: v.to(device) for k, v in labels.items()}
            
            # Measure inference time
            start_time = time.time()
            outputs = model(images) # Model returns (out_age, out_gender, out_disease, out_subject) when grad_reverse!=0
            end_time = time.time()
            total_inference_time += (end_time - start_time)
            num_batches += 1
            
            # Calculate loss for each task
            loss_age = criterion(outputs[0], labels['age_5'])
            loss_gender = criterion(outputs[1], labels['gender'])
            loss_disease = criterion(outputs[2], labels['disease'])
            loss_subject = subject_criterion(outputs[3], labels['subject'])
            
            # Total loss is sum of individual losses
            loss = loss_age + loss_gender + loss_disease + loss_subject
            
            total_loss += loss.item()
            total_loss_age += loss_age.item()
            total_loss_gender += loss_gender.item()
            total_loss_disease += loss_disease.item()
            total_loss_subject += loss_subject.item()
            
            # Calculate accuracy for each task
            _, predicted_age = outputs[0].max(1)
            _, predicted_gender = outputs[1].max(1)
            _, predicted_disease = outputs[2].max(1)
            _, predicted_subject = outputs[3].max(1)
            
            # Store predictions and targets for detailed analysis
            all_predictions['age_5'].extend(predicted_age.cpu().numpy())
            all_predictions['gender'].extend(predicted_gender.cpu().numpy())
            all_predictions['disease'].extend(predicted_disease.cpu().numpy())
            all_predictions['subject'].extend(predicted_subject.cpu().numpy())
            
            all_targets['age_5'].extend(labels['age_5'].cpu().numpy())
            all_targets['gender'].extend(labels['gender'].cpu().numpy())
            all_targets['disease'].extend(labels['disease'].cpu().numpy())
            all_targets['subject'].extend(labels['subject'].cpu().numpy())
            
            correct['age_5'] += predicted_age.eq(labels['age_5']).sum().item()
            correct['gender'] += predicted_gender.eq(labels['gender']).sum().item()
            correct['disease'] += predicted_disease.eq(labels['disease']).sum().item()
            correct['subject'] += predicted_subject.eq(labels['subject']).sum().item()
            
            total['age_5'] += labels['age_5'].size(0)
            total['gender'] += labels['gender'].size(0)
            total['disease'] += labels['disease'].size(0)
            total['subject'] += labels['subject'].size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader)
    avg_loss_age = total_loss_age / len(test_loader)
    avg_loss_gender = total_loss_gender / len(test_loader)
    avg_loss_disease = total_loss_disease / len(test_loader)
    avg_loss_subject = total_loss_subject / len(test_loader)
    
    # Calculate average inference time
    avg_inference_time = total_inference_time / num_batches
    
    # Calculate loss percentages
    loss_percentages = {
        'age_5': (avg_loss_age / avg_loss) * 100,
        'gender': (avg_loss_gender / avg_loss) * 100,
        'disease': (avg_loss_disease / avg_loss) * 100,
        'subject': (avg_loss_subject / avg_loss) * 100
    }
    
    accuracy = {
        'age_5': 100. * correct['age_5'] / total['age_5'],
        'gender': 100. * correct['gender'] / total['gender'],
        'disease': 100. * correct['disease'] / total['disease'],
        'subject': 100. * correct['subject'] / total['subject']
    }
    
    # Print loss magnitudes and percentages
    prefix = f"[{current_epoch}/{total_epochs}]" if current_epoch is not None else ""
    logging.info(f"\n{prefix} Fine-tuning Test loss magnitudes:")
    logging.info(f"{prefix} Age loss: {avg_loss_age:.4f} ({loss_percentages['age_5']:.1f}%)")
    logging.info(f"{prefix} Gender loss: {avg_loss_gender:.4f} ({loss_percentages['gender']:.1f}%)")
    logging.info(f"{prefix} Disease loss: {avg_loss_disease:.4f} ({loss_percentages['disease']:.1f}%)")
    logging.info(f"{prefix} Subject loss: {avg_loss_subject:.4f} ({loss_percentages['subject']:.1f}%)")
    
    # Calculate confusion matrices
    confusion_matrices = {}
    for task in ['age_5', 'gender', 'disease', 'subject']:
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
    
    # --- Phase 1: Initial Training (without subject adversarial training) ---
    print("\n--- Phase 1: Initial Training (without subject adversarial training) ---")
    logging.info("\n--- Phase 1: Initial Training (without subject adversarial training) ---")

    # Get dataloaders for Phase 1 (no validation set from get_face_dataloaders_subj_id)
    train_loader_stage1, test_loader_stage1, train_dataset_stage1, test_dataset_stage1, _ = get_face_dataloaders_subj_id(
        data_dir='./data/face',
        batch_size=32,
        num_workers=4,
        task='all',
        resize=32
    )
    
    # Get number of classes for each task
    n_classes_age_stage1 = len(train_dataset_stage1.label_mappings['age_5'])
    n_classes_gender_stage1 = len(train_dataset_stage1.label_mappings['gender'])
    n_classes_disease_stage1 = len(train_dataset_stage1.label_mappings['disease'])
    n_classes_subject_stage1 = len(train_dataset_stage1.label_mappings['subject'])
    
    # Initialize model for Phase 1 (grad_reverse=0)
    model_stage1 = AttentionMobileNetShallow_s_three_task(
        input_channels=3,
        n_classes_task1=n_classes_age_stage1,
        n_classes_task2=n_classes_gender_stage1,
        n_classes_task3=n_classes_disease_stage1,
        input_size=32,
        use_attention=True,
        grad_reverse=0, # No adversarial training in Phase 1
        num_subjects=n_classes_subject_stage1
    ).to(device)
    print(f"Model for Phase 1: AttentionMobileNetShallow_s_three_task (grad_reverse=0)")
    
    # Initialize optimizer and criterion for Phase 1
    optimizer_stage1 = get_optimizer(model_stage1.parameters(), name='adam', lr=0.001, weight_decay=1e-4)
    criterion_stage1 = CrossEntropyLoss()
    
    # Training loop for Phase 1
    num_epochs_stage1 = 80
    best_test_loss_stage1 = float('inf')
    best_model_path_stage1 = f'best_model_stage1_{run_suffix}.pth'
    
    print(f"\nStarting Phase 1 training for {num_epochs_stage1} epochs...")
    
    for epoch, i in enumerate(range(1, num_epochs_stage1 + 1), 1):
        start_time = time.time()
        
        # Train
        train_loss_stage1, train_acc_stage1 = train_epoch(model_stage1, train_loader_stage1, optimizer_stage1, criterion_stage1, device, i, num_epochs_stage1)
        
        # Test after each epoch
        test_loss_stage1, test_acc_stage1, test_confusion_matrices_stage1, test_avg_inference_time_stage1 = test(model_stage1, test_loader_stage1, criterion_stage1, device, i, num_epochs_stage1)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'\n[{i}/{num_epochs_stage1}] Phase 1 Epoch completed in {epoch_time:.2f}s')
        print(f'[{i}/{num_epochs_stage1}] Phase 1 Train Loss: {train_loss_stage1:.4f}, Train Acc - Age: {train_acc_stage1["age_5"]:.2f}%, Gender: {train_acc_stage1["gender"]:.2f}%, Disease: {train_acc_stage1["disease"]:.2f}%')
        print(f'[{i}/{num_epochs_stage1}] Phase 1 Test Loss: {test_loss_stage1:.4f}, Test Acc - Age: {test_acc_stage1["age_5"]:.2f}%, Gender: {test_acc_stage1["gender"]:.2f}%, Disease: {test_acc_stage1["disease"]:.2f}%')
        
        logging.info(f'[{i}/{num_epochs_stage1}] Phase 1 Epoch completed in {epoch_time:.2f}s')
        logging.info(f'[{i}/{num_epochs_stage1}] Phase 1 Train Loss: {train_loss_stage1:.4f}, Train Acc - Age: {train_acc_stage1["age_5"]:.2f}%, Gender: {train_acc_stage1["gender"]:.2f}%, Disease: {train_acc_stage1["disease"]:.2f}%')
        logging.info(f'[{i}/{num_epochs_stage1}] Phase 1 Test Loss: {test_loss_stage1:.4f}, Test Acc - Age: {test_acc_stage1["age_5"]:.2f}%, Gender: {test_acc_stage1["gender"]:.2f}%, Disease: {test_acc_stage1["disease"]:.2f}%')
        
        # Save best model based on test loss for Phase 1
        if test_loss_stage1 < best_test_loss_stage1:
            best_test_loss_stage1 = test_loss_stage1
            torch.save(model_stage1.state_dict(), best_model_path_stage1)
            print(f'[{i}/{num_epochs_stage1}] Saved best Phase 1 model to {best_model_path_stage1}')
            logging.info(f'[{i}/{num_epochs_stage1}] Saved best Phase 1 model to {best_model_path_stage1}')
    
    print("\nPhase 1 training completed. Testing best Phase 1 model...")
    model_stage1.load_state_dict(torch.load(best_model_path_stage1))
    final_test_loss_stage1, final_test_acc_stage1, final_confusion_matrices_stage1, final_avg_inference_time_stage1 = test(model_stage1, test_loader_stage1, criterion_stage1, device)
    
    print('\nBest Phase 1 Model Test Results:')
    print(f'Test Loss: {final_test_loss_stage1:.4f}')
    print(f'Test Accuracy:')
    print(f'Age: {final_test_acc_stage1["age_5"]:.2f}%')
    print(f'Gender: {final_test_acc_stage1["gender"]:.2f}%')
    print(f'Disease: {final_test_acc_stage1["disease"]:.2f}%')
    print(f'Average Inference Time: {final_avg_inference_time_stage1*1000:.2f} ms per batch')
    
    logging.info('\nBest Phase 1 Model Test Results:')
    logging.info(f'Test Loss: {final_test_loss_stage1:.4f}')
    logging.info(f'Test Accuracy:')
    logging.info(f'Age: {final_test_acc_stage1["age_5"]:.2f}%')
    logging.info(f'Gender: {final_test_acc_stage1["gender"]:.2f}%')
    logging.info(f'Disease: {final_test_acc_stage1["disease"]:.2f}%')
    logging.info(f'Average Inference Time: {final_avg_inference_time_stage1*1000:.2f} ms per batch')

    # --- Phase 2: Fine-tuning (with subject adversarial training) ---
    print("\n--- Phase 2: Fine-tuning (with subject adversarial training) ---")
    logging.info("\n--- Phase 2: Fine-tuning (with subject adversarial training) ---")

    # Load the best model from Phase 1
    model_finetune = AttentionMobileNetShallow_s_three_task(
        input_channels=3,
        n_classes_task1=n_classes_age_stage1,
        n_classes_task2=n_classes_gender_stage1,
        n_classes_task3=n_classes_disease_stage1,
        input_size=32,
        use_attention=True,
        grad_reverse=1.0, # Enable adversarial training in Phase 2
        num_subjects=n_classes_subject_stage1
    ).to(device)
    model_finetune.load_state_dict(torch.load(best_model_path_stage1))
    print(f"Model for Phase 2: AttentionMobileNetShallow_s_three_task (grad_reverse=1.0), loaded from {best_model_path_stage1}")

    # Initialize optimizer and criteria for Phase 2
    optimizer_finetune = get_optimizer(model_finetune.parameters(), name='adam', lr=0.0001, weight_decay=1e-4) # Smaller LR for fine-tuning
    criterion_finetune = CrossEntropyLoss()
    subject_criterion_finetune = CrossEntropyLoss()
    lambda_gr = 1.0 # Gradient reversal strength

    # Training loop for Phase 2
    num_epochs_finetune = 20 # Fewer epochs for fine-tuning
    best_test_loss_finetune = float('inf')
    best_model_path_finetune = f'best_model_finetuned_{run_suffix}.pth'

    print(f"\nStarting Phase 2 fine-tuning for {num_epochs_finetune} epochs...")

    for epoch, i in enumerate(range(1, num_epochs_finetune + 1), 1):
        start_time = time.time()
        
        # Train
        train_loss_finetune, train_acc_finetune = train_epoch_finetune(model_finetune, train_loader_stage1, optimizer_finetune, criterion_finetune, subject_criterion_finetune, device, i, num_epochs_finetune, lambda_gr)
        
        # Test after each epoch
        test_loss_finetune, test_acc_finetune, test_confusion_matrices_finetune, test_avg_inference_time_finetune = test_finetune(model_finetune, test_loader_stage1, criterion_finetune, subject_criterion_finetune, device, i, num_epochs_finetune)
        
        # Print epoch results
        epoch_time = time.time() - start_time
        print(f'\n[{i}/{num_epochs_finetune}] Phase 2 Epoch completed in {epoch_time:.2f}s')
        print(f'[{i}/{num_epochs_finetune}] Phase 2 Train Loss: {train_loss_finetune:.4f}, Train Acc - Age: {train_acc_finetune["age_5"]:.2f}%, Gender: {train_acc_finetune["gender"]:.2f}%, Disease: {train_acc_finetune["disease"]:.2f}%, Subject: {train_acc_finetune["subject"]:.2f}%')
        print(f'[{i}/{num_epochs_finetune}] Phase 2 Test Loss: {test_loss_finetune:.4f}, Test Acc - Age: {test_acc_finetune["age_5"]:.2f}%, Gender: {test_acc_finetune["gender"]:.2f}%, Disease: {test_acc_finetune["disease"]:.2f}%, Subject: {test_acc_finetune["subject"]:.2f}%')
        
        logging.info(f'[{i}/{num_epochs_finetune}] Phase 2 Epoch completed in {epoch_time:.2f}s')
        logging.info(f'[{i}/{num_epochs_finetune}] Phase 2 Train Loss: {train_loss_finetune:.4f}, Train Acc - Age: {train_acc_finetune["age_5"]:.2f}%, Gender: {train_acc_finetune["gender"]:.2f}%, Disease: {train_acc_finetune["disease"]:.2f}%, Subject: {train_acc_finetune["subject"]:.2f}%')
        logging.info(f'[{i}/{num_epochs_finetune}] Phase 2 Test Loss: {test_loss_finetune:.4f}, Test Acc - Age: {test_acc_finetune["age_5"]:.2f}%, Gender: {test_acc_finetune["gender"]:.2f}%, Disease: {test_acc_finetune["disease"]:.2f}%, Subject: {test_acc_finetune["subject"]:.2f}%')
        
        # Save best model based on test loss for Phase 2
        if test_loss_finetune < best_test_loss_finetune:
            best_test_loss_finetune = test_loss_finetune
            torch.save(model_finetune.state_dict(), best_model_path_finetune)
            print(f'[{i}/{num_epochs_finetune}] Saved best Phase 2 model to {best_model_path_finetune}')
            logging.info(f'[{i}/{num_epochs_finetune}] Saved best Phase 2 model to {best_model_path_finetune}')

    print("\nPhase 2 fine-tuning completed. Testing best fine-tuned model...")
    model_finetune.load_state_dict(torch.load(best_model_path_finetune))
    final_test_loss_finetune, final_test_acc_finetune, final_confusion_matrices_finetune, final_avg_inference_time_finetune = test_finetune(model_finetune, test_loader_stage1, criterion_finetune, subject_criterion_finetune, device)
    
    print('\nBest Fine-tuned Model Test Results:')
    print(f'Test Loss: {final_test_loss_finetune:.4f}')
    print(f'Test Accuracy:')
    print(f'Age: {final_test_acc_finetune["age_5"]:.2f}%')
    print(f'Gender: {final_test_acc_finetune["gender"]:.2f}%')
    print(f'Disease: {final_test_acc_finetune["disease"]:.2f}%')
    print(f'Subject: {final_test_acc_finetune["subject"]:.2f}%')
    print(f'Average Inference Time: {final_avg_inference_time_finetune*1000:.2f} ms per batch')
    
    logging.info('\nBest Fine-tuned Model Test Results:')
    logging.info(f'Test Loss: {final_test_loss_finetune:.4f}')
    logging.info(f'Test Accuracy:')
    logging.info(f'Age: {final_test_acc_finetune["age_5"]:.2f}%')
    logging.info(f'Gender: {final_test_acc_finetune["gender"]:.2f}%')
    logging.info(f'Disease: {final_test_acc_finetune["disease"]:.2f}%')
    logging.info(f'Subject: {final_test_acc_finetune["subject"]:.2f}%')
    logging.info(f'Average Inference Time: {final_avg_inference_time_finetune*1000:.2f} ms per batch')

    # Print confusion matrices for both models
    print('\nConfusion Matrices for Best Phase 1 Model:')
    for task in ['age_5', 'gender', 'disease']:
        print(f'\n{task.upper()} Confusion Matrix:')
        print(final_confusion_matrices_stage1[task])
        
        logging.info(f'\n{task.upper()} Confusion Matrix (Best Phase 1 Model):')
        logging.info(final_confusion_matrices_stage1[task])

    print('\nConfusion Matrices for Best Fine-tuned Model:')
    for task in ['age_5', 'gender', 'disease', 'subject']:
        print(f'\n{task.upper()} Confusion Matrix:')
        print(final_confusion_matrices_finetune[task])
        
        logging.info(f'\n{task.upper()} Confusion Matrix (Best Fine-tuned Model):')
        logging.info(final_confusion_matrices_finetune[task])
    
    print(f"\nTraining and testing completed successfully! (Run ID: {run_suffix})")

if __name__ == '__main__':
    main()
