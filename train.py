import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import time
from tqdm import tqdm
import gc
from dataset import get_dataloader  # Assuming dataset.py is in the same directory
# Set random seed for reproducibility
torch.manual_seed(42)
from model import CorrelationPredictor  

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=10, device='cuda:0'):
    print("Start Training...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': []
    }
    
    # Force check if CUDA is available and print more info about the GPU
    if torch.cuda.is_available():
        device = 'cuda:0'  # Explicitly use GPU 0
        torch.cuda.set_device(0)  # Set default CUDA device
        print(f"CUDA is available. Using GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Enable cuDNN benchmark for optimal performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        device = 'cpu'
        print("CUDA is not available. Using CPU.")
    
    print(f"Using device: {device}")
    model = model.to(device)  # move model to device
    
    # Initialize mixed precision training if available (for Ampere GPUs and newer)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    best_val_loss = float('inf')
    best_model_weights = None
    
    # Early stopping parameters
    counter = 0
    early_stop = False
    
    # epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0) 
    
    for epoch in epoch_pbar:
        start_time = time.time()

        # Clear GPU cache before each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc="Training", leave=False, position=1)
        
        for images, targets in train_pbar:
            # Explicitly move tensors to device and ensure they are float tensors
            images = images.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Use mixed precision training if available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                # Scale the loss and backpropagate
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
            batch_loss = loss.item() * images.size(0)
            running_loss += batch_loss
            
            # Update training progress bar
            train_pbar.set_postfix({"batch_loss": f"{batch_loss / images.size(0):.4f}"})
            
            # Force GPU synchronization periodically to prevent memory buildup
            if torch.cuda.is_available() and train_pbar.n % 10 == 0:
                torch.cuda.synchronize()
        
        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Display GPU utilization
        if torch.cuda.is_available():
            print(f"GPU Utilization: {torch.cuda.memory_allocated(0) / torch.cuda.get_device_properties(0).total_memory * 100:.2f}%")
        
        #validation
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        val_pbar = tqdm(val_loader, desc="Validation", leave=False, position=1)
        
        with torch.no_grad():
            for images, targets in val_pbar:
                # Explicitly move tensors to device and ensure they are float tensors
                images = images.to(device, non_blocking=True).float()
                targets = targets.to(device, non_blocking=True).float()
                
                # Use mixed precision for inference as well
                if scaler is not None:
                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                
                batch_loss = loss.item() * images.size(0)
                running_loss += batch_loss
                
                # Move predictions and targets to CPU to free up GPU memory
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                val_pbar.set_postfix({"batch_loss": f"{batch_loss / images.size(0):.4f}"})
        
        val_loss = running_loss / len(val_loader.dataset)
        history['val_loss'].append(val_loss)
        
        # Calculate R² score
        r2 = r2_score(all_targets, all_preds)
        history['val_r2'].append(r2)
        
        # Check if this is the best model so far
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            counter = 0  # Reset early stopping counter
        else:
            counter += 1  # Increment counter if validation loss doesn't improve
            
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            early_stop = True
        
        epoch_time = time.time() - start_time

        epoch_pbar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_R²": f"{r2:.4f}",
            "time": f"{epoch_time:.2f}s",
            "early_stop": f"{counter}/{patience}"
        })
        
        # Break the loop if early stopping is triggered
        if early_stop:
            break
    
    # Load best model weights
    model.load_state_dict(best_model_weights)
    
    return model, history

# Evaluate the model
def evaluate_model(model, test_loader, device='cuda:0'):
    # Force check if CUDA is available
    if torch.cuda.is_available():
        device = 'cuda:0'
        torch.cuda.set_device(0)  # Ensure operations default to GPU 0
    else:
        device = 'cpu'
        
    print(f"Evaluating on device: {device}")
    model = model.to(device)
    model.eval()
    
    # Initialize mixed precision if available
    use_amp = torch.cuda.is_available() and hasattr(torch.cuda.amp, 'autocast')
    
    all_preds = []
    all_targets = []
    test_pbar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for images, targets in test_pbar:
            # Explicitly move tensors to device and ensure they are float
            images = images.to(device, non_blocking=True).float()
            
            # Use mixed precision for inference if available
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(images)
            else:
                outputs = model(images)
                
            # Move predictions to CPU to free GPU memory
            cpu_outputs = outputs.cpu().numpy()
            all_preds.extend(cpu_outputs)
            all_targets.extend(targets.numpy())
            
            # Force synchronize to prevent memory buildup
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    # Clean up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
            
    mse = mean_squared_error(all_targets, all_preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets, all_preds)
    
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test R²: {r2:.4f}")
    
    # Plot predictions vs. actual values
    plt.figure(figsize=(10, 6))
    plt.scatter(all_targets, all_preds, alpha=0.5)
    
    # Plot perfect prediction line
    min_val = min(min(all_targets), min(all_preds))
    max_val = max(max(all_targets), max(all_preds))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.xlabel('True Correlation')
    plt.ylabel('Predicted Correlation')
    plt.title('Predicted vs. True Correlation')
    plt.grid(True)
    plt.savefig('prediction_plot.png')
    plt.show()
    
    return {
        'rmse': rmse,
        'r2': r2,
        'predictions': all_preds,
        'targets': all_targets
    }

# Run Training + Evaluation
def run_pipeline(csv_file, image_dir, batch_size=32, num_epochs=15):
    # Force CUDA check and print more detailed info
    if torch.cuda.is_available():
        device = torch.device('cuda:0')  # Explicitly use GPU 0
        torch.cuda.set_device(0)  # Ensure operations default to GPU 0
        # Enable cuDNN benchmark for faster training
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        # Clear GPU cache
        torch.cuda.empty_cache()
        gc.collect()
        
        print(f"CUDA is available. Using GPU 0: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"CUDA Version: {torch.version.cuda}")
        # Print memory info
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Current Memory Usage: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        print(f"Max Memory Usage: {torch.cuda.max_memory_allocated(0) / 1e9:.2f} GB")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    
    # Optimize batch size if on GPU
    if torch.cuda.is_available():
        # Try to estimate optimal batch size based on GPU memory
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Heuristic: larger GPU memory allows larger batches
        if gpu_mem_gb > 16:
            batch_size = max(batch_size, 256)  # Use at least 256 for large GPUs
        elif gpu_mem_gb > 8:
            batch_size = max(batch_size, 128)  # Use at least 128 for medium GPUs
        
        print(f"Using batch size: {batch_size} based on available GPU memory")
    
    # Create DataLoader with optimized settings
    print("Loading dataset...")
    dataloader = get_dataloader(csv_file, image_dir, batch_size=batch_size)
    
    # Split dataset into train, validation, test
    print("Splitting dataset...")
    dataset = dataloader.dataset
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)  # For reproducibility
    )
    
    # Optimize DataLoader settings for GPU
    num_workers = 4 if torch.cuda.is_available() else 0
    prefetch_factor = 2 if torch.cuda.is_available() else None
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        pin_memory=torch.cuda.is_available(),
        num_workers=num_workers,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize the model
    print("Initializing model...")
    model = CorrelationPredictor(pretrained=True, freeze_backbone=False)
    
    # Use a more GPU-efficient optimizer
    if torch.cuda.is_available():
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
    criterion = nn.MSELoss()
    
    # Move model to GPU before training
    model = model.to(device)
    
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, patience=7, device=device
    )

    print("Plotting training history...")
    plt.figure(figsize=(12, 5))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.title('Validation R²')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')

    print("Evaluating model...")
    results = evaluate_model(trained_model, test_loader, device)
    
    # Save model
    torch.save(trained_model.state_dict(), 'correlation_model.pth')
    print("Model saved to 'correlation_model.pth'")
    
    return trained_model, results

if __name__ == "__main__":
    # Paths
    csv_file = "correlation_assignment/responses.csv"
    image_dir = "correlation_assignment/images"
    
    # Run the pipeline
    model, results = run_pipeline(
        csv_file=csv_file, 
        image_dir=image_dir,
        batch_size=32,
        num_epochs=100  # Early stopping will likely trigger before reaching 100 epochs
    )