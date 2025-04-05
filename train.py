import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import time
from tqdm import tqdm
from dataset import get_dataloader  # Assuming dataset.py is in the same directory
# Set random seed for reproducibility
torch.manual_seed(42)


# Create a model class based on a pre-trained ResNet
class CorrelationPredictor(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=False):
        """
        Args:
            pretrained (bool): If True, uses pre-trained weights
            freeze_backbone (bool): If True, freezes the backbone layers
        """
        super(CorrelationPredictor, self).__init__()
        
        # Load pre-trained ResNet model (smaller ResNet18 for faster training)
        self.backbone = models.resnet18(pretrained=pretrained)
        
        # Freeze backbone layers if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # ResNet18's fc layer input features is 512
        in_features = self.backbone.fc.in_features
        
        # Replace with a regression head
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)  # Output a single value for correlation
        )
    
    def forward(self, x):
        x = self.backbone(x)
        # Squeeze to remove extra dimension and match target
        return x.squeeze()

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, patience=10, device='cuda'):
    print("Start Training...")
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_r2': []
    }
    
    # Move model to device
    model.to(device)
    
    best_val_loss = float('inf')
    best_model_weights = None
    
    # Early stopping parameters
    counter = 0
    early_stop = False
    
    # Create epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)
    
    for epoch in epoch_pbar:
        start_time = time.time()
        
        # Training phase
        model.train()
        running_loss = 0.0
        
        # Create training batch progress bar
        train_pbar = tqdm(train_loader, desc="Training", leave=False, position=1)
        
        for images, targets in train_pbar:
            images, targets = images.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            batch_loss = loss.item() * images.size(0)
            running_loss += batch_loss
            
            # Update training progress bar
            train_pbar.set_postfix({"batch_loss": f"{batch_loss / images.size(0):.4f}"})
        
        train_loss = running_loss / len(train_loader.dataset)
        history['train_loss'].append(train_loss)
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        # Create validation batch progress bar
        val_pbar = tqdm(val_loader, desc="Validation", leave=False, position=1)
        
        with torch.no_grad():
            for images, targets in val_pbar:
                images, targets = images.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                batch_loss = loss.item() * images.size(0)
                running_loss += batch_loss
                
                # Store predictions and targets for metrics
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
                # Update validation progress bar
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
            
        # Check for early stopping
        if counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            early_stop = True
        
        epoch_time = time.time() - start_time
        
        # Update epoch progress bar with summary info
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
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_targets = []
    
    # Create test batch progress bar
    test_pbar = tqdm(test_loader, desc="Testing")
    
    with torch.no_grad():
        for images, targets in test_pbar:
            images = images.to(device)
            outputs = model(images)
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.numpy())
    
    # Calculate metrics
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

# Main function to run the training and evaluation pipeline
def run_pipeline(csv_file, image_dir, batch_size=32, num_epochs=15):
    # Check for GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create DataLoader
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
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize the model
    print("Initializing model...")
    model = CorrelationPredictor(pretrained=True, freeze_backbone=False)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train model with early stopping
    trained_model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=num_epochs, patience=7, device=device
    )
    
    # Plot training history
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
    
    # Plot R²
    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Validation R²')
    plt.xlabel('Epoch')
    plt.ylabel('R²')
    plt.legend()
    plt.title('Validation R²')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    # plt.show()
    
    # Evaluate model
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
        batch_size=128,
        num_epochs=100  # Early stopping will likely trigger before reaching 100 epochs
    )