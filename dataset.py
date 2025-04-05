import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import time
class CorrelationDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (str): Path to the CSV file with correlation values.
            image_dir (str): Directory with all the scatter plot images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.data = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.image_dir, f"{self.data.iloc[idx, 0]}.png")
        image = Image.open(img_name).convert("RGB")
        correlation = torch.tensor(self.data.iloc[idx, 1], dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, correlation

# Define the dataset and dataloader
def get_dataloader(csv_file, image_dir, batch_size=32, shuffle=True, num_workers=4):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize images to 224x224
        transforms.ToTensor(),         # Convert images to tensors
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize
    ])

    dataset = CorrelationDataset(csv_file=csv_file, image_dir=image_dir, transform=transform)
    
    print(f"Number of samples in dataset: {len(dataset)}")
    time.sleep(2)  # Pause for 2 seconds to allow user to see the output
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return dataloader


if __name__ == "__main__":

    # Paths
    csv_file = "correlation_assignment/responses.csv"
    image_dir = "correlation_assignment/images"  # Directory containing scatter plot images

    # Create DataLoader
    dataloader = get_dataloader(csv_file, image_dir, batch_size=32)

    # Iterate through the DataLoader
    for images, correlations in dataloader:
        print(images.shape)  # Shape of the batch of images
        print(correlations)  # Corresponding correlation values