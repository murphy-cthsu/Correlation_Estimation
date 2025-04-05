import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


def split_dataset(csv_file, image_dir, output_dir, train_size=0.7, val_size=0.15):
    """
    Splits the dataset into training, validation, and test sets,
    copies images, and saves split CSV files.

    Args:
        csv_file (str): Path to the CSV file with correlation values.
        image_dir (str): Directory with all the scatter plot images.
        output_dir (str): Base output directory for split images and CSVs.
        train_size (float): Proportion of the dataset for training.
        val_size (float): Proportion for validation (rest goes to test).
    """
    # Load and clean CSV
    data = pd.read_csv(csv_file)
    data['id'] = data['id'].astype(str).str.strip().str.lower()

    # Check for missing images
    missing = []
    for img_id in data['id']:
        img_path = os.path.join(image_dir, f"{img_id}.png")
        if not os.path.exists(img_path):
            missing.append(img_id)
    if missing:
        print(f"\n❌ Total missing images: {len(missing)}")
        for m in missing:
            print(f"Missing: {m}.png")
        print("\n⚠️ Please fix missing images before proceeding.")
        return
    else:
        print("✅ All images are present.")

    # Split dataset
    train_data, temp_data = train_test_split(data, train_size=train_size, random_state=42)
    val_size_adjusted = val_size / (1 - train_size)
    val_data, test_data = train_test_split(temp_data, train_size=val_size_adjusted, random_state=42)

    # Paths for image folders
    train_dir = os.path.join(output_dir, "train_images")
    val_dir = os.path.join(output_dir, "val_images")
    test_dir = os.path.join(output_dir, "test_images")

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # Function to copy images
    def copy_images(data_subset, target_dir):
        for _, row in data_subset.iterrows():
            img_name = f"{row['id']}.png"
            src_path = os.path.join(image_dir, img_name)
            dst_path = os.path.join(target_dir, img_name)
            shutil.copy(src_path, dst_path)

    # Copy and save CSVs
    print("\n📂 Processing training set...")
    copy_images(train_data, train_dir)
    train_data.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    print("✅ Training set ready.")

    print("📂 Processing validation set...")
    copy_images(val_data, val_dir)
    val_data.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    print("✅ Validation set ready.")

    print("📂 Processing test set...")
    copy_images(test_data, test_dir)
    test_data.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    print("✅ Test set ready.")

    print("\n🎉 Dataset split, images copied, and CSVs saved successfully!")


if __name__ == "__main__":
    # Set paths
    base_dir = "correlation_assignment"
    csv_file = os.path.join(base_dir, "responses.csv")
    image_dir = os.path.join(base_dir, "images")
    output_dir = base_dir  # Saves everything inside correlation_assignment/

    split_dataset(csv_file, image_dir, output_dir)
