import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from PIL import Image
import time


# Custom Dataset for CSV files.
class CIFAR10CSV(Dataset):
    def __init__(self, csv_file, transform=None, test_mode=False):
        """
        Args:
            csv_file (str): Path to the CSV file.
            transform (callable, optional): Transformations to apply on the image.
            test_mode (bool): If True, the dataset will not expect a label column.
        """
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.test_mode = test_mode

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        if self.test_mode:
            # In test mode, only "id" is provided along with pixel columns.
            img_id = row['id']
            pixels = row.drop(labels=['id']).values.astype(np.uint8)
        else:
            label = int(row['label'])
            pixels = row.drop(labels=['id', 'label']).values.astype(np.uint8)
        # Reshape the flat array (3072 values) into a 32x32x3 image.
        image_array = pixels.reshape((32, 32, 3))
        image = Image.fromarray(image_array)
        if self.transform:
            image = self.transform(image)
        if self.test_mode:
            return image, img_id
        else:
            return image, label

class VGG16(nn.Module):
    def __init__(self, num_classes=10):
        super(VGG16, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 1 * 1, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        
        # print("Shape before flattening:", out.shape)  # Add this line to debug the shape
        
        out = out.view(out.size(0), -1)  # Flatten
        # print("Flattened shape:", out.shape)  # Debugging
        
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

            
            
# Training function.
def train(model, device, train_loader, optimizer, criterion, epoch, histor):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch}: Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")
    sys.stdout.flush()
    # Store metrics
    history["train_loss"].append(epoch_loss)
    history["train_acc"].append(epoch_acc)

    


# Validation function.
def validate(model, device, val_loader, criterion, epoch, histor):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    print(f"Epoch {epoch}: Val Loss: {epoch_loss:.4f}, Val Acc: {epoch_acc:.4f}")
    sys.stdout.flush
    # Store metrics
    history["val_loss"].append(epoch_loss)
    history["val_acc"].append(epoch_acc)


import time

def test(model, device, test_loader, output_csv_path):
    model.eval()
    predictions = []
    ids = []
    latencies = []

    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)
            
            # Measure latency for each batch
            for i in range(images.size(0)):  # Process one image at a time for latency measurement
                single_image = images[i].unsqueeze(0)  # Add batch dimension
                start_time = time.time()
                output = model(single_image)
                end_time = time.time()
                
                latency = (end_time - start_time) * 1000  # Convert to milliseconds
                _, predicted = torch.max(output, 1)
                
                predictions.append(predicted.item())
                ids.append(img_ids[i].item())
                latencies.append(latency)

    # Create a DataFrame for submission
    output_df = pd.DataFrame({'id': ids, 'label': predictions, 'latency': latencies})
    
    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"Test results saved to {output_csv_path}")

    return latencies



def plot_and_save(history, save_path="training_plot.png"):
    epochs = range(1, len(history["train_loss"]) + 1)
    
    plt.figure(figsize=(12, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training & Validation Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training & Validation Accuracy")
    plt.legend()

    # Save plot
    plt.savefig(save_path)
    print(f"Training plot saved as {save_path}")
    # plt.show()

    # Call this function after training
    plot_and_save(history, save_path=f"{root}/training_plot.png")



def main():
    
    root = "/work/nayeem/Neuromorphic/A_1.2"
    output_csv_path = f"{root}/submission.csv"
    
    # Use CIFAR-10 normalization values.
    cifar10_mean = (0.4914, 0.4822, 0.4465)
    cifar10_std = (0.2023, 0.1994, 0.2010)

    # Define transforms.
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
    ])
    
    # Create datasets.
    train_dataset = CIFAR10CSV(csv_file='/work/nayeem/Neuromorphic/A_1.2/train.csv', transform=transform_train, test_mode=False)
    val_dataset   = CIFAR10CSV(csv_file='/work/nayeem/Neuromorphic/A_1.2/val.csv', transform=transform_train, test_mode=False)
    test_dataset  = CIFAR10CSV(csv_file='/work/nayeem/Neuromorphic/A_1.2/test.csv', transform=transform_test, test_mode=True)
    
    # Create DataLoaders.
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    sys.stdout.flush()
    
    model = VGG16(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    # Switch to SGD with momentum.
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
    
    # Track history
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    
    num_epochs = 30  # You might need more epochs when training from scratch.
    
    for epoch in range(1, num_epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch, history)
        validate(model, device, val_loader, criterion, epoch, history)
        # Save checkpoint every 5 epochs.
        if epoch % 5 == 0:
            checkpoint_path = f"vgg16_epoch_{epoch}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    plot_and_save(history, save_path=f"{root}/training_plot.png")

    # Load final model for testing
    final_checkpoint = f"{root}/vgg16_epoch_{num_epochs}.pth"
    model.load_state_dict(torch.load(final_checkpoint, map_location=device))
    print(f"Model loaded from {final_checkpoint} for testing.")

    # Compute test accuracy and latency
    latencies = test(model, device, test_loader, output_csv_path)

    # Compute average latency
    avg_latency = sum(latencies) / len(latencies)

    # Compute test accuracy
    correct = sum([1 for _, pred in enumerate(pd.read_csv(output_csv_path)['label']) if pred == test_dataset[i][1]])
    total = len(test_dataset)
    test_accuracy = correct / total

    # Compute final SCORE
    SCORE = test_accuracy / avg_latency

    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Average Latency: {avg_latency:.6f} ms")
    print(f"Final SCORE: {SCORE:.6f}")


if __name__ == '__main__':
    main()
        
    
    

