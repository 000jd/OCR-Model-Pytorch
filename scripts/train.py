import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
from torchsummary import summary
import matplotlib.pyplot as plt
from models.ocr_model import OCRModel
from dataset import OCRDataset

# Training function
def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    train_loss = []
    train_accuracy = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100. * correct / total
        train_loss.append(epoch_loss)
        train_accuracy.append(epoch_accuracy)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

    return train_loss, train_accuracy

def plot_metrics(train_loss, train_accuracy, output_dir):
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure()
    plt.plot(epochs, train_loss, 'r', label='Training loss')
    plt.title('Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_loss.png'))
    
    plt.figure()
    plt.plot(epochs, train_accuracy, 'b', label='Training accuracy')
    plt.title('Training accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'training_accuracy.png'))

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load dataset
    data_dir = 'data/train'  
    dataset = OCRDataset(data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Initialize the model
    num_classes = len(dataset.class_to_idx)
    model = OCRModel(num_classes).to(device)

    # Print model summary
    summary(model, (1, 28, 28))

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

    # Train the model
    train_loss, train_accuracy = train_model(model, train_loader, criterion, optimizer, device, num_epochs=20)

    # Save the trained model
    if not os.path.exists('run'):
        os.makedirs('run')
    torch.save(model.state_dict(), 'run/ocr_model.pth')

    # Save loss and accuracy metrics
    with open('run/training_metrics.txt', 'w') as f:
        for epoch in range(len(train_loss)):
            f.write(f'Epoch {epoch+1}, Loss: {train_loss[epoch]:.4f}, Accuracy: {train_accuracy[epoch]:.2f}%\n')

    # Plot and save loss and accuracy graphs
    plot_metrics(train_loss, train_accuracy, 'run')

if __name__ == '__main__':
    main()
