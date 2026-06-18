"""
CIFAR-10 MobileNetV2 Training
==============================
Trains a MobileNetV2 model on the filtered CIFAR-10 dataset using
two phases (warm-up -> fine-tune).  Logs to TensorBoard.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, 'data', 'filtered', 'cifar10_filtered.pt')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'models')
LOG_DIR    = os.path.join(SCRIPT_DIR, 'logs', 'cifar10_mobilenet')
PLOT_PATH  = os.path.join(SCRIPT_DIR, 'training_plot.png')

EPOCHS_WARMUP = 5
EPOCHS_FINE = 10
BATCH_SIZE = 32
LR_WARMUP = 1e-3
LR_FINE = 1e-4
NUM_CLASSES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


class CustomCIFARDataset(Dataset):
    def __init__(self, data, targets, class_map, transform=None):
        self.data = data
        # remap targets to 0..4
        sorted_keys = sorted(list(class_map.keys()))
        mapping = {old: new for new, old in enumerate(sorted_keys)}
        self.targets = [mapping[t] for t in targets]
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        target = self.targets[idx]
        
        # PIL Image expected by transforms
        from PIL import Image
        img = Image.fromarray(img)
        
        if self.transform:
            img = self.transform(img)
        return img, target

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    try:
        import torch_directml
        return torch_directml.device()
    except:
        return torch.device('cpu')

def build_model(num_classes, device):
    weights = models.MobileNet_V2_Weights.DEFAULT
    model = models.mobilenet_v2(weights=weights)
    
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model.to(device)

def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, writer, global_step):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        writer.add_scalar('Batch/Loss', loss.item(), global_step)
        global_step += 1
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, global_step

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
    return running_loss / total, correct / total

def main():
    print("=" * 60)
    print("  MOBILENET V2 -> CIFAR-10 (5 CLASSES)")
    print("=" * 60)
    
    device = get_device()
    print(f"\\n[Device]\\n  Using {device}")
    
    # Load dataset
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
        
    dataset_dict = torch.load(DATA_PATH)
    class_names = dataset_dict['class_names']
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    train_ds = CustomCIFARDataset(dataset_dict['train_data'], dataset_dict['train_targets'], class_names, transform=transform_train)
    val_ds = CustomCIFARDataset(dataset_dict['val_data'], dataset_dict['val_targets'], class_names, transform=transform_test)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = build_model(NUM_CLASSES, device)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    global_step = 0
    best_val_acc = 0.0
    
    total_epochs = EPOCHS_WARMUP + EPOCHS_FINE
    
    for epoch in range(1, total_epochs + 1):
        if epoch == 1:
            print("\\n[Phase 1] Warm-up: Training Classifier Head Only")
            for param in model.parameters():
                param.requires_grad = False
            for param in model.classifier.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.classifier.parameters(), lr=LR_WARMUP)
            
        elif epoch == EPOCHS_WARMUP + 1:
            print("\\n[Phase 2] Fine-tuning: Unfreezing entire model")
            for param in model.parameters():
                param.requires_grad = True
            optimizer = optim.Adam(model.parameters(), lr=LR_FINE)
            
        t0 = time.time()
        tr_loss, tr_acc, global_step = train_one_epoch(model, train_dl, criterion, optimizer, device, epoch, writer, global_step)
        va_loss, va_acc = validate(model, val_dl, criterion, device)
        t1 = time.time()
        
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)
        
        writer.add_scalar('Epoch/Train_Loss', tr_loss, epoch)
        writer.add_scalar('Epoch/Val_Loss', va_loss, epoch)
        writer.add_scalar('Epoch/Train_Acc', tr_acc, epoch)
        writer.add_scalar('Epoch/Val_Acc', va_acc, epoch)
        
        print(f"Epoch {epoch:2d}/{total_epochs} [{t1-t0:.1f}s] - Train loss: {tr_loss:.4f} acc: {tr_acc*100:.2f}% | Val loss: {va_loss:.4f} acc: {va_acc*100:.2f}%")
        
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            
    # Plotting
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.title('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Val')
    plt.title('Accuracy')
    plt.legend()
    plt.savefig(PLOT_PATH)
    print(f"\\nTraining complete! Best Val Acc: {best_val_acc*100:.2f}%")
    print(f"Plots saved to {PLOT_PATH}")

if __name__ == '__main__':
    main()
