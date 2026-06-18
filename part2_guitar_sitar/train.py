import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import models, transforms, datasets
from torch.utils.tensorboard import SummaryWriter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'models')
LOG_DIR = os.path.join(SCRIPT_DIR, 'logs', 'guitar_sitar')
PLOT_PATH = os.path.join(SCRIPT_DIR, 'training_plot.png')

EPOCHS_WARMUP = 3
EPOCHS_FINE = 7
BATCH_SIZE = 16
NUM_CLASSES = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def get_device():
    if torch.cuda.is_available(): return torch.device('cuda')
    try:
        import torch_directml
        return torch_directml.device()
    except:
        return torch.device('cpu')

def train():
    device = get_device()
    print(f"Using device: {device}")
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), transform=transform_train)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'), transform=transform_test)
    
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, NUM_CLASSES)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=LOG_DIR)
    
    best_val_acc = 0.0
    
    for epoch in range(1, EPOCHS_WARMUP + EPOCHS_FINE + 1):
        if epoch == 1:
            for p in model.parameters(): p.requires_grad = False
            for p in model.classifier.parameters(): p.requires_grad = True
            opt = optim.Adam(model.classifier.parameters(), lr=1e-3)
        elif epoch == EPOCHS_WARMUP + 1:
            for p in model.parameters(): p.requires_grad = True
            opt = optim.Adam(model.parameters(), lr=1e-4)
            
        model.train()
        for inputs, labels in train_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            out = model(inputs)
            loss = criterion(out, labels)
            loss.backward()
            opt.step()
            
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_dl:
                inputs, labels = inputs.to(device), labels.to(device)
                out = model(inputs)
                _, pred = torch.max(out, 1)
                total += labels.size(0)
                correct += (pred == labels).sum().item()
                
        val_acc = correct / total
        print(f"Epoch {epoch}: Val Acc {val_acc*100:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'best_model.pth'))
            
if __name__ == '__main__':
    train()
