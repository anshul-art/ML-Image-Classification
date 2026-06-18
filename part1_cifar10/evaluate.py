import os
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from train import build_model, get_device, CustomCIFARDataset
from torchvision import transforms

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'filtered', 'cifar10_filtered.pt')
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'best_model.pth')
CM_PATH = os.path.join(SCRIPT_DIR, 'confusion_matrix.png')

def evaluate():
    device = get_device()
    dataset_dict = torch.load(DATA_PATH)
    class_names = dataset_dict['class_names']
    sorted_keys = sorted(list(class_names.keys()))
    class_labels = [class_names[k] for k in sorted_keys]
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    test_ds = CustomCIFARDataset(dataset_dict['test_data'], dataset_dict['test_targets'], class_names, transform=transform_test)
    test_dl = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    model = build_model(len(class_names), device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_dl:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    cm = confusion_matrix(all_labels, all_preds)
    
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall: {rec*100:.2f}%")
    
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_labels, yticklabels=class_labels, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix - CIFAR-10')
    plt.savefig(CM_PATH)
    print(f"Confusion matrix saved -> {CM_PATH}")

if __name__ == '__main__':
    evaluate()
