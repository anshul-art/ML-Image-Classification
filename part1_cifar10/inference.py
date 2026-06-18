import os
import torch
import cv2
from train import build_model, get_device
from torchvision import transforms
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'best_model.pth')
DATA_PATH = os.path.join(SCRIPT_DIR, 'data', 'filtered', 'cifar10_filtered.pt')

def infer(image_path):
    device = get_device()
    dataset_dict = torch.load(DATA_PATH)
    class_names = dataset_dict['class_names']
    sorted_keys = sorted(list(class_names.keys()))
    class_labels = [class_names[k] for k in sorted_keys]
    
    model = build_model(len(class_names), device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        _, pred = torch.max(out, 1)
        
    print(f"Prediction: {class_labels[pred.item()]} ({(probs[pred.item()]*100):.2f}%)")
    for i, label in enumerate(class_labels):
        print(f"{label}: {probs[i]*100:.2f}%")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        infer(sys.argv[1])
    else:
        print("Usage: python inference.py <image_path>")
