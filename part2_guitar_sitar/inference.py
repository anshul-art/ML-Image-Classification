import os
import torch
import cv2
from torchvision import models, transforms
from train import get_device
import torch.nn as nn
from PIL import Image
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'best_model.pth')

def infer(image_path):
    device = get_device()
    class_names = ['Guitar', 'Sitar'] # Usually ordered alphabetically by ImageFolder
    
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model = model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    
    img = Image.open(image_path).convert('RGB')
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        out = model(input_tensor)
        probs = torch.nn.functional.softmax(out, dim=1)[0]
        _, pred = torch.max(out, 1)
        
    cls = class_names[pred.item()]
    prob = probs[pred.item()]*100
    
    print(f"Prediction: {cls} ({prob:.2f}%)")
    
    # OpenCV overlay
    cv_img = cv2.imread(image_path)
    text = f"{cls}: {prob:.1f}%"
    cv2.putText(cv_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    out_path = f"pred_{os.path.basename(image_path)}"
    cv2.imwrite(out_path, cv_img)
    print(f"Saved -> {out_path}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        infer(sys.argv[1])
    else:
        print("Usage: python inference.py <image_path>")
