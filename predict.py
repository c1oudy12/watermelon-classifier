import argparse
import torch
from torchvision import transforms, models
from PIL import Image
import cv2
import numpy as np
import os


def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    classes = checkpoint.get('classes', None)
    num_classes = len(classes) if classes else 3
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device).eval()
    return model, classes


def preprocess(image_path):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    img = Image.open(image_path).convert('RGB')
    return img, transform(img).unsqueeze(0)


def annotate_and_save(pil_img, label_text, out_path):
    # convert PIL to BGR for OpenCV
    arr = np.array(pil_img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    cv2.putText(bgr, label_text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imwrite(out_path, bgr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='path to image')
    parser.add_argument('--model', default='saved/watermelon_resnet18.pth')
    parser.add_argument('--out', default='result.jpg')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    model, classes = load_model(args.model, device)
    pil_img, tensor = preprocess(args.image)
    tensor = tensor.to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.nn.functional.softmax(logits, dim=1)[0]
        topidx = probs.argmax().item()
        label = classes[topidx] if classes else str(topidx)
        confidence = probs[topidx].item()
    label_text = f"{label} {confidence*100:.1f}%"
    print('Predicted:', label_text)
    annotate_and_save(pil_img, label_text, args.out)
    print('Saved annotated image to', args.out)

if __name__ == '__main__':
    main()
