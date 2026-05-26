import argparse
import os
from tqdm import tqdm
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split


def main():
    parser = argparse.ArgumentParser(description="Train watermelon classifier")
    parser.add_argument('--data', default='dataset/train', help='train data directory')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--out', default='saved/watermelon_resnet18.pth')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Use ImageFolder; for simplicity we split the dataset (same transform used for both splits here)
    full_dataset = datasets.ImageFolder(args.data, transform=transform_train)
    num_classes = len(full_dataset.classes)
    print('Found classes:', full_dataset.classes)

    # train/val split
    val_ratio = 0.1
    n = len(full_dataset)
    if n < 2:
        raise RuntimeError('Not enough images in dataset. Add images into dataset/train/<class>/')
    val_size = max(1, int(n * val_ratio))
    train_size = n - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False, num_workers=2)

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [train]')
        for imgs, labels in loop:
            imgs = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * imgs.size(0)
            loop.set_postfix(loss=running_loss / ((loop.n+1) * args.batch))

        scheduler.step()

        # validation
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total if total>0 else 0
        print(f'Epoch {epoch+1} validation loss: {val_loss/total:.4f} acc: {acc:.4f}')

        # save best
        if acc > best_acc:
            best_acc = acc
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({'model_state': model.state_dict(), 'classes': full_dataset.classes}, args.out)
            print('Saved best model to', args.out)

    print('Training finished. Best val acc:', best_acc)

if __name__ == '__main__':
    main()
