import torch
from torch import nn, optim
import time
import copy

from model import create_model
from data import get_train_val_loaders


def train():
    EPOCHS = 50
    LR = 1e-3
    BATCH_SIZE = 32
    PATIENCE = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    model = create_model()
    model.to(device)

    train_loader, val_loader = get_train_val_loaders(batch_size=BATCH_SIZE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=LR)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    min_val_loss = float('inf')
    early_stop_counter = 0

    print(f"Start training resnet50...")

    for e in range(EPOCHS):
        start = time.time()

        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)

                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_acc = correct / total * 100

        epoch_time = time.time() - start
        print(f"Epoch {e + 1}/{EPOCHS} | Time: {epoch_time:.0f}s | "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(), f"resnet50_catdog_GPU.pth")
            print(f"--> New Best Model Saved! (Acc: {best_acc:.2f}%)")

        if avg_val_loss < min_val_loss:
            min_val_loss = avg_val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= PATIENCE:
                print(f"Early stopping triggered after {e + 1} epochs.")
                break

    print(f"\nTraining Finished. Best Validation Accuracy: {best_acc:.2f}%")

    model.load_state_dict(best_model_wts)
    model.to('cpu')
    torch.save(model.state_dict(), f"resnet50_catdog_CPU.pth")
    print(f"CPU inference model saved to resnet50_catdog_CPU.pth")


if __name__ == "__main__":
    train()