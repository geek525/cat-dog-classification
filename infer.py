import random
import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
from model import create_model
from data import get_test_loader

def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_checkpoint_path(device):
    if device.type == "cuda":
        return f"resnet50_catdog_GPU.pth"
    else:
        return f"resnet50_catdog_CPU.pth"


def load_model(device, checkpoint_path):
    if checkpoint_path is None:
        checkpoint_path = get_checkpoint_path(device)

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Model file not found: {checkpoint_path}")

    print(f"Loading model from: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device)

    model = create_model()
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_full_testset_accuracy(checkpoint_path=None, batch_size=32):
    device = get_device()
    try:
        model = load_model(device, checkpoint_path)
    except FileNotFoundError as e:
        print(e)
        return

    test_loader = get_test_loader(batch_size=batch_size)
    if len(test_loader) == 0:
        print("Test loader is empty.")
        return

    correct = 0
    total = 0

    print("Running full test set evaluation...")
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total * 100
    print(f"[Full Testset] Accuracy: {acc:.4f}%")
    return acc


def infer_single_random_test_image(checkpoint_path=None):
    device = get_device()
    try:
        model = load_model(device, checkpoint_path)
    except FileNotFoundError:
        return

    test_loader = get_test_loader(batch_size=1)
    dataset = test_loader.dataset
    if len(dataset) == 0:
        return

    idx = random.randrange(len(dataset))
    image, true_label_idx = dataset[idx]           # image: Tensor(C,H,W) after eval_transforms
    image_batch = image.unsqueeze(0).to(device)    # (1,C,H,W)

    with torch.no_grad():
        logits = model(image_batch)
        probs = torch.softmax(logits, dim=1).squeeze(0)  # (num_classes,)
        confidence, pred_label_idx = torch.max(probs, 0)

    class_names = dataset.classes
    img_path = dataset.samples[idx][0]

    result = {
        "img_path": img_path,
        "true_label": class_names[true_label_idx],
        "pred_label": class_names[pred_label_idx.item()],
        "confidence": confidence.item()
    }

    print("\n[Single Random Test Image]")
    print(f"Image: {os.path.basename(result['img_path'])}")
    print(f"True: {result['true_label']}")
    print(f"Pred: {result['pred_label']} (Confidence: {result['confidence']:.4f})")

    img = Image.open(img_path).convert("RGB")
    probs_np = probs.detach().cpu().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.imshow(img)
    ax1.axis("off")
    ax1.set_title(f"True: {result['true_label']}\nPred: {result['pred_label']} ({result['confidence']:.2%})")

    colors = ['green' if name == result['pred_label'] else 'blue' for name in class_names]
    ax2.bar(class_names, probs_np, color=colors)
    ax2.set_ylim(0, 1.1)
    ax2.set_ylabel("Confidence")
    ax2.set_title("Prediction Probability")

    for i, v in enumerate(probs_np):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.tight_layout()
    plt.show()

    return result


if __name__ == "__main__":
    infer_full_testset_accuracy()
    infer_single_random_test_image()