# Implementation: Oleh Bakumenko, Univerity of Duisburg-Essen

import torch
import torch.nn

def calculate_confusion_matrix(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = None):
    """
        Computes  confusion matrix for 3 class segmentation
        Inputs:
            model: classification model
            dataloader: source of the images
        Outputs:
            confusion_matrix: 3*3 int Torch tensor
            per_class_accuracy: 1*3 Torch tensor
    """
    if device is not None:
        model = model.to(device)
        
    confusion_matrix = torch.zeros(3, 3)
    with torch.no_grad():
        for i, (img, targets) in enumerate(dataloader):
            if device is not None:
                img = img.to(device)
                targets = targets.to(device)
            outputs = model(img)
            preds = torch.argmax(outputs, 1)
            for t, p in zip(targets.view(-1), preds.view(-1)):
                    confusion_matrix[t.int(), p.int()] += 1
    per_class_accuracy = confusion_matrix.diag()/confusion_matrix.sum(1)
    return confusion_matrix.int(), per_class_accuracy
