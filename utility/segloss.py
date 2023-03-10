# Credit: Institute for Artificial Intelligence in Medicine.
# url: https://mml.ikim.nrw/

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as nnf
import torchvision
from typing import List

class ExampleSegmentationLoss(torch.nn.Module):

    """
    An example loss module for segmentation.
    The forward function expects predictions as a single tensor, and targets as a list of tensors.
    Different classes can weighted using the optional argument w_l, which should be a tensor.
    """

    def __init__(self, classes: int, w_l: torch.Tensor = None):
        super(ExampleSegmentationLoss, self).__init__()
        self.classes = classes

        # Weight classes, default to 1 to 1 weighting of classes
        if w_l is None:
            w_l = torch.Tensor([1 for c in range(self.classes)])
        self.XE = nn.CrossEntropyLoss(weight = w_l, reduction = "mean")

    def forward(self, predictions: torch.Tensor, targets: List[torch.Tensor,]):

        # Since LiTS does not come with target masks for the background, we can:
        # 1) Improvise the background target by making a tensor that has background everywhere and subtracting all other masks.
        # 2) Make the background target ourselves and read it from disk like all other targets.
        # Here, we opt for option 1.
        all_targets = [torch.ones_like(targets[0]).to(targets[0].device)]
        all_targets.extend(targets)

        # Sometimes, two targets exist for a pixel. For example, a liver tumor and liver may overlap, depending on your dataset.
        # Generally speaking, this should not be the case. If there is a lot of overlap, then our neural network would have to
        # learn the concept of objects being partially obscured by other objects - a difficult task! 
        # We will pretend here that we don't know if there is any overlap, and that we do not want any overlap.

        # First, we stack all our targets along the channel dimension in reverse order.
        # Second, we take the argmax of this tensor along the channel dimension. Essentially, we ask "which target is in this pixel?"
        # This way, if there is a "tie" (two target masks give a value of 1 for a pixel), the highest class index is chosen.
        # In our example, this means if there is a lesion, it always trumps liver.
        # Conveniently, this also means that any non-background class always trumps background.
        # Thirdly, we compute self.classes - our argmax - 1. This way we restore the original class indices (currently, they are still reversed because of our argmax trick).
        c_targets = torch.squeeze(self.classes - torch.argmax(torch.stack(tensors = all_targets[::-1], dim = 1), dim = 1) - 1)

        # Now, we want to compute the cross entropy loss
        xe_loss = self.XE.forward(
            torch.moveaxis(predictions, 1, -1).flatten(end_dim = -2), 
            c_targets.flatten()
            )
        return xe_loss

        # If you are confused, try to print out the sizes and/or contents for each step so that you know what happens.
        # You can even split apart lines into smaller pieces and print out the sizes or contents there, too.