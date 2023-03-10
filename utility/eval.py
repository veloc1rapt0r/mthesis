# Credit: Institute for Artificial Intelligence in Medicine.
# url: https://mml.ikim.nrw/

from typing import List
from collections import OrderedDict
import torch
import torch.nn
import torch.nn.functional as nnf

def evaluate_classifier_model(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader, device: str = None):
    """
    Evaluates a classifier on a given dataset.
    The model is expected to return one-hot predictions as first element.
    Any other value that is returned is ignored.
    The dataloader is expected to return at least data and targets as first and second elements.
    Any other value that is returned is ignored.
    """
    if device is not None:
        model = model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        seen = []
        test_losses = []
        hits = 0
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get data and target, throw away any other returns from the dataloader.
        for data, targets, *_ in dataloader:
            if device is not None:
                data, targets = data.to(device), targets.to(device)
            model_returns = model(data)
            
            # Get predictions from model. If there is other returns, toss them.
            if isinstance(model_returns, tuple) and len(model_returns) > 1:
                oh_predictions, *_ = model_returns
            else:
                oh_predictions = model_returns

            loss = criterion(oh_predictions, targets)
            c_predictions = torch.argmax(oh_predictions, dim=1)
            hits += sum([1 if p == t else 0 for p, t in zip(c_predictions, targets)])
            seen.append(targets.size()[0])
            test_losses.append(loss.item())

        accuracy = hits/sum(seen)
        avg_test_loss = sum([l*s for l, s in zip(test_losses, seen)])/sum(seen)

    return accuracy, avg_test_loss

class Segmentation_Metrics():

    """
    Computes:

    Weighted Dice Score. 
    Weighted IoU.
    Weighted Precision.
    Weighted Recall.
    Targets must be a List of List of Tensors.
    (Outer list has batches, inner list has each target as a separate tensor.)
    """

    def __init__(self, verbose: bool = False, **kwargs):
        self.results = OrderedDict()
        self.verbose = verbose
        self.kwargs = kwargs
        self.weights = self.kwargs.get("weights", None)

    def forward(self, predictions: List[torch.Tensor, ], targets: List[torch.Tensor, ]):
        eps = 1e-6 # for stability reasons
        weighted_avg_dice = 0
        weighted_avg_iou = 0
        weighted_avg_precision = 0
        weighted_avg_recall = 0
        seen = 0
        bsl = {}

        for b in range(len(targets)):
            nt = len(targets[b])
            # Convert predictions to binary one-hot format for proper scoring
            p_arg = nnf.one_hot(torch.argmax(predictions[b].to(torch.float32), dim = 1), num_classes = nt+1).moveaxis(-1, 1)
            
            for t in range(nt+1):
                if t == 0:
                    # Build the background label target on the fly
                    all_targets = [torch.ones_like(targets[b][t]).to(targets[b][t].device)]
                    all_targets.extend(targets[b])
                    # Convert to onehot, last class index always has priority if two masks match in one location
                    c_targets = torch.squeeze(nt - torch.argmax(torch.stack(tensors = all_targets[::-1], dim = 1), dim = 1))
                    oh_targets = nnf.one_hot(c_targets, num_classes = 3).moveaxis(-1, 1)
                    target = oh_targets[:, t, :, :].type(torch.bool)
                else:
                    # All other targets already exist
                    target = oh_targets[:, t, :, :].type(torch.bool)
                
                prediction = p_arg[:, t, :, :].type(torch.bool)
                intersection = torch.sum(prediction * target)
                p_cardinality = torch.sum(prediction)
                t_cardinality = torch.sum(target)
                cardinality = p_cardinality + t_cardinality
                union = torch.sum((prediction + target))

                bs = target.size()[0]
                bsl[f"{b}_{t}"] = bs
                if self.weights is None:
                    weight = 1
                else:
                    weight = self.weights[t]
                seen += bs * weight

                # Dice Score
                if intersection.item() == 0 and cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    dice_score = 1.
                else:
                    # Regular case
                    dice_score = (2. * intersection / (cardinality + eps)).item()

                self.results[f"#dice_{b}_{t}"] = dice_score
                weighted_avg_dice += self.results[f"#dice_{b}_{t}"] * bs * weight
                #####

                # IoU
                if intersection.item() == 0 and union.item() == 0:
                    # Special case where we match an all-empty target
                    iou = 1.
                else:
                    # Regular case
                    iou = (intersection / (union + eps)).item() # DEBUG 2?

                self.results[f"#iou_{b}_{t}"] = iou
                weighted_avg_iou += self.results[f"#iou_{b}_{t}"] * bs * weight
                #####

                # Precision
                if intersection.item() == 0 and p_cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    precision = 1.
                else:
                    # Regular case
                    precision = (intersection / (p_cardinality + eps)).item()

                self.results[f"#precision_{b}_{t}"] = precision
                weighted_avg_precision += self.results[f"#precision_{b}_{t}"] * bs * weight
                #####

                # Recall
                if intersection.item() == 0 and t_cardinality.item() == 0:
                    # Special case where we match an all-empty target
                    recall = 1.
                else:
                    # Regular case
                    recall = (intersection / (t_cardinality + eps)).item()

                self.results[f"#recall_{b}_{t}"] = recall
                weighted_avg_recall += self.results[f"#recall_{b}_{t}"] * bs * weight
                #####

        self.results["#weighted_dice_avg"] = weighted_avg_dice / seen
        self.results["#weighted_iou_avg"] = weighted_avg_iou / seen
        self.results["#weighted_precision_avg"] = weighted_avg_precision / seen
        self.results["#weighted_recall_avg"] = weighted_avg_recall / seen

        for t in range(nt+1):
            total = sum([v for b, v in bsl.items() if b.endswith(str(t))])
            self.results[f"dice_avg_class_{t}"] = sum([self.results[f"#dice_{b}_{t}"] * bsl[f"{b}_{t}"] for b in range(len(targets))]) / total
            self.results[f"iou_avg_class_{t}"] = sum([self.results[f"#iou_{b}_{t}"] * bsl[f"{b}_{t}"] for b in range(len(targets))]) / total
            self.results[f"precision_avg_class_{t}"] = sum([self.results[f"#precision_{b}_{t}"] * bsl[f"{b}_{t}"] for b in range(len(targets))]) / total
            self.results[f"recall_avg_class_{t}"] = sum([self.results[f"#recall_{b}_{t}"] * bsl[f"{b}_{t}"] for b in range(len(targets))]) / total

        return self.results

def evaluate_segmentation_model(model, dataloader, device, w_l = None):
    
    """
    Evaluates a segmentation model on a given dataset.
    The model is expected to return one-hot predictions as first element.
    Any other value that is returned is ignored.
    The dataloader is expected to return at least data and targets as first and second elements.
    Any other value that is returned is ignored.
    """

    if device is not None:
        model = model.to(device)
    
    with torch.no_grad():
        model.eval()
        
        pl, tl = [], []
        Seg_Metrics = Segmentation_Metrics(weights = w_l)
        
        # Get data and target, throw away any other returns from the dataloader.
        for data, targets, *_ in dataloader:
            if device is not None:
                data = data.to(device)
                targets = [target.to(device) for target in targets]
            model_returns = model(data)
            
            # Get predictions from model. If there is other returns, toss them.
            if isinstance(model_returns, tuple) and len(model_returns) > 1:
                oh_predictions, *_ = model_returns
            else:
                oh_predictions = model_returns

            pl.extend([oh_predictions])
            tl.extend([targets])

        metrics = Seg_Metrics.forward(pl, tl)

    return metrics