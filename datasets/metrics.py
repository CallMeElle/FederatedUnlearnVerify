"""
Evaluation metrics
"""
from torch.nn import functional as F
import copy
import os
import torch
from typing import Tuple
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds)) * 100

def validation_step(model, batch, device):
    images, labels, clabels = batch
    images, clabels = images.to(device), clabels.to(device)
    out = model(images)  # Generate predictions
    loss = F.cross_entropy(out, clabels)  # Calculate loss
    acc = accuracy(out, clabels)  # Calculate accuracy
    return {"Loss": loss.detach(), "Acc": acc}

def validation_epoch_end(model, outputs):
    batch_losses = [x["Loss"] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()  # Combine losses
    batch_accs = [x["Acc"] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()  # Combine accuracies
    return {"Loss": epoch_loss.item(), "Acc": epoch_acc.item()}

@torch.no_grad()
def evaluate(model, val_loader, device):
    model.eval()
    outputs = [validation_step(model, batch, device) for batch in val_loader]
    return validation_epoch_end(model, outputs)


def metrics_fl(
        retain_client_trainloader,
        unlearn_client_trainloader,
        retain_client_testloader,
        unlearn_client_testloader,
        global_model,
        device
) -> Tuple[float, float, float, float]:
    retain_client_train_acc = evaluate(val_loader=retain_client_trainloader,
                                       model=copy.deepcopy(global_model),
                                       device=device)["Acc"]
    unlearn_client_train_acc = evaluate(val_loader=unlearn_client_trainloader,
                                        model=copy.deepcopy(global_model),
                                        device=device)["Acc"]
    retain_client_test_acc = evaluate(val_loader=retain_client_testloader,
                                      model=copy.deepcopy(global_model),
                                      device=device)["Acc"]
    unlearn_client_test_acc = evaluate(val_loader=unlearn_client_testloader,
                                       model=copy.deepcopy(global_model),
                                       device=device)["Acc"]
    return round(retain_client_train_acc, 4), round(unlearn_client_train_acc,4), round(retain_client_test_acc,4), round(unlearn_client_test_acc,4)

def metrics_unlearn(
        unlearn_client_trainloader,
        retain_client_trainloader,
        testloader,
        model_unlearn,
        device
) -> Tuple[float, float, float]:
    unlearn_client_acc = evaluate(val_loader=unlearn_client_trainloader,
                                  model= copy.deepcopy(model_unlearn),
                                  device= device)["Acc"]

    retain_client_acc = evaluate(val_loader= retain_client_trainloader,
                                 model= copy.deepcopy(model_unlearn),
                                 device= device)["Acc"]

    test_acc = evaluate(val_loader= testloader,
                        model= copy.deepcopy(model_unlearn),
                        device= device)['Acc']

    return round(unlearn_client_acc, 4), round(retain_client_acc, 4), round(test_acc, 4)