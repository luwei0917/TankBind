
import torchmetrics
import torch
from torch import nn
import pandas as pd

def myMetric(y_pred, y, threshold=0.5):
    y = y.float()
    criterion = nn.BCELoss()
    with torch.no_grad():
        loss = criterion(y_pred, y)

    # y = y.long()
    y = y.bool()
    acc = torchmetrics.functional.accuracy(y_pred, y, threshold=threshold)
    auroc = torchmetrics.functional.auroc(y_pred, y)
    precision_0, precision_1 = torchmetrics.functional.precision(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    recall_0, recall_1 = torchmetrics.functional.recall(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    f1_0, f1_1 = torchmetrics.functional.f1_score(y_pred, y, 
                                      num_classes=2, multiclass=True,
                                      average='none', threshold=threshold)
    return {"BCEloss":loss.item(),
            "acc":acc, "auroc":auroc, "precision_1":precision_1,
           "recall_1":recall_1, "f1_1":f1_1,"precision_0":precision_0,
           "recall_0":recall_0, "f1_0":f1_0}

def affinity_metrics(affinity_pred, affinity):
    pearson = torchmetrics.functional.pearson_corrcoef(affinity_pred, affinity)
    rmse = torchmetrics.functional.mean_squared_error(affinity_pred, affinity, squared=False)
    return {"pearson":pearson, "rmse":rmse}
    

def print_metrics(metrics):
    out_list = []
    for key in metrics:
        out_list.append(f"{key}:{metrics[key]:6.3f}")
    out = ", ".join(out_list)
    return out


def compute_individual_metrics(pdb_list, inputFile_list, y_list):
    r_ = []
    for i in range(len(pdb_list)):
        pdb = pdb_list[i]
        # inputFile = f"{pre}/input/{pdb}.pt"
        inputFile = inputFile_list[i]
        y = y_list[i]
        (coords, y_pred, protein_nodes_xyz, 
         compound_pair_dis_constraint, pdb, sdf_fileName, mol2_fileName, pre) = torch.load(inputFile)
        result = myMetric(torch.tensor(y_pred).reshape(-1), y.reshape(-1))
        for key in result:
            result[key] = float(result[key])
        result['idx'] = i
        result['pdb'] = pdb
        result['p_length'] = protein_nodes_xyz.shape[0]
        result['c_length'] = coords.shape[0]
        result['y_length'] = y.reshape(-1).shape[0]
        result['num_contact'] = int(y.sum())
        r_.append(result)
    result = pd.DataFrame(r_)
    return result

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report