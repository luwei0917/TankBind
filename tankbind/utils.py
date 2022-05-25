
import torch
from metrics import *
import numpy as np
import pandas as pd
import scipy.spatial
from torch_geometric.data import Data
from torch_geometric.data import HeteroData
import torch.nn.functional as F
from tqdm import tqdm
import torchmetrics

def read_pdbbind_data(fileName):
    with open(fileName) as f:
        a = f.readlines()
    info = []
    for line in a:
        if line[0] == '#':
            continue
        lines, ligand = line.split('//')
        pdb, resolution, year, affinity, raw = lines.strip().split('  ')
        ligand = ligand.strip().split('(')[1].split(')')[0]
        # print(lines, ligand)
        info.append([pdb, resolution, year, affinity, raw, ligand])
    info = pd.DataFrame(info, columns=['pdb', 'resolution', 'year', 'affinity', 'raw', 'ligand'])
    info.year = info.year.astype(int)
    info.affinity = info.affinity.astype(float)
    return info

def compute_dis_between_two_vector(a, b):
    return (((a - b)**2).sum())**0.5

def get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode):
    # protein
    input_edge_list = []
    input_protein_edge_feature_idx = []
    new_node_index = np.cumsum(keepNode) - 1
    keepEdge = keepNode[protein_edge_index].min(axis=0)
    new_edge_inex = new_node_index[protein_edge_index]
    input_edge_idx = torch.tensor(new_edge_inex[:, keepEdge], dtype=torch.long)
    input_protein_edge_s = protein_edge_s[keepEdge]
    input_protein_edge_v = protein_edge_v[keepEdge]
    return input_edge_idx, input_protein_edge_s, input_protein_edge_v

def get_keepNode(com, protein_node_xyz, n_node, pocket_radius, use_whole_protein, 
                     use_compound_com_as_pocket, add_noise_to_com, chosen_pocket_com):
    if use_whole_protein:
        keepNode = np.ones(n_node, dtype=bool)
    else:
        keepNode = np.zeros(n_node, dtype=bool)
        # extract node based on compound COM.
        if use_compound_com_as_pocket:
            if add_noise_to_com:
                com = com + add_noise_to_com * (2 * np.random.rand(*com.shape) - 1)
            for i, node in enumerate(protein_node_xyz):
                dis = compute_dis_between_two_vector(node, com)
                keepNode[i] = dis < pocket_radius

    if chosen_pocket_com is not None:
        another_keepNode = np.zeros(n_node, dtype=bool)
        for a_com in chosen_pocket_com:
            if add_noise_to_com:
                a_com = a_com + add_noise_to_com * (2 * np.random.rand(*a_com.shape) - 1)
            for i, node in enumerate(protein_node_xyz):
                dis = compute_dis_between_two_vector(node, a_com)
                another_keepNode[i] |= dis < pocket_radius
        keepNode |= another_keepNode
    return keepNode


def construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                  protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                                 coords, compound_node_features, input_atom_edge_list, 
                                 input_atom_edge_attr_list, includeDisMap=True, contactCutoff=8.0, pocket_radius=20, interactionThresholdDistance=10, compoundMode=1, 
                                 add_noise_to_com=None, use_whole_protein=False, use_compound_com_as_pocket=True, chosen_pocket_com=None):
    n_node = protein_node_xyz.shape[0]
    n_compound_node = coords.shape[0]
    # centroid instead of com. 
    com = coords.mean(axis=0)
    keepNode = get_keepNode(com, protein_node_xyz.numpy(), n_node, pocket_radius, use_whole_protein, 
                             use_compound_com_as_pocket, add_noise_to_com, chosen_pocket_com)

    if keepNode.sum() < 5:
        # if only include less than 5 residues, simply add first 100 residues.
        keepNode[:100] = True
    input_node_xyz = protein_node_xyz[keepNode]
    input_edge_idx, input_protein_edge_s, input_protein_edge_v = get_protein_edge_features_and_index(protein_edge_index, protein_edge_s, protein_edge_v, keepNode)

    # construct graph data.
    data = HeteroData()

    # only if your ligand is real this y_contact is meaningful.
    dis_map = scipy.spatial.distance.cdist(input_node_xyz.cpu().numpy(), coords)
    y_contact = dis_map < contactCutoff
    if includeDisMap:
        # treat all distance above 10A as the same.
        dis_map[dis_map>interactionThresholdDistance] = interactionThresholdDistance
        data.dis_map = torch.tensor(dis_map, dtype=torch.float).flatten()

    # additional information. keep records.
    data.node_xyz = input_node_xyz
    data.coords = torch.tensor(coords, dtype=torch.float)
    data.y = torch.tensor(y_contact, dtype=torch.float).flatten()
    data.seq = protein_seq[keepNode]
    data['protein'].node_s = protein_node_s[keepNode] # [num_protein_nodes, num_protein_feautre]
    data['protein'].node_v = protein_node_v[keepNode]
    data['protein', 'p2p', 'protein'].edge_index = input_edge_idx
    data['protein', 'p2p', 'protein'].edge_s = input_protein_edge_s
    data['protein', 'p2p', 'protein'].edge_v = input_protein_edge_v

    if compoundMode == 0:
        data['compound'].x = torch.tensor(compound_node_features, dtype=torch.bool)  # [num_compound_nodes, num_compound_feature]
        data['compound', 'c2c', 'compound'].edge_index = torch.tensor(input_atom_edge_list, dtype=torch.long).t().contiguous()
        c2c = torch.tensor(input_atom_edge_attr_list, dtype=torch.long)
        data['compound', 'c2c', 'compound'].edge_attr = F.one_hot(c2c-1, num_classes=1)  # [num_edges, num_edge_features]
    elif compoundMode == 1:
        data['compound'].x = compound_node_features
        data['compound', 'c2c', 'compound'].edge_index = input_atom_edge_list[:,:2].long().t().contiguous()
        data['compound', 'c2c', 'compound'].edge_weight = torch.ones(input_atom_edge_list.shape[0])
        data['compound', 'c2c', 'compound'].edge_attr = input_atom_edge_attr_list
    return data, input_node_xyz, keepNode


def my_affinity_criterion(y_pred, y, mask, decoy_gap=1.0):
    affinity_loss = torch.zeros(y_pred.shape).to(y_pred.device)
    affinity_loss[mask] = (((y_pred - y)**2)[mask])
    affinity_loss[~mask] = (((y_pred - (y - decoy_gap)).relu())**2)[~mask]
    return affinity_loss.mean()

def evaulate(data_loader, model, criterion, device, saveFileName=None):
    y_list = []
    y_pred_list = []
    batch_loss = 0.0
    for data in data_loader:
        data = data.to(device)
        y_pred = model(data)
        with torch.no_grad():
            loss = criterion(y_pred, data.y)
        batch_loss += len(y_pred)*loss.item()
        y_list.append(data.y)
        y_pred_list.append(y_pred.sigmoid().detach())
        # torch.cuda.empty_cache()
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    metrics = {"loss":batch_loss/len(y_pred)}
    metrics.update(myMetric(y_pred, y))
    if saveFileName:
        torch.save((y, y_pred), saveFileName)
    return metrics

def is_ligand_pocket(pdb):
    if len(pdb) == 4:
        return True
    else:
        return False
    

def select_pocket_by_predicted_affinity(info):
    info['is_ligand_pocket'] = info.pdb.apply(lambda x:is_ligand_pocket(x))
    pdb_to_num_contact = info.query("is_ligand_pocket").set_index("pdb")['num_contact'].to_dict()
    info['base_pdb'] = info.pdb.apply(lambda x: x.split("_")[0])
    info['native_num_contact'] = info.base_pdb.apply(lambda x: pdb_to_num_contact[x])
    info['cover_contact_ratio'] = info['num_contact'] / info['native_num_contact']
    use_whole_protein_list = set(info.base_pdb.unique()) - set(info.query("not is_ligand_pocket").base_pdb)
    # assume we don't know the true ligand binding site.
    selected = info.query("not is_ligand_pocket or (base_pdb in @use_whole_protein_list)").sort_values(['base_pdb', 
                      'affinity_pred']).groupby("base_pdb").tail(1).sort_values("index").reset_index(drop=True)
    return selected


def compute_numpy_rmse(x, y):
    return np.sqrt(((x - y)**2).mean())

def extract_list_from_prediction(info, y, y_pred, selected=None, smiles_to_mol_dict=None, coords_generated_from_smiles=False):
    idx = 0
    y_list = []
    y_pred_list = []
    for i, line in info.iterrows():
        n_protein = line['p_length']
        n_compound = line['c_length']
        n_y = n_protein * n_compound
        if selected is None or (i in selected['index'].values):
            y_pred_list.append(y_pred[idx:idx+n_y].reshape(n_protein, n_compound))
            y_list.append(y[idx:idx+n_y].reshape(n_protein, n_compound))
        idx += n_y
    d = (y_list, y_pred_list)
    return d

def evaulate_with_affinity(data_loader, model, criterion, affinity_criterion, relative_k, device, pred_dis=False, info=None, saveFileName=None, use_y_mask=False, skip_y_metrics_evaluation=False):
    y_list = []
    y_pred_list = []
    affinity_list = []
    affinity_pred_list = []
    real_y_mask_list = []
    p_length_list = []
    c_length_list = []

    batch_loss = 0.0
    affinity_batch_loss = 0.0
    for data in tqdm(data_loader):
        protein_ptr = data['protein']['ptr']
        p_length_list += [int(protein_ptr[ptr] - protein_ptr[ptr-1]) for ptr in range(1, len(protein_ptr))]
        compound_ptr = data['compound']['ptr']
        c_length_list += [int(compound_ptr[ptr] - compound_ptr[ptr-1]) for ptr in range(1, len(compound_ptr))]

        data = data.to(device)
        y_pred, affinity_pred = model(data)
        y = data.y
        dis_map = data.dis_map
        if use_y_mask:
            y_pred = y_pred[data.real_y_mask]
            y = y[data.real_y_mask]
            dis_map = dis_map[data.real_y_mask]
        with torch.no_grad():
            if pred_dis:
                contact_loss = criterion(y_pred, dis_map) if len(dis_map) > 0 else torch.tensor([0]).to(dis_map.device)
            else:
                contact_loss = criterion(y_pred, y) if len(y) > 0 else torch.tensor([0]).to(y.device)
                y_pred = y_pred.sigmoid()
            affinity_loss = relative_k * affinity_criterion(affinity_pred, data.affinity)
            loss = contact_loss + affinity_loss
        batch_loss += len(y_pred)*contact_loss.item()
        affinity_batch_loss += len(affinity_pred)*affinity_loss.item()
        y_list.append(y)
        y_pred_list.append(y_pred.detach())
        affinity_list.append(data.affinity)
        affinity_pred_list.append(affinity_pred.detach())
        real_y_mask_list.append(data.real_y_mask)
        # torch.cuda.empty_cache()
    y = torch.cat(y_list)
    y_pred = torch.cat(y_pred_list)
    if pred_dis:
        y_pred = torch.clip(1 - (y_pred / 10.0), min=1e-6, max=0.99999)
        # we define 8A as the cutoff for contact, therefore, contact_threshold will be 1 - 8/10 = 0.2
        threshold = 0.2

    real_y_mask = torch.cat(real_y_mask_list)
    affinity = torch.cat(affinity_list)
    affinity_pred = torch.cat(affinity_pred_list)
    if saveFileName:
        torch.save((y, y_pred, affinity, affinity_pred), saveFileName)

    metrics = {"loss":batch_loss/len(y_pred) + affinity_batch_loss/len(affinity_pred)}
    metrics.update({"y loss":(batch_loss/len(y_pred))})
    metrics.update({"affinity loss":(affinity_batch_loss/len(affinity_pred))})
    if info is not None:
        # print(affinity, affinity_pred)
        info['affinity'] = affinity.cpu().numpy()
        info['affinity_pred'] = affinity_pred.cpu().numpy()
        selected = select_pocket_by_predicted_affinity(info)
        result = {}
        real_affinity = 'real_affinity' if 'real_affinity' in selected.columns else 'affinity'
        # result['Pearson'] = selected['affinity'].corr(selected['affinity_pred'])
        result['Pearson'] = selected[real_affinity].corr(selected['affinity_pred'])
        result['RMSE'] = compute_numpy_rmse(selected[real_affinity], selected['affinity_pred'])

        native_y = y[real_y_mask].bool()
        native_y_pred = y_pred[real_y_mask]
        native_auroc = torchmetrics.functional.auroc(native_y_pred, native_y)
        result['native_auroc'] = native_auroc

        info['p_length'] = p_length_list
        info['c_length'] = c_length_list
        y_list, y_pred_list = extract_list_from_prediction(info, y.cpu(), y_pred.cpu(), selected=selected, smiles_to_mol_dict=None, coords_generated_from_smiles=False)
        selected_y = torch.cat([y.flatten() for y in y_list]).long()
        selected_y_pred = torch.cat([y_pred.flatten() for y_pred in y_pred_list])
        selected_auroc = torchmetrics.functional.auroc(selected_y_pred, selected_y)
        result['selected_auroc'] = selected_auroc

        for i in [90, 80, 50]:
            # cover ratio, CR.
            result[f'CR_{i}'] = (selected.cover_contact_ratio > i / 100).sum() / len(selected)
        metrics.update(result)
    if not skip_y_metrics_evaluation:
        metrics.update(myMetric(y_pred, y, threshold=threshold))
    metrics.update(affinity_metrics(affinity_pred, affinity))
    return metrics

def evaulate_affinity_only(data_loader, model, criterion, affinity_criterion, relative_k, device, info=None, saveFileName=None, use_y_mask=False):
    y_list = []
    y_pred_list = []
    affinity_list = []
    affinity_pred_list = []
    batch_loss = 0.0
    affinity_batch_loss = 0.0
    for data in tqdm(data_loader):
        data = data.to(device)
        y_pred, affinity_pred = model(data)
        y = data.y
        if use_y_mask:
            y_pred = y_pred[data.real_y_mask]
            y = y[data.real_y_mask]
        with torch.no_grad():
            contact_loss = criterion(y_pred, y) if len(y) > 0 else torch.tensor([0]).to(y.device)
            affinity_loss = relative_k * affinity_criterion(affinity_pred, data.affinity)
            loss = contact_loss + affinity_loss
        batch_loss += len(y_pred)*contact_loss.item()
        affinity_batch_loss += len(affinity_pred)*affinity_loss.item()

        affinity_list.append(data.affinity)
        affinity_pred_list.append(affinity_pred.detach())
        # torch.cuda.empty_cache()

    affinity = torch.cat(affinity_list)
    affinity_pred = torch.cat(affinity_pred_list)
    if saveFileName:
        torch.save((None, None, affinity, affinity_pred), saveFileName)
    metrics = {"loss":affinity_batch_loss/len(affinity_pred)}
    if info is not None:
        # print(affinity, affinity_pred)
        info['affinity'] = affinity.cpu().numpy()
        info['affinity_pred'] = affinity_pred.cpu().numpy()
        selected = select_pocket_by_predicted_affinity(info)
        result = {}
        real_affinity = 'real_affinity' if 'real_affinity' in selected.columns else 'affinity'
        # result['Pearson'] = selected['affinity'].corr(selected['affinity_pred'])
        result['Pearson'] = selected[real_affinity].corr(selected['affinity_pred'])
        result['RMSE'] = compute_numpy_rmse(selected[real_affinity], selected['affinity_pred'])
        for i in [90, 80, 50]:
            # cover ratio, CR.
            result[f'CR_{i}'] = (selected.cover_contact_ratio > i / 100).sum() / len(selected)
        metrics.update(result)

    metrics.update(affinity_metrics(affinity_pred, affinity))

    return metrics

