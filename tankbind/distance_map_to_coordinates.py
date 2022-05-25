# conda activate py39
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib

from tqdm import tqdm
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm
import glob
import torch
from torch import nn
import argparse
parser = argparse.ArgumentParser(description='Generate Ligand Coordinates based on Inter-molecular distance map, \
                                                intra-molecular distance map of compound, and coornidates of protein nodes.')
parser.add_argument("inputFile", type=str, 
                    help="input torch file.")
parser.add_argument("-t", "--to", type=str, default=".",
                    help="mode specify the data to save.")
parser.add_argument("-m", "--mode", type=int, default=0,
                    help="mode specify the model to use.")
parser.add_argument("-r", "--repeat", type=int, default=1,
                    help="repeat optimization r times.")
parser.add_argument("--use_LAS_distance_constraint_mask", action='store_true',
                    help="used for flexible docking.")
args = parser.parse_args()


from generation_utils import *
from datetime import datetime
import logging
from io import StringIO
import sys

# # limit the cpu cores used by this program.
torch.set_num_threads(1)

def batch_run(inputFile, toFolder, mode, use_LAS_distance_constraint_mask=False, n_repeat=1):
    (coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, pdb, sdf_fileName, mol2_fileName, pre) = torch.load(inputFile)
    if os.path.exists(f"{toFolder}/info/{pdb}.pkl"):
        return None
    device = 'cpu'
    coords = torch.tensor(coords).to(device)
    y_pred = torch.tensor(y_pred) if type(y_pred) != torch.Tensor else y_pred
    protein_nodes_xyz = torch.tensor(protein_nodes_xyz) if type(protein_nodes_xyz) != torch.Tensor else protein_nodes_xyz
    compound_pair_dis_constraint = torch.tensor(compound_pair_dis_constraint)

    if use_LAS_distance_constraint_mask:
        mol, _ = read_mol(sdf_fileName, mol2_fileName)
        LAS_distance_constraint_mask = get_LAS_distance_constraint_mask(mol).bool()
    else:
        LAS_distance_constraint_mask = None

    if mode == 0:
        info = get_info_pred_distance(coords, y_pred, protein_nodes_xyz, compound_pair_dis_constraint, LAS_distance_constraint_mask=LAS_distance_constraint_mask, n_repeat=n_repeat, show_progress=False)

    toFile = f'{toFolder}/sdf/{pdb}.sdf'
    new_coords = info.sort_values("loss")['coords'].iloc[0].astype(np.double)
    mol, _ = read_mol(sdf_fileName, mol2_fileName)
    write_with_new_coords(mol, new_coords, toFile)
    info.to_pickle(f"{toFolder}/info/{pdb}.pkl")


batch_run(args.inputFile, args.to, args.mode, n_repeat=args.repeat, use_LAS_distance_constraint_mask=args.use_LAS_distance_constraint_mask)