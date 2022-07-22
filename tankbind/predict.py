import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import hashlib

from tqdm import tqdm
# from helper_functions import *
import rdkit.Chem as Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from tqdm import tqdm
import glob
import torch
# %matplotlib inline
import logging
import argparse
from datetime import datetime

from data import *
from torch_geometric.loader import DataLoader
from model import *
from metrics import *
from utils import *
import sys
from torch.utils.data import RandomSampler

parser = argparse.ArgumentParser(description='Go Infinity and Beyond')
parser.add_argument("-m", "--mode", type=int, default=0,
                    help="mode specify the model to use.")
parser.add_argument("-d", "--data", type=str, default="0",
                    help="specify the data to evaulate.")
parser.add_argument("--batch_size", type=int, default=1,
                    help="batch size.")

parser.add_argument("--model", type=str, default=None,
                    help="load the saved model.")

parser.add_argument("--posweight", type=int, default=16,
                    help="pos weight in loss.")
parser.add_argument("--pred_dis", type=int, default=1,
                    help="pred distance map or predict contact map.")

args = parser.parse_args()



timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

# handlers = [logging.FileHandler(f'predict_{timestamp}.log'), logging.StreamHandler()]
# logger = logging.basicConfig(level=logging.INFO, format='%(message)s', handlers=handlers)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("")
handler = logging.FileHandler(f'eval_{timestamp}.log')
handler.setFormatter(logging.Formatter('%(message)s', ""))
logger.addHandler(handler)

logging.info(f'''\
{' '.join(sys.argv)}
{timestamp}
evaluation using trained model.
--------------------------------
''')

pre = f"./backup/{timestamp}"
os.system(f"mkdir -p {pre}/models")
os.system(f"mkdir -p {pre}/results")
os.system(f"mkdir -p {pre}/src")
os.system(f"cp *.py {pre}/src/")

torch.set_num_threads(1)
# # ----------without this, I got 'RuntimeError: received 0 items of ancdata'-----------
torch.multiprocessing.set_sharing_strategy('file_system')

global_test_loader = None

if args.data == "1":
    logging.info(f"protein feature GVP, compound feature torchdrug. ligand conformation generated using RDKit")
    new_dataset = TankBindDataSet("./dataset/apr23_testset_pdbbind_gvp_pocket_radius20/", proteinMode=0, compoundMode=1, pocket_radius=20, predDis=True)
    new_dataset.compound_dict = torch.load("./dataset/pdbbind_test_compound_dict_based_on_rdkit.pt")
    valid = new_dataset
    test = new_dataset
    info = pd.read_csv("./dataset/apr23_testset_pdbbind_gvp_pocket_radius20_info.csv", index_col=0)


logging.info(f"data point, valid: {len(valid)}, test: {len(test)}")


valid_loader = DataLoader(valid, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, num_workers=8, pin_memory=True)
test_loader = DataLoader(test, batch_size=args.batch_size, follow_batch=['x', 'compound_pair'], shuffle=False, num_workers=8, pin_memory=True)
# torch.cuda.set_device(0)


device = 'cuda'
# first_data = next(iter(loader)).to(device)
# model = IaBNet().to(device)
model = get_model(args.mode, logging, device)

if args.model:
    model.load_state_dict(torch.load(args.model))
# with torch.no_grad():  # Initialize lazy modules.
#      out = model(data)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
# model.train()
if args.pred_dis:
    criterion = nn.MSELoss()
    pred_dis = True
else:
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.posweight))
affinity_criterion = nn.MSELoss()
relative_k = 0.1

metrics_list = []
valid_metrics_list = []
test_metrics_list = []

epoch = 0
model.eval()
# metrics = evaulate_with_affinity(valid_loader, model, criterion, affinity_criterion, relative_k, device)
# logging.info(f"epoch {epoch:<4d}, valid, " + print_metrics(metrics))

saveFileName = f"{pre}/results/epoch_{epoch}.pt"
metrics = evaulate_with_affinity(test_loader, model, criterion, affinity_criterion, relative_k,
                                    device, pred_dis=pred_dis, info=info, saveFileName=saveFileName)
logging.info(f"epoch {epoch:<4d}, test,  " + print_metrics(metrics))

os.system(f"cp eval_{timestamp}.log {pre}/")
