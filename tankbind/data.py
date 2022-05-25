import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Dataset, InMemoryDataset, download_url
from utils import construct_data_from_graph_gvp

class TankBind_prediction(Dataset):
    def __init__(self, root, data=None, protein_dict=None, compound_dict=None, proteinMode=0, compoundMode=1,
                pocket_radius=20, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.protein_dict = torch.load(self.processed_paths[1])
        self.compound_dict = torch.load(self.processed_paths[2])
        self.proteinMode = proteinMode
        self.pocket_radius = pocket_radius
        self.compoundMode = compoundMode
        self.shake_nodes = shake_nodes
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt', 'compound.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])
        torch.save(self.compound_dict, self.processed_paths[2])

    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        pocket_com = line['pocket_com']
        pocket_com = np.array(pocket_com.split(",")).astype(float) if type(pocket_com) == str else pocket_com
        pocket_com = pocket_com.reshape((1, 3))
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False

        protein_name = line['protein_name']
        protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[protein_name]

        compound_name = line['compound_name']
        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = self.compound_dict[compound_name]

        # y is distance map, instead of contact map.
        data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                              protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                              coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list,
                              pocket_radius=self.pocket_radius, use_whole_protein=use_whole_protein, includeDisMap=True,
                              use_compound_com_as_pocket=False, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)

        return data

class TankBindDataSet(Dataset):
    def __init__(self, root, data=None, protein_dict=None, compound_dict=None, proteinMode=0, compoundMode=1,
                add_noise_to_com=None, pocket_radius=20, contactCutoff=8.0, predDis=True, shake_nodes=None,
                 transform=None, pre_transform=None, pre_filter=None):
        self.data = data
        self.protein_dict = protein_dict
        self.compound_dict = compound_dict
        super().__init__(root, transform, pre_transform, pre_filter)
        print(self.processed_paths)
        self.data = torch.load(self.processed_paths[0])
        self.protein_dict = torch.load(self.processed_paths[1])
        self.compound_dict = torch.load(self.processed_paths[2])
        self.add_noise_to_com = add_noise_to_com
        self.proteinMode = proteinMode
        self.compoundMode = compoundMode
        self.pocket_radius = pocket_radius
        self.contactCutoff = contactCutoff
        self.predDis = predDis
        self.shake_nodes = shake_nodes
    @property
    def processed_file_names(self):
        return ['data.pt', 'protein.pt', 'compound.pt']

    def process(self):
        torch.save(self.data, self.processed_paths[0])
        torch.save(self.protein_dict, self.processed_paths[1])
        torch.save(self.compound_dict, self.processed_paths[2])

    def len(self):
        return len(self.data)

    def get(self, idx):
        line = self.data.iloc[idx]
        # uid = line['uid']
        # smiles = line['smiles']
        pocket_com = line['pocket_com']
        use_compound_com = line['use_compound_com']
        use_whole_protein = line['use_whole_protein'] if "use_whole_protein" in line.index else False
        group = line['group'] if "group" in line.index else 'train'
        add_noise_to_com = self.add_noise_to_com if group == 'train' else None

        protein_name = line['protein_name']
        if self.proteinMode == 0:
            # protein embedding follow GVP protocol.
            protein_node_xyz, protein_seq, protein_node_s, protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v = self.protein_dict[protein_name]

        name = line['compound_name']
        coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, pair_dis_distribution = self.compound_dict[name]

        # node_xyz could add noise too.
        shake_nodes = self.shake_nodes if group == 'train' else None
        if shake_nodes is not None:
            protein_node_xyz = protein_node_xyz + shake_nodes * (2 * np.random.rand(*protein_node_xyz.shape) - 1)
            coords = coords  + shake_nodes * (2 * np.random.rand(*coords.shape) - 1)

        if self.proteinMode == 0:
            data, input_node_list, keepNode = construct_data_from_graph_gvp(protein_node_xyz, protein_seq, protein_node_s, 
                                  protein_node_v, protein_edge_index, protein_edge_s, protein_edge_v,
                                  coords, compound_node_features, input_atom_edge_list, input_atom_edge_attr_list, contactCutoff=self.contactCutoff, includeDisMap=self.predDis,
                          pocket_radius=self.pocket_radius, add_noise_to_com=add_noise_to_com, use_whole_protein=use_whole_protein, 
                          use_compound_com_as_pocket=use_compound_com, chosen_pocket_com=pocket_com, compoundMode=self.compoundMode)

        # affinity = affinity_to_native_pocket * min(1, float((data.y.numpy() > 0).sum()/(5*coords.shape[0])))
        affinity = float(line['affinity'])
        data.affinity = torch.tensor([affinity], dtype=torch.float)
        data.compound_pair = pair_dis_distribution.reshape(-1, 16)
        data.pdb = line['pdb'] if "pdb" in line.index else f'smiles_{idx}'
        data.group = group

        data.real_affinity_mask = torch.tensor([use_compound_com], dtype=torch.bool)
        data.real_y_mask = torch.ones(data.y.shape).bool() if use_compound_com else torch.zeros(data.y.shape).bool()
        # fract_of_native_contact = float(line['fract_of_native_contact']) if "fract_of_native_contact" in line.index else 1
        # equivalent native pocket
        if "native_num_contact" in line.index:
            fract_of_native_contact = (data.y.numpy() > 0).sum() / float(line['native_num_contact'])
            is_equivalent_native_pocket = fract_of_native_contact >= 0.9
            data.is_equivalent_native_pocket = torch.tensor([is_equivalent_native_pocket], dtype=torch.bool)
            data.equivalent_native_y_mask = torch.ones(data.y.shape).bool() if is_equivalent_native_pocket else torch.zeros(data.y.shape).bool()
        else:
            # native_num_contact information is not available.
            # use ligand com to determine if this pocket is equivalent to native pocket.
            if "ligand_com" in line.index:
                ligand_com = line["ligand_com"]
                pocket_com = data.node_xyz.numpy().mean(axis=0)
                dis = np.sqrt(((ligand_com - pocket_com)**2).sum())
                # is equivalent native pocket if ligand com is less than 8 A from pocket com.
                is_equivalent_native_pocket = dis < 8
                data.is_equivalent_native_pocket = torch.tensor([is_equivalent_native_pocket], dtype=torch.bool)
                data.equivalent_native_y_mask = torch.ones(data.y.shape).bool() if is_equivalent_native_pocket else torch.zeros(data.y.shape).bool()
            else:
                # data.is_equivalent_native_pocket and data.equivalent_native_y_mask will not be available.
                pass
        return data


def get_data(data_mode, logging, addNoise=None):
    pre = "./"
    if data_mode == "0":
        logging.info(f"re-docking, using dataset: apr22_pdbbind_gvp_pocket_radius20 pred distance map.")
        logging.info(f"compound feature based on torchdrug")
        add_noise_to_com = float(addNoise) if addNoise else None

        # compoundMode = 1 is for GIN model.
        new_dataset = TankBindDataSet(f"{pre}/apr22_pdbbind_gvp_pocket_radius20", add_noise_to_com=add_noise_to_com)
        # load compound features extracted using torchdrug.
        # new_dataset.compound_dict = torch.load(f"{pre}/compound_dict.pt")
        new_dataset.data = new_dataset.data.query("c_length < 100 and native_num_contact > 5").reset_index(drop=True)
        d = new_dataset.data
        only_native_train_index = d.query("use_compound_com and group =='train'").index.values
        train = new_dataset[only_native_train_index]
        train_index = d.query("group =='train'").index.values
        train_after_warm_up = new_dataset[train_index]
        # train = torch.utils.data.ConcatDataset([train1, train2])
        valid_index = d.query("use_compound_com and group =='valid'").index.values
        valid = new_dataset[valid_index]
        test_index = d.query("use_compound_com and group =='test'").index.values
        test = new_dataset[test_index]

        all_pocket_test_fileName = f"{pre}/apr23_testset_pdbbind_gvp_pocket_radius20/"
        all_pocket_test = TankBindDataSet(all_pocket_test_fileName)
        all_pocket_test.compound_dict = torch.load(f"{pre}/compound_dict.pt")
        # info is used to evaluate the test set. 
        info = None
        # info = pd.read_csv(f"{pre}/apr23_testset_pdbbind_gvp_pocket_radius20_info.csv", index_col=0)

    if data_mode == "1":
        logging.info(f"self-docking, same as data mode 0 except using LAS_distance constraint masked compound pair distance")
        add_noise_to_com = float(addNoise) if addNoise else None

        # compoundMode = 1 is for GIN model.
        new_dataset = TankBindDataSet(f"{pre}/apr22_pdbbind_gvp_pocket_radius20", add_noise_to_com=add_noise_to_com)
        # load GIN embedding for compounds.
        new_dataset.compound_dict = torch.load(f"{pre}/pdbbind_compound_dict_with_LAS_distance_constraint_mask.pt")
        new_dataset.data = new_dataset.data.query("c_length < 100 and native_num_contact > 5").reset_index(drop=True)
        d = new_dataset.data
        only_native_train_index = d.query("use_compound_com and group =='train'").index.values
        train = new_dataset[only_native_train_index]
        # train = train1
        train_index = d.query("group =='train'").index.values
        train_after_warm_up = new_dataset[train_index]

        # train = torch.utils.data.ConcatDataset([train1, train2])
        valid_index = d.query("use_compound_com and group =='valid'").index.values
        valid = new_dataset[valid_index]
        test_index = d.query("use_compound_com and group =='test'").index.values
        test = new_dataset[test_index]

        all_pocket_test_fileName = f"{pre}/apr23_testset_pdbbind_gvp_pocket_radius20/"
        all_pocket_test = TankBindDataSet(all_pocket_test_fileName)
        all_pocket_test.compound_dict = torch.load(f"{pre}/pdbbind_test_compound_dict_based_on_rdkit.pt")
        # info is used to evaluate the test set.
        info = None
        # info = pd.read_csv(f"{pre}/apr23_testset_pdbbind_gvp_pocket_radius20_info.csv", index_col=0)


    return train, train_after_warm_up, valid, test, all_pocket_test, info