import msgpack
import os
import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, Dataset, SequentialSampler
import argparse
from qm9.data import collate as qm9_collate


def extract_conformers(args):
    drugs_file = os.path.join(args.data_dir, args.data_file)
    save_file = f"geom_drugs_{'no_h_' if args.remove_h else ''}{args.conformations}"
    smiles_list_file = 'geom_drugs_smiles.txt'
    number_atoms_file = f"geom_drugs_n_{'no_h_' if args.remove_h else ''}{args.conformations}"

    unpacker = msgpack.Unpacker(open(drugs_file, "rb"))

    all_smiles = []
    all_number_atoms = []
    dataset_conformers = []
    mol_id = 0
    for i, drugs_1k in enumerate(unpacker):
        print(f"Unpacking file {i}...")
        for smiles, all_info in drugs_1k.items():
            all_smiles.append(smiles)
            conformers = all_info['conformers']
            # Get the energy of each conformer. Keep only the lowest values
            all_energies = []
            for conformer in conformers:
                all_energies.append(conformer['totalenergy'])
            all_energies = np.array(all_energies)
            argsort = np.argsort(all_energies)
            lowest_energies = argsort[:args.conformations]
            for id in lowest_energies:
                conformer = conformers[id]
                coords = np.array(conformer['xyz']).astype(float)        # n x 4   [atomic_num, x, y, z]
                if args.remove_h:
                    mask = coords[:, 0] != 1.0    # hydrogen's atomic_num = 1
                    coords = coords[mask]
                n = coords.shape[0]
                all_number_atoms.append(n)
                mol_id_arr = mol_id * np.ones((n, 1), dtype=float)
                id_coords = np.hstack((mol_id_arr, coords))

                dataset_conformers.append(id_coords)
                mol_id += 1

    print("Total number of conformers saved", mol_id)
    all_number_atoms = np.array(all_number_atoms)
    dataset = np.vstack(dataset_conformers)

    print("Total number of atoms in the dataset", dataset.shape[0])
    print("Average number of atoms per molecule", dataset.shape[0] / mol_id)

    # Save conformations
    np.save(os.path.join(args.data_dir, save_file), dataset)
    # Save SMILES
    with open(os.path.join(args.data_dir, smiles_list_file), 'w') as f:
        for s in all_smiles:
            f.write(s)
            f.write('\n')

    # Save number of atoms per conformation
    np.save(os.path.join(args.data_dir, number_atoms_file), all_number_atoms)
    print("Dataset processed.")



def load_split_data(conformation_file, val_proportion=0.1, test_proportion=0.1,
                    filter_size=None, permutation_file_path=None, 
                    dataset_name=None, training_mode=None, filter_pocket_size=None):
    from pathlib import Path
    path = Path(conformation_file)
    base_path = path.parent.absolute()
    # if dataset_name is None:
    #     dataset_name = 'geom'

    # base_path = os.path.dirname(conformation_file)
    all_data = np.load(conformation_file)  # 2d array: num_atoms x 5

    if training_mode is None:
        # original code (for geom & other eval & sampling scripts)
        mol_id = all_data[:, 0].astype(int)
        conformers = all_data[:, 1:]
        split_indices = np.nonzero(mol_id[:-1] - mol_id[1:])[0] + 1
        data_list = np.split(conformers, split_indices)

        # Keep only molecules <= filter_size
        if filter_size is not None:
            data_list = [molecule for molecule in data_list
                        if molecule.shape[0] <= filter_size]
            assert len(data_list) > 0, 'No molecules left after filter.'

        if permutation_file_path is not None:
            print(">> Loading permutation file from:", permutation_file_path)
            perm = np.load(permutation_file_path)
        else:
            default_permutation_file_path = os.path.join(base_path, f'{dataset_name}_permutation.npy')
            # CAREFUL! Only for first time run:
            perm = np.random.permutation(len(data_list)).astype('int32')
            print('Warning, currently taking a random permutation for '
                'train/val/test partitions, this needs to be fixed for'
                'reproducibility.')
            assert not os.path.exists(default_permutation_file_path)
            np.save(default_permutation_file_path, perm)
            
        data_list = [data_list[i] for i in perm]

        num_mol = len(data_list)
        val_index = int(num_mol * val_proportion)
        test_index = val_index + int(num_mol * test_proportion)
        print(f">> Data Splits: len(data_list):{len(data_list)},  [val_index, test_index]:{[val_index, test_index]}")
        # !!!                                      6922516    [692251, 1384502]
        # val_data, test_data, train_data = np.split(data_list, [val_index, test_index])
        val_data, test_data, train_data = data_list[:val_index], data_list[val_index:test_index], data_list[test_index:]
    
    elif training_mode == 'VAE':
        # load and combine Ligand+Pocket data
        ligand_data, pocket_data = all_data['ligand'], all_data['pocket']
        
        # ligand
        ligand_mol_id = ligand_data[:, 0].astype(int)
        ligands = ligand_data[:, 1:]
        ligand_split_indices = np.nonzero(ligand_mol_id[:-1] - ligand_mol_id[1:])[0] + 1
        ligand_data_list = np.split(ligands, ligand_split_indices)
        
        # pocket
        pocket_mol_id = pocket_data[:, 0].astype(int)
        pockets = pocket_data[:, 1:]
        pocket_split_indices = np.nonzero(pocket_mol_id[:-1] - pocket_mol_id[1:])[0] + 1
        pocket_data_list = np.split(pockets, pocket_split_indices)
        
        # Keep only molecules <= filter_size
        if filter_size is not None and filter_pocket_size is not None:
            
            assert len(list(ligand_data_list)) == len(list(pocket_data_list))
            
            tmp_ligand_data_list, tmp_pocket_data_list = [], []
            for i in range(len(list(ligand_data_list))):
                if (ligand_data_list[i].shape[0] <= filter_size) and (pocket_data_list[i].shape[0] <= filter_pocket_size):
                    tmp_ligand_data_list.append(ligand_data_list[i])
                    tmp_pocket_data_list.append(pocket_data_list[i])
            
            ligand_data_list = tmp_ligand_data_list
            pocket_data_list = tmp_pocket_data_list
            assert len(ligand_data_list) > 0, 'No molecules left after filter.'
        
        # permutation
        if permutation_file_path is not None:
            print(">> Loading permutation file from:", permutation_file_path)
            perm = np.load(permutation_file_path)
        else:
            file_name = conformation_file.split(os.path.sep)[-1][:-4]
            default_permutation_file_path = os.path.join(base_path, f'{file_name}_permutation.npy')
            # CAREFUL! Only for first time run:
            assert len(list(ligand_data_list)) == len(list(pocket_data_list)), 'Invalid Ligand-Pocket pairs'
            perm = np.random.permutation(len(ligand_data_list)).astype('int32')
            print('Warning, currently taking a random permutation for '
                'train/val/test partitions, this needs to be fixed for'
                'reproducibility.')
            assert not os.path.exists(default_permutation_file_path)
            np.save(default_permutation_file_path, perm)
        
        assert len(list(ligand_data_list)) == len(list(pocket_data_list)), 'Invalid Ligand-Pocket pairs'
        assert len(list(ligand_data_list)) == len(list(perm)), 'Invalid permutation file! Did you change [filter_size] and/or [filter_pocket_size]?'
        
        ligand_data_list = [ligand_data_list[i] for i in perm]
        pocket_data_list = [pocket_data_list[i] for i in perm]
        
        # combine both ligand+pocket together manually with [lg, pkt, lg, pkt, ..] ordering to ensure VAE sees same amount of both ligand and pockets
        all_data = []
        for i in range(len(list(ligand_data_list))):
            all_data.append(ligand_data_list[i])
            all_data.append(pocket_data_list[i])

        # split
        num_mol = len(all_data)
        val_index = int(num_mol * val_proportion)
        test_index = val_index + int(num_mol * test_proportion)
        print(f">> Data Splits: len(data_list):{len(all_data)},  [val_index, test_index]:{[val_index, test_index]}")
        val_data, test_data, train_data = all_data[:val_index], all_data[val_index:test_index], all_data[test_index:]

    elif training_mode == 'LDM':
        # load ligand data only
        ligand_data = all_data['ligand']
        
        # ligand
        ligand_mol_id = ligand_data[:, 0].astype(int)
        ligands = ligand_data[:, 1:]
        ligand_split_indices = np.nonzero(ligand_mol_id[:-1] - ligand_mol_id[1:])[0] + 1
        ligand_data_list = np.split(ligands, ligand_split_indices)

        # Keep only molecules <= filter_size
        if filter_size is not None and filter_pocket_size is not None:
            ligand_data_list = [lg for lg in ligand_data_list if lg.shape[0] <= filter_size]
            assert len(ligand_data_list) > 0, 'No molecules left after filter.'
        
        # permutation
        if permutation_file_path is not None:
            print(">> Loading permutation file from:", permutation_file_path)
            perm = np.load(permutation_file_path)
        else:
            file_name = conformation_file.split(os.path.sep)[-1][:-4]
            default_permutation_file_path = os.path.join(base_path, f'{file_name}_permutation.npy')
            # CAREFUL! Only for first time run:
            perm = np.random.permutation(len(ligand_data_list)).astype('int32')
            print('Warning, currently taking a random permutation for '
                'train/val/test partitions, this needs to be fixed for'
                'reproducibility.')
            assert not os.path.exists(default_permutation_file_path)
            np.save(default_permutation_file_path, perm)

        assert len(list(ligand_data_list)) == len(list(perm)), 'Invalid permutation file! Did you change [filter_size] and/or [filter_pocket_size]?'

        all_data = [ligand_data_list[i] for i in perm]

        # split
        num_mol = len(all_data)
        val_index = int(num_mol * val_proportion)
        test_index = val_index + int(num_mol * test_proportion)
        print(f">> Data Splits: len(data_list):{len(all_data)},  [val_index, test_index]:{[val_index, test_index]}")
        val_data, test_data, train_data = all_data[:val_index], all_data[val_index:test_index], all_data[test_index:]

    elif training_mode == 'ControlNet':
        # load and combine Ligand+Pocket data
        ligand_data, pocket_data = all_data['ligand'], all_data['pocket']
        
        # ligand
        ligand_mol_id = ligand_data[:, 0].astype(int)
        ligands = ligand_data[:, 1:]
        ligand_split_indices = np.nonzero(ligand_mol_id[:-1] - ligand_mol_id[1:])[0] + 1
        ligand_data_list = np.split(ligands, ligand_split_indices)
        
        # pocket
        pocket_mol_id = pocket_data[:, 0].astype(int)
        pockets = pocket_data[:, 1:]
        pocket_split_indices = np.nonzero(pocket_mol_id[:-1] - pocket_mol_id[1:])[0] + 1
        pocket_data_list = np.split(pockets, pocket_split_indices)
        
        # Keep only molecules <= filter_size
        if filter_size is not None and filter_pocket_size is not None:
            
            assert len(list(ligand_data_list)) == len(list(pocket_data_list))
            
            tmp_ligand_data_list, tmp_pocket_data_list = [], []
            for i in range(len(list(ligand_data_list))):
                if (ligand_data_list[i].shape[0] <= filter_size) and (pocket_data_list[i].shape[0] <= filter_pocket_size):
                    tmp_ligand_data_list.append(ligand_data_list[i])
                    tmp_pocket_data_list.append(pocket_data_list[i])
            
            ligand_data_list = tmp_ligand_data_list
            pocket_data_list = tmp_pocket_data_list
            assert len(ligand_data_list) > 0, 'No molecules left after filter.'
        
        # permutation
        if permutation_file_path is not None:
            print(">> Loading permutation file from:", permutation_file_path)
            perm = np.load(permutation_file_path)
        else:
            file_name = conformation_file.split(os.path.sep)[-1][:-4]
            default_permutation_file_path = os.path.join(base_path, f'{file_name}_permutation.npy')
            # CAREFUL! Only for first time run:
            assert len(list(ligand_data_list)) == len(list(pocket_data_list)), 'Invalid Ligand-Pocket pairs'
            perm = np.random.permutation(len(ligand_data_list)).astype('int32')
            print('Warning, currently taking a random permutation for '
                'train/val/test partitions, this needs to be fixed for'
                'reproducibility.')
            assert not os.path.exists(default_permutation_file_path)
            np.save(default_permutation_file_path, perm)
        
        assert len(list(ligand_data_list)) == len(list(pocket_data_list)), 'Invalid Ligand-Pocket pairs'
        assert len(list(ligand_data_list)) == len(list(perm)), 'Invalid permutation file! Did you change [filter_size] and/or [filter_pocket_size]?'
        
        ligand_data_list = [ligand_data_list[i] for i in perm]
        pocket_data_list = [pocket_data_list[i] for i in perm]

        # split
        num_mol = len(ligand_data_list)
        val_index = int(num_mol * val_proportion)
        test_index = val_index + int(num_mol * test_proportion)
        print(f">> Data Splits: len(data_list):{len(all_data)},  [val_index, test_index]:{[val_index, test_index]}")
        ligand_val_data, ligand_test_data, ligand_train_data = ligand_data_list[:val_index], ligand_data_list[val_index:test_index], ligand_data_list[test_index:]
        pocket_val_data, pocket_test_data, pocket_train_data = pocket_data_list[:val_index], pocket_data_list[val_index:test_index], pocket_data_list[test_index:]
        
        train_data, val_data, test_data = dict(), dict(), dict()
        train_data['ligand'], train_data['pocket'] = ligand_train_data, pocket_train_data
        test_data['ligand'],  test_data['pocket']  = ligand_test_data,  pocket_test_data
        val_data['ligand'],   val_data['pocket']   = ligand_val_data,   pocket_val_data

    return train_data, val_data, test_data


class GeomDrugsDataset(Dataset):
    def __init__(self, data_list, transform=None, training_mode=None):
        """
        Args:
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform
        self.training_mode = training_mode

        if (training_mode is None) or (training_mode == 'VAE') or (training_mode == 'LDM'):
            # Sort the data list by size
            lengths = [s.shape[0] for s in data_list]
            argsort = np.argsort(lengths)               # Sort by decreasing size
            self.data_list = [data_list[i] for i in argsort]
            # Store indices where the size changes (will be access in other methods)
            self.split_indices = np.unique(np.sort(lengths), return_index=True)[1][1:]
            
        elif training_mode == 'ControlNet':
            ligand_data_list = data_list['ligand']
            pocket_data_list = data_list['pocket']
            
            # sort according to ligand
            lengths = [s.shape[0] for s in ligand_data_list]
            argsort = np.argsort(lengths)
            self.data_list = [ligand_data_list[i] for i in argsort]
            self.data_list_pocket = [pocket_data_list[i] for i in argsort]
            
            self.split_indices = np.unique(np.sort(lengths), return_index=True)[1][1:]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if (self.training_mode is None) or (self.training_mode == 'VAE') or (self.training_mode == 'LDM'):
            sample = self.data_list[idx]
            if self.transform:
                sample = self.transform(sample)

        elif self.training_mode == 'ControlNet':
            sample_ligand, sample_pocket = self.data_list[idx], self.data_list_pocket[idx]
            if self.transform:
                sample_ligand, sample_pocket = self.transform(sample_ligand), self.transform(sample_pocket)
            sample = dict()
            sample['ligand'] = sample_ligand
            sample['pocket'] = sample_pocket

        return sample


# class CustomBatchSampler(BatchSampler):
#     """ Creates batches where all sets have the same size. """
#     def __init__(self, sampler, batch_size, drop_last, split_indices):
#         super().__init__(sampler, batch_size, drop_last)
#         self.split_indices = split_indices

#     def __iter__(self):
#         batch = []
#         for idx in self.sampler:
#             batch.append(idx)
#             if len(batch) == self.batch_size or idx + 1 in self.split_indices:
#                 yield batch
#                 batch = []
#         if len(batch) > 0 and not self.drop_last:
#             yield batch

#     def __len__(self):
#         count = 0
#         batch = 0
#         for idx in self.sampler:
#             batch += 1
#             if batch == self.batch_size or idx + 1 in self.split_indices:
#                 count += 1
#                 batch = 0
#         if batch > 0 and not self.drop_last:
#             count += 1
#         return count


def collate_fn(batch):
    # zero padding done here
    batch = {prop: qm9_collate.batch_stack([mol[prop] for mol in batch])
             for prop in batch[0].keys()}

    atom_mask = batch['atom_mask']

    # Obtain edges
    batch_size, n_nodes = atom_mask.size()
    edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)

    # mask diagonal
    diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool,
                           device=edge_mask.device).unsqueeze(0)
    edge_mask *= diag_mask

    # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)

    return batch



def collate_fn_controlnet(batch):
    # you're getting a batch of dictionaries essentially with keys 'ligand' and 'pocket'
    ligand_batch = [batch[i]['ligand'] for i in range(len(list(batch)))]
    pocket_batch = [batch[i]['pocket'] for i in range(len(list(batch)))]
    
    # zero padding done here
    ligand_batch = {prop: qm9_collate.batch_stack([mol[prop] for mol in ligand_batch])
                    for prop in ligand_batch[0].keys()}
    pocket_batch = {prop: qm9_collate.batch_stack([mol[prop] for mol in pocket_batch])
                    for prop in pocket_batch[0].keys()}

    ligand_atom_mask = ligand_batch['atom_mask']
    pocket_atom_mask = pocket_batch['atom_mask']

    # Obtain edges
    ligand_batch_size, ligand_n_nodes = ligand_atom_mask.size()
    pocket_batch_size, pocket_n_nodes = pocket_atom_mask.size()
    
    ligand_edge_mask = ligand_atom_mask.unsqueeze(1) * ligand_atom_mask.unsqueeze(2)
    pocket_edge_mask = pocket_atom_mask.unsqueeze(1) * pocket_atom_mask.unsqueeze(2)

    # mask diagonal
    ligand_diag_mask = ~torch.eye(ligand_edge_mask.size(1), dtype=torch.bool,
                           device=ligand_edge_mask.device).unsqueeze(0)
    pocket_diag_mask = ~torch.eye(pocket_edge_mask.size(1), dtype=torch.bool,
                           device=pocket_edge_mask.device).unsqueeze(0)
    ligand_edge_mask *= ligand_diag_mask
    pocket_edge_mask *= pocket_diag_mask

    # edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
    ligand_batch['edge_mask'] = ligand_edge_mask.view(ligand_batch_size * ligand_n_nodes * ligand_n_nodes, 1)
    pocket_batch['edge_mask'] = pocket_edge_mask.view(pocket_batch_size * pocket_n_nodes * pocket_n_nodes, 1)
    
    batch = dict()
    batch['ligand'] = ligand_batch
    batch['pocket'] = pocket_batch
    
    return batch



class GeomDrugsDataLoader(DataLoader):
    def __init__(self, sequential, dataset, batch_size, shuffle, drop_last=False, training_mode=None):

        if sequential:
            raise NotImplementedError()
            # # This goes over the data sequentially, advantage is that it takes
            # # less memory for smaller molecules, but disadvantage is that the
            # # model sees very specific orders of data.
            # assert not shuffle
            # sampler = SequentialSampler(dataset)
            # batch_sampler = CustomBatchSampler(sampler, batch_size, drop_last,
            #                                    dataset.split_indices)
            # super().__init__(dataset, batch_sampler=batch_sampler)

        else:
            # Dataloader goes through data randomly and pads the molecules to
            # the largest molecule size.
            if (training_mode is None) or (training_mode == 'VAE') or (training_mode == 'LDM'):
                super().__init__(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn, drop_last=drop_last)
            elif training_mode == 'ControlNet':
                super().__init__(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn_controlnet, drop_last=drop_last)



class GeomDrugsTransform(object):
    def __init__(self, dataset_info, include_charges, device, sequential):
        self.atomic_number_list = torch.Tensor(dataset_info['atomic_nb'])[None, :]
        self.device = device
        self.include_charges = include_charges
        self.sequential = sequential

    def __call__(self, data):
        n = data.shape[0]
        new_data = {}
        new_data['positions'] = torch.from_numpy(data[:, -3:])
        atom_types = torch.from_numpy(data[:, 0].astype(int)[:, None])
        one_hot = atom_types == self.atomic_number_list
        new_data['one_hot'] = one_hot
        if self.include_charges:
            # ~!to ~!mp
            new_data['charges'] = torch.zeros(n, 1, device=self.device)
        else:
            # ~!to ~!mp
            new_data['charges'] = torch.zeros(0, device=self.device)
        
        # ~!to ~!mp
        new_data['atom_mask'] = torch.ones(n, device=self.device)

        if self.sequential:
            # ~!to ~!mp
            edge_mask = torch.ones((n, n), device=self.device)
            edge_mask[~torch.eye(edge_mask.shape[0], dtype=torch.bool)] = 0
            new_data['edge_mask'] = edge_mask.flatten()
        return new_data




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--conformations", type=int, default=30,
                        help="Max number of conformations kept for each molecule.")
    parser.add_argument("--remove_h", action='store_true', help="Remove hydrogens from the dataset.")
    parser.add_argument("--data_dir", type=str, default='data/geom/')
    parser.add_argument("--data_file", type=str, default="drugs_crude.msgpack")
    args = parser.parse_args()
    extract_conformers(args)
    print("DONE.")
