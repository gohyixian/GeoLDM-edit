
import os
import argparse
import numpy as np
from tqdm import tqdm

# path to .npy file containing the combined/processed conformations/molecules,
# all stored in a single array of the format below. Atoms of the same molecule
# will have the same idx number to group them together.
#
# [[idx, atomic_num, x, y, z],
#  [idx, atomic_num, x, y, z],
#  [idx, atomic_num, x, y, z],
#   ...
#  [idx, atomic_num, x, y, z]]


def main(args):
    all_data = np.load(args.data)

    all_ligand_data = all_data['ligand']  # TODO: fix script for ligand, pocket, ligand+pocket
    all_pocket_data = all_data['pocket']

    # ligand
    ligand_mol_id = np.copy(all_ligand_data[:, 0]).astype(int)
    ligand_split_indices = np.nonzero(ligand_mol_id[:-1] - ligand_mol_id[1:])[0] + 1
    ligand_data_list = np.split(all_ligand_data, ligand_split_indices)

    # pocket
    pocket_mol_id = np.copy(all_pocket_data[:, 0]).astype(int)
    pocket_split_indices = np.nonzero(pocket_mol_id[:-1] - pocket_mol_id[1:])[0] + 1
    pocket_data_list = np.split(all_pocket_data, pocket_split_indices)


    assert len(list(ligand_data_list)) == len(list(pocket_data_list)), f"len(ligand_data_list)={len(list(ligand_data_list))}, len(pocket_data_list)={len(list(pocket_data_list))}"

    if args.randomize:
        perm = np.random.permutation(len(list(ligand_data_list)))
        ligand_data_list = [ligand_data_list[i] for i in perm]
        pocket_data_list = [pocket_data_list[i] for i in perm]

    ligand_data_list_subset = []
    pocket_data_list_subset = []
    print(f"Subsetting {args.portion*100}% of data ..")
    for i in tqdm(range(int(args.portion * len(ligand_data_list)))):
        ligand_data_list_subset.append(ligand_data_list[i])
        pocket_data_list_subset.append(pocket_data_list[i])

    print(f"len(original) : {len(list(ligand_data_list))}")
    print(f"len(subset)   : {len(list(ligand_data_list_subset))}")
    ligand_data_list_subset = np.vstack(ligand_data_list_subset)
    pocket_data_list_subset = np.vstack(pocket_data_list_subset)

    dir = os.path.dirname(args.saveas)
    if not os.path.exists(dir):
        print(f"Creating {dir}")
        os.makedirs(dir)

    print(f"Saving subset data file as {args.saveas}")
    np.savez(args.saveas, ligand=ligand_data_list_subset, pocket=pocket_data_list_subset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A.npz')
    parser.add_argument('--saveas', type=str, default='../data/d_20240623_CrossDocked_LG_PKT/d_20240623_CrossDocked_LG_PKT__10.0A__subset_0.01.npz')
    parser.add_argument('--portion', type=float, default=0.01)
    parser.add_argument('--randomize', type=bool, default=True)
    args = parser.parse_args()
    
    main(args)
    print('DONE.')