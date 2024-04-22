import os
import pickle

from GEOM import GEOM
from QM9 import QM9
from PDBBIND import PDBBIND

geom_data_object_pkl = 'data_object_cache/GEOM_data_object.pkl'
qm9_data_object_pkl  = 'data_object_cache/QM9_data_object.pkl'

# Load the object back from the file
with open(geom_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
    geom_data_obj = pickle.load(file)

# Load the object back from the file
with open(qm9_data_object_pkl, 'rb') as file:  # Use 'rb' mode for binary reading
    qm9_data_obj = pickle.load(file)

joined_data_obj = geom_data_obj  # use GEOM as base

for k,v in qm9_data_obj.atomic_num_freq.items():
    joined_data_obj.atomic_num_freq[k] = joined_data_obj.atomic_num_freq.get(k, 0) + v

joined_data_obj.num_atoms += qm9_data_obj.num_atoms
joined_data_obj.num_atoms_no_H += qm9_data_obj.num_atoms_no_H
joined_data_obj.radius_mean += qm9_data_obj.radius_mean
joined_data_obj.radius_min += qm9_data_obj.radius_min
joined_data_obj.radius_max += qm9_data_obj.radius_max
joined_data_obj.mol_count += qm9_data_obj.mol_count

join_data_object_pkl = 'data_object_cache/JOINED_GEOM_QM9_data_object.pkl'
with open(join_data_object_pkl, 'wb') as file:  # Use 'wb' mode for binary writing
    pickle.dump(joined_data_obj, file)
