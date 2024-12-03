import pickle
from configs.constants_colors_radius import get_radius, get_colors
from configs.dataset_config_QM9 import *
from configs.dataset_config_GEOM import *
from configs.dataset_config_GEOM_PDBB_combined import *
from configs.dataset_config_GEOM_CrossDocked import *
from configs.dataset_config_CrossDocked import *
from configs.dataset_config_BindingMOAD import *



def get_dataset_info(dataset_name, remove_h=False):
    if dataset_name == 'qm9':
        if not remove_h:
            return qm9_with_h
        else:
            return qm9_without_h
    elif dataset_name == 'geom':
        if not remove_h:
            return geom_with_h
        else:
            raise Exception('Missing config for %s without hydrogens' % dataset_name)
    elif dataset_name == 'qm9_second_half':
        if not remove_h:
            return qm9_second_half
        else:
            raise Exception('Missing config for %s without hydrogens' % dataset_name)
    elif dataset_name == 'd_20240428_combined_geom_PDBB_full_refined_core_LG_PKT':
        return d_20240428_combined_geom_PDBB_full_refined_core_LG_PKT

    # BindingMOAD
    elif dataset_name == 'd_20240623_BindingMOAD_LG_PKT__10.0A__MaxOcc50__CA_Only__no_H__LIGAND+POCKET' and remove_h == True:
        return d_20240623_BindingMOAD__10A__MaxOcc50__CA_Only__no_H__LIGAND_POCKET
    elif dataset_name == 'd_20240623_BindingMOAD_LG_PKT__10.0A__MaxOcc50__CA_Only__no_H__LIGAND' and remove_h == True:
        return d_20240623_BindingMOAD__10A__MaxOcc50__CA_Only__no_H__LIGAND
    elif dataset_name == 'd_20240623_BindingMOAD_LG_PKT__10.0A__MaxOcc50__LIGAND+POCKET' and remove_h == False:
        return d_20240623_BindingMOAD__10A__MaxOcc50__LIGAND_POCKET
    elif dataset_name == 'd_20240623_BindingMOAD_LG_PKT__10.0A__MaxOcc50__LIGAND' and remove_h == False:
        return d_20240623_BindingMOAD__10A__MaxOcc50__LIGAND

    # CrossDocked
    elif dataset_name == 'd_20240623_CrossDocked_LG_PKT__10A__CA_Only__no_H__LIGAND+POCKET' and remove_h == True:
        return d_20240623_CrossDocked_LG_PKT__10A__CA_Only__no_H__LIGAND_POCKET
    elif dataset_name == 'd_20240623_CrossDocked_LG_PKT__10A__CA_Only__no_H__LIGAND' and remove_h == True:
        return d_20240623_CrossDocked_LG_PKT__10A__CA_Only__no_H__LIGAND
    elif dataset_name == 'd_20240623_CrossDocked_LG_PKT__10A__LIGAND+POCKET' and remove_h == False:
        return d_20240623_CrossDocked_LG_PKT__10A__LIGAND_POCKET
    elif dataset_name == 'd_20240623_CrossDocked_LG_PKT__10A__LIGAND' and remove_h == False:
        return d_20240623_CrossDocked_LG_PKT__10A__LIGAND
    
    elif dataset_name == 'd_20241115_GEOM_LDM_CrossDocked_LG_PKT_MMseq2_split__10A__LIGAND' and remove_h == False:
        return d_20241115_GEOM_LDM_CrossDocked_LG_PKT_MMseq2_split__10A__LIGAND
    
    elif dataset_name == 'd_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only__10A__POCKET' and remove_h == False:
        return d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only__10A__POCKET
    elif dataset_name == 'd_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only__10A__LIGAND' and remove_h == False:
        return d_20241203_CrossDocked_LG_PKT_MMseq2_split_CA_only__10A__LIGAND
    
    else:
        raise Exception(f"Wrong dataset {dataset_name} with remove_h={remove_h}")
