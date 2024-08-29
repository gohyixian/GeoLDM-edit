import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_combined_boxplot(data_lists, colors, dataset_names, title='', xlabel='', ylabel='', img_save_path='', ymin=None, ymax=None):
    
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=data_lists, palette=colors)
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    plt.title(title)
    plt.xticks(range(len(dataset_names)), dataset_names)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(img_save_path)


def euclidean_distance(point1, point2, axis=0):
    """
    Calculate the Euclidean distance between two points.
    
    Parameters:
        point1 (ndarray): Array containing the coordinates of the first point.
        point2 (ndarray): Array containing the coordinates of the second point.
    
    Returns:
        float: Euclidean distance between the two points.
    """
    return np.sqrt(np.sum((point1 - point2) ** 2, axis=axis))


def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list


def stratified_sampling_by_atom_distribution(molecule_data_np, splits=[0.8, 0.1, 0.1]):
    # Find unique rows (strata) in molecule_data
    unique_strata, unique_indices = np.unique(molecule_data_np, axis=0, return_inverse=True)
    sampled_train_indices = []
    sampled_test_indices = []
    sampled_val_indices = []
    
    # Iterate over each unique stratum
    for stratum_idx in range(unique_strata.shape[0]):
        # Find indices of all molecules that match the current stratum
        stratum_indices = np.where(unique_indices == stratum_idx)[0].tolist()
        # Shuffle the indices and select a subset
        np.random.shuffle(stratum_indices)
        num_indices = len(stratum_indices)
        if num_indices > 3:
            # num_train = int(splits[0] * num_indices)
            num_test = int(splits[1] * num_indices)
            num_val = int(splits[2] * num_indices)
            
            # val
            if num_val > 0:
                sampled_val_indices.append(stratum_indices[-num_val:])
            
            # test
            test_start = num_val + num_test
            test_end = num_val
            if (test_start > 0) and (test_start != test_end):
                if test_end > 0:
                    sampled_test_indices.append(stratum_indices[-test_start:-test_end])
                else:
                    sampled_test_indices.append(stratum_indices[-test_start:])
            
            # train
            if test_start > 0:
                sampled_train_indices.append(stratum_indices[:-test_start])
            else:
                sampled_train_indices.append(stratum_indices)
        elif num_indices == 3:
            print(f">> indices == 3 >> {unique_strata[stratum_idx]}")
            sampled_train_indices.append(stratum_indices[0])
            sampled_test_indices.append(stratum_indices[1])
            sampled_val_indices.append(stratum_indices[2])
        elif num_indices == 2:
            print(f">> indices == 2 >> {unique_strata[stratum_idx]}")
            sampled_train_indices.append(stratum_indices[0])
            sampled_val_indices.append(stratum_indices[1])
        elif num_indices == 1:
            print(f">> indices == 1 >> {unique_strata[stratum_idx]}")
            sampled_train_indices.append(stratum_indices[0])
        else:
            raise Exception(f"num_indices==0 for {unique_strata[stratum_idx]}")
    
    return flatten_list(sampled_train_indices), \
           flatten_list(sampled_test_indices), \
           flatten_list(sampled_val_indices)
