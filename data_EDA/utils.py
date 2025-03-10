import numpy as np

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



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_histogram(data_list, color, label, title='', img_save_path='', xlabel='', ylabel='Frequency', log=False, ymin=None, ymax=None, alpha=0.7):

    maximum_num_atoms = max(data_list)
    minimum_num_atoms = min(data_list)

    bin_edges = np.arange(minimum_num_atoms, maximum_num_atoms + 1)

    plt.figure(figsize=(10, 5))
    plt.hist(data_list, bins=bin_edges, color=color, alpha=alpha, label=label, log=log)
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    plt.title(title)
    plt.xlabel(xlabel)
    log_scale_txt = " (log scale)" if log else ""
    plt.ylabel(ylabel + log_scale_txt)
    plt.savefig(img_save_path)
    plt.show()

def plot_combined_histogram(data_lists, colors, labels, title='', img_save_path='', xlabel='', ylabel='Frequency', log=False, ymin=None, ymax=None, alpha=0.3):
    
    plt.figure(figsize=(10, 5))
    maximum = max([max(d) for d in data_lists])
    minimum = min([min(d) for d in data_lists])
    bin_edges = np.arange(minimum, maximum + 1)

    for i in range(len(data_lists)):
        plt.hist(data_lists[i], bins=bin_edges, color=colors[i], alpha=alpha, label=labels[i], log=log)
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)
    plt.title(title)
    plt.xlabel(xlabel)
    log_scale_txt = " (log scale)" if log else ""
    plt.ylabel(ylabel + log_scale_txt)
    plt.legend()
    plt.savefig(img_save_path)
    plt.show()

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
    plt.show()



from prettytable import PrettyTable

def get_min_max_mean_std(data_lists, data_lists_names, round_dp=1, align='r'):
    table = PrettyTable()
    
    table.field_names = ['', 'Min', 'Mean', 'Max', 'Std']
    
    for i in range(len(data_lists)):
        values = []
        min_, mean_, max_, std_ = np.min(data_lists[i]), np.mean(data_lists[i]), np.max(data_lists[i]), np.std(data_lists[i])
        values.append(round(min_,  round_dp))
        values.append(round(mean_, round_dp))
        values.append(round(max_,  round_dp))
        values.append(round(std_,  round_dp))
        table.add_row([data_lists_names[i]] + values)
    
    if align in ['l', 'r']:
        for field in table.field_names:
            table.align[field] = align
    
    return table



# ! pip install scipy
from scipy import stats
# ! pip install prettytable
from prettytable import PrettyTable

def all_vs_all_welch_t_test(data_lists, data_lists_names, alpha=0.05, align='r'):
    t_table = PrettyTable()
    p_table = PrettyTable()
    
    t_table.field_names = [f'(T) Alpha:{alpha}'] + data_lists_names
    p_table.field_names = [f'(P) Alpha:{alpha}'] + data_lists_names
    
    for i in range(len(data_lists)):
        values_t, values_p = [], []
        for j in range(len(data_lists)):
            if i == j:
                values_t.append('-')
                values_p.append('-')
            else:
                t_stat, p_value = stats.ttest_ind(data_lists[i], data_lists[j], equal_var=False)
                sig_diff = '(Sig)' if p_value < alpha else ''
                values_t.append(str(t_stat))
                values_p.append(str(p_value) + " " + sig_diff)
        t_table.add_row([data_lists_names[i]] + values_t)
        p_table.add_row([data_lists_names[i]] + values_p)
    
    if align in ['l', 'r']:
        for field in t_table.field_names:
            t_table.align[field] = align
        for field in p_table.field_names:
            p_table.align[field] = align

    return p_table, t_table