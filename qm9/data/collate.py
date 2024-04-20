import torch


def batch_stack(props):
    """
    Stack a list of torch.tensors so they are padded to the size of the
    largest tensor along each axis.

    Parameters
    ----------
    props : list of Pytorch Tensors
        Pytorch tensors to stack

    Returns
    -------
    props : Pytorch tensor
        Stacked pytorch tensor.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return torch.tensor(props)
    elif props[0].dim() == 0:
        return torch.stack(props)
    else:
        return torch.nn.utils.rnn.pad_sequence(props, batch_first=True, padding_value=0)


def drop_zeros(props, to_keep):
    """
    Function to drop zeros from batches when the entire dataset is padded to the largest molecule size.

    Parameters
    ----------
    props : Pytorch tensor
        Full Dataset


    Returns
    -------
    props : Pytorch tensor
        The dataset with  only the retained information.

    Notes
    -----
    TODO : Review whether the behavior when elements are not tensors is safe.
    """
    if not torch.is_tensor(props[0]):
        return props
    elif props[0].dim() == 0:
        return props
    else:
        return props[:, to_keep, ...]


class PreprocessQM9:
    def __init__(self, load_charges=True):
        self.load_charges = load_charges

    def add_trick(self, trick):
        self.tricks.append(trick)

    def collate_fn(self, batch):
        """
        Collation function that collates datapoints into the batch format for cormorant

        Parameters
        ----------
        batch : list of datapoints
            The data to be collated.

        Returns
        -------
        batch : dict of Pytorch tensors
            The collated data.
        """
        # list of [bs * [max_num_nodes=29, 3]]
        batch = {prop: batch_stack([mol[prop] for mol in batch]) for prop in batch[0].keys()}
        # [bs, max_num_nodes=29, 3]

        # batch['charges']: [64, 29] --sum--> [29]
        to_keep = (batch['charges'].sum(0) > 0)   # [29]

        # excess ZEROs dropped here, keep only size for max molecule size in this batch
        batch = {key: drop_zeros(prop, to_keep) for key, prop in batch.items()}  # [bs, <=29, 3]

        atom_mask = batch['charges'] > 0
        batch['atom_mask'] = atom_mask
        # atom mask shape: [64, about-23-29] cuz dropped 0

        #Obtain edges
        batch_size, n_nodes = atom_mask.size()    # assume n_nodes=29
        edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)   # [64, 1, 29] * [64, 29, 1] = [64, 29, 29]

        #mask diagonal (remove atom self-to-self connections)
        inv_diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)  # [1, 29, 29] diagonal
        edge_mask *= inv_diag_mask      # remove diagonals / self connections

        #edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
        batch['edge_mask'] = edge_mask.view(batch_size * n_nodes * n_nodes, 1)     # [64x29x29, 1]

        if self.load_charges:    # include_charges=True
            batch['charges'] = batch['charges'].unsqueeze(2)   # [64, 29, 1]
        else:
            batch['charges'] = torch.zeros(0)
        return batch
