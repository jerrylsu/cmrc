import collections
import torch


def custom_collate(batch):
    """Solving for RuntimeError: each element in list of batch should be of equal size
    """
    elem = batch[0]
    columns = ['input_ids', 'token_type_ids', 'attention_mask', 'start_positions', 'end_positions']
    if isinstance(elem, collections.Mapping):
        return {key: custom_collate([torch.tensor(d[key]) if key in columns else d[key] for d in batch]) for key in elem}
    elif isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    else:
        return batch
