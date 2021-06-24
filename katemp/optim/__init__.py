from torch.optim import Adam, SGD, AdamW

from .sam import SAM
from .ranger import Ranger


def get_optimizer(model, optimizer_name='Adam', use_sam=False, *args, **kwargs):
    if optimizer_name == 'Adam':
        optimizer =  Adam
    elif optimizer_name == 'SGD':
        optimizer = SGD
    elif optimizer_name == 'AdamW':
        optimizer = AdamW
    elif optimizer_name == 'Ranger':
        optimizer = Ranger
    else:
        raise(f'Optimizer {optimizer_name} is not implemented')
    
    if use_sam:
        return SAM(model.parameters(), optimizer, *args, **kwargs)
    else:
        return optimizer(model.parameters(), *args, **kwargs)