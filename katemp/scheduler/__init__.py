from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR, CosineAnnealingWarmRestarts

from .scheduler import GradualWarmupSchedulerV2

def get_scheduler(
    optimizer,
    scheduler,
    mode='min',
    factor=0.2,
    patience=4,
    eps=1e-6,
    T_max=6,
    T_0=6,
    multiplier=10.0,
    eta_min=1e-6,
    warmup_epochs=1,
    cosine_epochs=5,
    *args, **kwargs ):
    """
    Args:
        optimizer (optim.Optimizer): optimzer 
        scheduler (str): name of scheduler, 
            [ReduceOnPlateau, CosineAnnealingLR, 
             CosineAnnealingWarmRestarts, GradualWarmupSchedulerV2] 
            are implemented.
        mode (str): only for ReduceOnPlateau
        factor (float): only for ReduceOnPlateau
        patience  (float): only for ReduceOnPlateau
        eps  (float): only for ReduceOnPlateau
        T_max  (int): for CosineAnnealingLR, GradualWarmupSchedulerV2
        T_0  (int): for CosineAnnealingLR, CosineAnnealingWarmRestarts, GradualWarmupSchedulerV2
        multiplier  (float): only for GradualWarmupSchedulerV2, use lr_max / lr_start
        eta_min  (float): for CosineAnnealingLR, CosineAnnealingWarmRestarts, GradualWarmupSchedulerV2
        warmup_epochs  (int): only for GradualWarmupSchedulerV2, number of epochs to warmup
        cosine_epochs  (int): only for GradualWarmupSchedulerV2, number of epochs used for CosineAnnealingLR. Usually epochs - warmup_epochs

        
    """

    if scheduler=='ReduceLROnPlateau':
        scheduler_ = ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, verbose=True, eps=eps, *args, **kwargs)
    elif scheduler=='CosineAnnealingLR':
        scheduler_ = CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, last_epoch=-1, *args, **kwargs)
    elif scheduler=='CosineAnnealingWarmRestarts':
        scheduler_ = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=eta_min, last_epoch=-1, *args, **kwargs)
    elif scheduler == 'GradualWarmupSchedulerV2':
        scheduler_cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs - warmup_epochs, eta_min=eta_min, last_epoch=-1, *args, **kwargs)
        scheduler_ = GradualWarmupSchedulerV2(optimizer, multiplier=multiplier, total_epoch=warmup_epochs, after_scheduler=scheduler_cosine)
    else:
        raise(f'Scheduler {scheduler} is not implementated')
    return scheduler_