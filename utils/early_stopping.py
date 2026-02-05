import posteriors
import numpy as np

class EarlyStopping:
    """Early stops training if validation metric doesn't improve after a given patience.
    
    Supports both minimizing (loss) and maximizing (AUROC) metrics.
    
    Ref. https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py
    """
    def __init__(self, patience=3, verbose=True, delta=0.005, mode='max', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation metric improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation metric improvement. 
                            Default: True
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0.005
            mode (str): 'min' for loss (lower is better), 'max' for AUROC (higher is better)
                        Default: 'max'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.mode = mode
        self.trace_func = trace_func
        
        # Track best value for reporting
        if mode == 'max':
            self.best_value = -np.Inf
        else:
            self.best_value = np.Inf
        
    def __call__(self, val_metric):
        """
        Check if training should stop.
        
        Args:
            val_metric: The validation metric to monitor (AUROC or loss)
            
        Returns:
            bool: True if training should stop, False otherwise
        """
        if self.mode == 'max':
            score = val_metric
            improved = score > self.best_score + self.delta if self.best_score is not None else True
        else:  # mode == 'min'
            score = -val_metric
            improved = score > self.best_score + self.delta if self.best_score is not None else True
        
        if self.best_score is None:
            self.best_score = score
            self.best_value = val_metric
            if self.verbose:
                self.trace_func(f'EarlyStopping: Initial {self._metric_name()} = {val_metric:.4f}')
        elif improved:
            if self.verbose:
                self.trace_func(f'EarlyStopping: {self._metric_name()} improved ({self.best_value:.4f} -> {val_metric:.4f})')
            self.best_score = score
            self.best_value = val_metric
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping: {self.counter}/{self.patience} (best {self._metric_name()} = {self.best_value:.4f})')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    self.trace_func(f'EarlyStopping: Stopping early! Best {self._metric_name()} = {self.best_value:.4f}')
        
        return self.early_stop
    
    def _metric_name(self):
        return 'AUROC' if self.mode == 'max' else 'loss'
    
    def reset(self):
        """Reset the early stopping state."""
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        if self.mode == 'max':
            self.best_value = -np.Inf
        else:
            self.best_value = np.Inf