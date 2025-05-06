import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossEntropyLoss(nn.Module):
    """
    Standard Cross Entropy Loss implementation.
    This combines softmax activation with negative log likelihood loss.
    
    Args:
        reduction (str): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. Default: 'mean'
        ignore_index (int): Specifies a target value that is ignored and does
            not contribute to the input gradient. Default: -100
    """
    def __init__(self, reduction='mean', ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.ignore_index = ignore_index
        
    def forward(self, logits, targets):
        """
        Forward pass of the cross entropy loss.
        
        Args:       w
            logits (torch.Tensor): Predicted logits of shape (N, C) where C is the number of classes
            targets (torch.Tensor): Ground truth class indices of shape (N,) where values are 0 <= targets[i] <= C-1
            
        Returns:
            torch.Tensor: The computed loss value
        """
        return F.cross_entropy(
            logits,
            targets,
            reduction=self.reduction,
            ignore_index=self.ignore_index
        )
