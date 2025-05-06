# utils/optimizers.py
import logging
import torch.optim as optim

def get_optimizer(model_parameters, name, lr, weight_decay=0.0, momentum=0.9):
    """
    Get an optimizer with specified parameters.
    
    Args:
        model_parameters: Model parameters to optimize
        name (str): Optimizer name ('adam' or 'sgd')
        lr (float): Learning rate
        weight_decay (float, optional): Weight decay (L2 penalty). Defaults to 0.0
        momentum (float, optional): Momentum factor for SGD. Defaults to 0.9
    
    Returns:
        torch.optim.Optimizer: The optimizer instance
    """
    name = name.lower()
    
    if name == "adam":
        logging.info(f"Using Adam optimizer with weight_decay={weight_decay} and learning_rate={lr}")
        return optim.Adam(
            model_parameters,
            lr=lr,
            weight_decay=weight_decay
        )
    elif name == "sgd":
        logging.info(f"Using SGD optimizer with momentum={momentum} and weight_decay={weight_decay} and learning_rate={lr}")
        return optim.SGD(
            model_parameters,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {name}. Supported optimizers: ['adam', 'sgd']")