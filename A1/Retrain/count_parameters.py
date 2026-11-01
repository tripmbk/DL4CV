def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Args:
        model (torch.nn.Module): The PyTorch model to count parameters for.
        trainable (bool, optional): Whether to count only trainable parameters. Default is True.

    Returns:
        int: The number of parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total_params": total_params / 1e6,  # Convert to millions
        "trainable_params": trainable_params / 1e6
    }