"""
Gaussian Process model creation and training utilities.

This module provides functions to:
- Create GP models with custom mean and covariance functions
- Train GP models using various strategies
- Evaluate model performance
"""

import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.fit import fit_gpytorch_mll
from botorch.models.transforms.outcome import Standardize
import torch.optim as optim
from sklearn.metrics import r2_score, root_mean_squared_error

from mean_functions import VectorizedPhysicsMean
from kernels import build_kernel


def create_gp_model(train_X, train_Y, databank, atom_ids, mean_config, kernel_config, device):
    """
    Create a GP model with custom mean and kernel functions.
    
    Args:
        train_X: Training inputs [n_train, n_features]
        train_Y: Training outputs [n_train, 1]
        databank: DataBank instance
        atom_ids: List of atom IDs to include
        mean_config: Configuration for mean function
        kernel_config: Configuration for kernel
        device: Device to use (CPU or CUDA)
        
    Returns:
        SingleTaskGP model instance
    """
    # Create mean function
    mean_module = None
    if mean_config["type"] == "VectorizedPhysicsMean":
        # When using Standardize transform, the mean function works in standardized space
        # Initialize constant to 0 (will learn the offset in standardized space)
        constant_mean = 0.0
        
        mean_module = VectorizedPhysicsMean(
            databank=databank,
            atom_ids=atom_ids,
            J_prior_mean=mean_config.get("J_prior_mean", 0.5),
            J_prior_std=mean_config.get("J_prior_std", 0.2),
            # J_lin_prior_mean=mean_config.get("J_lin_prior_mean", 0.1),
            # J_lin_prior_std=mean_config.get("J_lin_prior_std", 0.05),
            U_prior_mean=mean_config.get("U_prior_mean", 4.5),
            U_prior_std=mean_config.get("U_prior_std", 1.0),
            constant_mean=constant_mean
        )
    
    # Create kernel
    covar_module = build_kernel(databank, atom_ids, kernel_config)
    
    # Use Standardize transform to automatically normalize outputs
    # This prevents issues with large absolute energy values overwhelming the physics-based mean
    model = SingleTaskGP(
        train_X=train_X,
        train_Y=train_Y,
        mean_module=mean_module,
        covar_module=covar_module,
        outcome_transform=Standardize(m=1),  # Standardize outputs to mean=0, std=1
        # outcome_transform= None
    )
    
    return model


def train_gp_model(model, train_X, train_Y, training_config):
    """
    Train a GP model using the specified training strategy.
    
    Args:
        model: The GP model to train
        train_X: Training inputs
        train_Y: Training outputs
        training_config: Dictionary with training parameters
        
    Returns:
        Trained model
    """
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    mll = mll.to(train_X)
    
    method = training_config.get("method", "fit_gpytorch_mll")
    
    if method == "fit_gpytorch_mll":
        # Default: Use BoTorch's fit_gpytorch_mll (L-BFGS-B)
        fit_gpytorch_mll(mll)
        
    elif method == "sgd":
        # Custom SGD training loop (following BoTorch tutorial)
        # _train_with_sgd(model, mll, train_X, train_Y, training_config)
        raise NotImplementedError("SGD training method is currently disabled due to issues with Standardize transform")
        
    elif method == "two_stage":
        # Two-stage training: first mean only, then kernel only
        # _train_two_stage(model, mll, train_X, train_Y, training_config)
        raise NotImplementedError("Two-stage training method is currently disabled due to issues with Standardize transform")
        
    else:
        # Fallback to default
        fit_gpytorch_mll(mll)
    
    return model


def _train_with_sgd(model, mll, train_X, train_Y, training_config):
    """
    Train GP model using SGD optimizer (following BoTorch tutorial).
    
    This allows customization of the training loop and works well when you need
    fine control over the optimization process or want to enforce constraints
    on physics parameters during training.
    
    Note: When using outcome_transform=Standardize, the model internally handles
    the transformation, so we train on the original train_Y values.
    
    Args:
        model: The GP model
        mll: Marginal log likelihood
        train_X: Training inputs
        train_Y: Training outputs (will be squeezed to 1D, in original scale)
        training_config: Dictionary with training parameters
    """
    from torch.optim import SGD, Adam
    
    sgd_config = training_config.get("sgd", {})
    num_epochs = sgd_config.get("epochs", 150)
    lr = sgd_config.get("lr", 0.025)
    optimizer_type = sgd_config.get("optimizer", "sgd")  # "sgd" or "adam"
    print_every = sgd_config.get("print_every", 10)
    freeze_kernel = sgd_config.get("freeze_kernel", False)  # Only train mean parameters
    
    # Freeze kernel parameters if requested, but KEEP NOISE TRAINABLE
    # The noise (likelihood) must be trainable for the mean to fit properly
    if freeze_kernel:
        print("Freezing kernel parameters - training mean function and noise")
        for name, param in model.named_parameters():
            if 'mean_module' not in name and 'likelihood' not in name:
                param.requires_grad = False
    
    # Collect trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {len(trainable_params)}")
    
    # Choose optimizer
    if optimizer_type == "adam":
        optimizer = Adam(trainable_params, lr=lr)
        print(f"Training with Adam optimizer (lr={lr}, epochs={num_epochs})")
    else:
        optimizer = SGD(trainable_params, lr=lr)
        print(f"Training with SGD optimizer (lr={lr}, epochs={num_epochs})")
    
    # Set model to training mode
    model.train()
    
    # Debug: Check if we have outcome transform and inspect standardized targets
    has_transform = hasattr(model, 'outcome_transform') and model.outcome_transform is not None
    if has_transform:
        print(f"Using outcome transform: {type(model.outcome_transform).__name__}")
        # Show what the standardized targets look like
        with torch.no_grad():
            standardized_Y, _ = model.outcome_transform(train_Y)
            print(f"Original Y: mean={train_Y.mean():.2f}, std={train_Y.std():.2f}")
            print(f"Standardized Y: mean={standardized_Y.mean():.4f}, std={standardized_Y.std():.4f}")
    
    for epoch in range(num_epochs):
        
        # Clear gradients
        optimizer.zero_grad()
        
        # Forward pass through the model to obtain the output MultivariateNormal
        output = model(train_X)
        
        # Compute negative marginal log likelihood
        # The model's outcome_transform automatically handles standardization if present
        loss = -mll(output, train_Y.squeeze(-1))
        
        # Back prop gradients
        loss.backward()
        
        # Apply optimizer step
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % print_every == 0:
            loss_val = loss.item()
            noise_val = model.likelihood.noise.item()
            
            print(f"Epoch {epoch+1:>3}/{num_epochs} - Loss: {loss_val:>4.3f} - Noise: {noise_val:>4.3f}", end="")
            
            # Print physics parameters if available
            if hasattr(model.mean_module, 'J'):
                print(f" - J: {model.mean_module.J.item():.4f} "
                    #   f"J_lin: {model.mean_module.J_lin.item():.4f} "
                      f"U: {model.mean_module.U.item():.4f} "
                      f"const: {model.mean_module.constant.item():.2f}")
                
                # Diagnostic: check mean predictions in standardized space
                if (epoch + 1) == print_every:  # Only print once at start
                    with torch.no_grad():
                        mean_pred = model.mean_module(train_X)
                        if has_transform:
                            standardized_Y, _ = model.outcome_transform(train_Y)
                            print(f"    DEBUG: Mean pred range: [{mean_pred.min():.2f}, {mean_pred.max():.2f}], "
                                  f"Standardized Y range: [{standardized_Y.min():.2f}, {standardized_Y.max():.2f}]")
            else:
                print()
    
    print("Training complete")


def _train_two_stage(model, mll, train_X, train_Y, training_config):
    """
    Two-stage training strategy:
    1. Train mean function aggressively with kernel frozen
    2. Freeze mean and train kernel/likelihood to learn residuals
    
    This forces the mean to capture the physics-based structure,
    then the kernel learns corrections/correlations.
    
    Note: When using outcome_transform=Standardize, the model internally handles
    the transformation, so we train on the original train_Y values.
    
    Args:
        model: The GP model
        mll: Marginal log likelihood
        train_X: Training inputs
        train_Y: Training outputs (will be squeezed to 1D, in original scale)
        training_config: Dictionary with training parameters
    """
    from torch.optim import Adam
    
    two_stage_config = training_config.get("two_stage", {})
    
    # Stage 1 config: Train mean only
    stage1_epochs = two_stage_config.get("stage1_epochs", 500)
    stage1_lr = two_stage_config.get("stage1_lr", 0.01)
    stage1_print_every = two_stage_config.get("stage1_print_every", 50)
    
    # Stage 2 config: Train kernel only
    stage2_epochs = two_stage_config.get("stage2_epochs", 200)
    stage2_lr = two_stage_config.get("stage2_lr", 0.01)
    stage2_print_every = two_stage_config.get("stage2_print_every", 20)
    
    print("\n" + "="*60)
    print("STAGE 1: Training mean function (kernel frozen)")
    print("="*60)
    
    # Freeze kernel and likelihood parameters
    for name, param in model.named_parameters():
        if 'mean_module' not in name:
            param.requires_grad = False
    
    # Collect trainable parameters (mean only)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)} (mean function only)")
    
    optimizer_stage1 = Adam(trainable_params, lr=stage1_lr)
    model.train()
    
    # Debug: Check if we have outcome transform
    has_transform = hasattr(model, 'outcome_transform') and model.outcome_transform is not None
    if has_transform:
        print(f"Using outcome transform: {type(model.outcome_transform).__name__}")
    
    for epoch in range(stage1_epochs):
        # Enforce constraints BEFORE forward pass
        # Tightened for standardized space (baseline: J~0.25, U~2.7)
        # with torch.no_grad():
        #     if hasattr(model.mean_module, 'J'):
        #         model.mean_module.J.clamp_(0.0, 1.0)
        #         model.mean_module.U.clamp_(0.0, 6.0)
        
        optimizer_stage1.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y.squeeze(-1))
        loss.backward()
        optimizer_stage1.step()
        
        if (epoch + 1) % stage1_print_every == 0:
            print(f"Epoch {epoch+1:>3}/{stage1_epochs} - Loss: {loss.item():>4.3f}", end="")
            if hasattr(model.mean_module, 'J'):
                print(f" - J: {model.mean_module.J.item():.4f} "
                      f"U: {model.mean_module.U.item():.4f} "
                      f"const: {model.mean_module.constant.item():.2f}")
            else:
                print()
    
    print("\n" + "="*60)
    print("STAGE 2: Training kernel/likelihood (mean frozen)")
    print("="*60)
    
    # Freeze mean, unfreeze kernel and likelihood
    for name, param in model.named_parameters():
        if 'mean_module' in name:
            param.requires_grad = False
        else:
            param.requires_grad = True
    
    # Collect trainable parameters (kernel + likelihood)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {len(trainable_params)} (kernel + likelihood)")
    
    optimizer_stage2 = Adam(trainable_params, lr=stage2_lr)
    
    for epoch in range(stage2_epochs):
        # Enforce kernel parameter constraints BEFORE forward pass
        with torch.no_grad():
            if hasattr(model.likelihood, 'noise_covar'):
                if hasattr(model.likelihood.noise_covar, 'raw_noise'):
                    model.likelihood.noise_covar.raw_noise.clamp_(min=-5.0)
            
            for name, param in model.named_parameters():
                if 'raw_lengthscale' in name or 'raw_outputscale' in name:
                    param.clamp_(min=-10.0, max=10.0)
        
        optimizer_stage2.zero_grad()
        output = model(train_X)
        loss = -mll(output, train_Y.squeeze(-1))
        loss.backward()
        optimizer_stage2.step()
        
        if (epoch + 1) % stage2_print_every == 0:
            noise_val = model.likelihood.noise.item()
            print(f"Epoch {epoch+1:>3}/{stage2_epochs} - Loss: {loss.item():>4.3f} - Noise: {noise_val:>4.3f}")
    
    # Unfreeze all parameters for subsequent use
    for param in model.parameters():
        param.requires_grad = True
    
    print("="*60)
    print("Two-stage training complete")
    print("="*60 + "\n")


def evaluate_model(model, test_X, test_Y):
    """
    Evaluate model performance on test data.
    
    Args:
        model: Trained GP model
        test_X: Test inputs
        test_Y: Test outputs
        
    Returns:

        Dictionary with evaluation metrics
    """
    model.eval()
    with torch.no_grad():
        posterior = model.posterior(test_X)
        y_pred_mean = posterior.mean
        y_pred_variance = posterior.variance

    # Move to numpy for scikit-learn metrics
    y_true_np = test_Y.cpu().numpy().ravel()
    y_pred_np = y_pred_mean.cpu().numpy().ravel()
    y_std_np = torch.sqrt(y_pred_variance).cpu().numpy().ravel()

    # Calculate metrics
    r2 = r2_score(y_true_np, y_pred_np)
    rmse = root_mean_squared_error(y_true_np, y_pred_np)

    return {
        "r2": r2,
        "rmse": rmse,
        "y_true": y_true_np,
        "y_pred": y_pred_np,
        "y_std": y_std_np,
    }


def print_kernel_diagnostics(model):
    """
    Print diagnostic information about kernel components and their learned variances.
    
    Args:
        model: Trained GP model
    """
    import gpytorch
    
    print("\n" + "="*60)
    print(f"{'KERNEL TYPE':<30} | {'VARIANCE (Outputscale)':<25}")
    print("="*60)
    
    # Get the list of additive components
    if hasattr(model.covar_module, 'kernels'):
        sub_kernels = model.covar_module.kernels
    else:
        sub_kernels = [model.covar_module]

    for i, sub_kernel in enumerate(sub_kernels):
        name = f"Term {i}"
        variance = "N/A"
        
        # Unwrap ScaleKernel
        if isinstance(sub_kernel, gpytorch.kernels.ScaleKernel):
            variance = f"{sub_kernel.outputscale.item():.5f}"
            base = sub_kernel.base_kernel
        else:
            base = sub_kernel
            
        # Unwrap SpinFlipInvariantKernel
        if hasattr(base, 'base_kernel'):
            if "SpinFlip" in base.__class__.__name__:
                base = base.base_kernel

        # Identify the physics type
        if isinstance(base, gpytorch.kernels.ProductKernel):
            k1 = base.kernels[0]
            k2 = base.kernels[1]
            
            if isinstance(k1, gpytorch.kernels.LinearKernel):
                name = "NON-LOCAL: Heisenberg (Lin x Lin)"
            elif isinstance(k1, gpytorch.kernels.PolynomialKernel):
                name = "NON-LOCAL: Kugel-Khomskii (Poly x Poly)"
            elif isinstance(k1, gpytorch.kernels.MaternKernel):
                name = "NON-LOCAL: Residuals (Mat x Mat)"
            else:
                name = "NON-LOCAL: Mixed Product"

        elif isinstance(base, gpytorch.kernels.LinearKernel):
            name = "LOCAL: Linear"
        elif isinstance(base, gpytorch.kernels.PolynomialKernel):
            name = "LOCAL: Poly"
        elif isinstance(base, gpytorch.kernels.MaternKernel):
            name = "LOCAL: Texture (Matern)"
            
        print(f"{name:<30} | {variance:<25}")
    
    print("="*60 + "\n")
