import os, sys
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
import matplotlib.pyplot as plt
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    """
    Encoder network for VAE.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of the input data (default is 30).
    depth : int, optional
        Number of hidden layers in the encoder (default is 3).
    hidden_dim : int, optional
        Number of neurons in each hidden layer (default is 64).
    dropout_rate : float, optional
        Dropout rate for regularization (default is 0.3).
    latent_dim : int, optional
        Dimension of the latent space (default is 10).

    Methods
    -------
    forward(x)
        Forward pass to compute latent mean and log variance.
    """
    def __init__(self, input_dim=30, depth=3, hidden_dim=64, dropout_rate=0.3, latent_dim=10, use_batch_norm=True):
        super(Encoder, self).__init__()

        # Hidden layers
        hidden = []
        for d in range(depth):
            hidden.extend(
                [
                    nn.Dropout(dropout_rate),  # Dropout to reduce overfitting
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                    nn.LeakyReLU(),
                ]
            )

        # Encoder structure
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            *hidden,
        )

        # Latent space: mean and log variance
        self.latent_mu = nn.Linear(hidden_dim, latent_dim)
        self.latent_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.body(x)
        mu = self.latent_mu(h)
        logvar = self.latent_logvar(h)
        return mu, logvar
    

class Decoder(nn.Module):
    """
    Decoder network for VAE.

    Parameters
    ----------
    output_dim : int, optional
        Dimension of the reconstructed output (default is 30).
    depth : int, optional
        Number of hidden layers in the decoder (default is 3).
    hidden_dim : int, optional
        Number of neurons in each hidden layer (default is 64).
    dropout_rate : float, optional
        Dropout rate for regularization (default is 0.3).
    latent_dim : int, optional
        Dimension of the latent space (default is 10).

    Methods
    -------
    forward(z)
        Forward pass to reconstruct the input from the latent representation.
    """
    def __init__(self, output_dim=30, depth=3, hidden_dim=64, dropout_rate=0.3, latent_dim=10, use_batch_norm=True):
        super(Decoder, self).__init__()

        # Hidden layers
        hidden = []
        for d in range(depth):
            hidden.extend(
                [
                    nn.Dropout(dropout_rate),  # Dropout to reduce overfitting
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                    nn.LeakyReLU(),
                ]
            )

        # Decoder structure
        self.body = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            *hidden,
        )

        # Output layer
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.body(z)  # where z is the latent feature
        recon = self.output_layer(h)
        return recon
    

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) with Encoder and Decoder.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of the input data (default is 30).
    depth : int, optional
        Number of hidden layers in encoder/decoder (default is 3).
    hidden_dim : int, optional
        Number of neurons in each hidden layer (default is 64).
    dropout_rate : float, optional
        Dropout rate for regularization (default is 0.3).
    latent_dim : int, optional
        Dimension of the latent space (default is 10).

    Methods
    -------
    forward(x)
        Forward pass to encode, sample, and decode the input.
    reparameterize(mu, logvar)
        Reparameterization trick for sampling latent space.
    """
    def __init__(self, input_dim=30, depth=3, hidden_dim=64, dropout_rate=0.3, latent_dim=10):
        super(VAE, self).__init__()
        self.encoder = Encoder(input_dim, depth, hidden_dim, dropout_rate, latent_dim)
        self.decoder = Decoder(input_dim, depth, hidden_dim, dropout_rate, latent_dim)
    
    @staticmethod
    def reparameterize(mu, logvar):
        """
        Reparameterization trick to sample latent representation.

        Parameters
        ----------
        mu : torch.Tensor
            Latent mean tensor.
        logvar : torch.Tensor
            Latent log variance tensor.

        Returns
        -------
        torch.Tensor
            Sampled latent tensor with shape (batch_size, latent_dim).
        """
        # Reparameterization trick: z = mu + std * eps
        # where std = exp(0.5 * logvar) and eps ~ N(0, I)
        std = torch.exp(0.5 * logvar)  # Compute standard deviation from log variance
        eps = torch.randn_like(std)   # Sample standard Gaussian noise
        return mu + eps * std         # Reparameterize to compute latent vector z

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        return recon, mu, logvar, z

class MultiTaskPredictor(nn.Module):
    """
    Multi-task prediction network.

    Parameters
    ----------
    latent_dim : int, optional
        Dimension of the latent space input (default is 10).
    depth : int, optional
        Number of shared hidden layers (default is 3).
    hidden_dim : int, optional
        Number of neurons in each shared hidden layer (default is 64).
    dropout_rate : float, optional
        Dropout rate for regularization (default is 0.3).
    task_count : int, optional
        Number of parallel prediction tasks (default is 2).

    Methods
    -------
    forward(z)
        Forward pass through shared layers and task-specific heads.
    """
    def __init__(self, latent_dim=10, depth=3, hidden_dim=64, dropout_rate=0.3, task_count=2, use_batch_norm=True):
        super(MultiTaskPredictor, self).__init__()

        # Shared layers
        hidden = []
        for d in range(depth):
            hidden.extend(
                [
                    nn.Dropout(dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
                    nn.LeakyReLU(),
                ]
            )

        # Shared body
        self.body = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.LeakyReLU(),
            *hidden,
        )

        # Task-specific output layers
        self.task_heads = nn.ModuleList([
            nn.Linear(hidden_dim, 1) for _ in range(task_count)
        ])

    def forward(self, z):
        h = self.body(z)  # Shared feature extraction
        outputs = [torch.sigmoid(head(h)) for head in self.task_heads]
        return outputs


class HybridVAEMultiTaskModel(nn.Module):
    """
    Complete model combining VAE and Multi-Task Predictor.

    Parameters
    ----------
    input_dim : int, optional
        Dimension of the input data (default is 30).
    vae_hidden_dim : int, optional
        Number of neurons in each hidden layer of the VAE (default is 64).
    vae_depth : int, optional
        Number of hidden layers in the VAE (default is 3).
    vae_dropout_rate : float, optional
        Dropout rate for VAE hidden layers to prevent overfitting (default is 0.3).
    latent_dim : int, optional
        Dimension of the latent space for the VAE (default is 10).
    predictor_hidden_dim : int, optional
        Number of neurons in each hidden layer of the Multi-Task Predictor (default is 64).
    predictor_depth : int, optional
        Number of shared hidden layers in the Multi-Task Predictor (default is 3).
    predictor_dropout_rate : float, optional
        Dropout rate for Multi-Task Predictor hidden layers to prevent overfitting (default is 0.3).
    task_count : int, optional
        Number of parallel prediction tasks (default is 2).
    vae_lr : float, optional
        Learning rate for the VAE optimizer (default is 1e-3).
    vae_weight_decay : float, optional
        Weight decay (L2 regularization) for the VAE optimizer (default is 1e-3).
    multitask_lr : float, optional
        Learning rate for the MultiTask predictor optimizer (default is 1e-3).
    multitask_weight_decay : float, optional
        Weight decay (L2 regularization) for the MultiTask predictor optimizer (default is 1e-3).
    alphas : list or torch.Tensor, optional
        Per-task weights for the task loss term, shape (num_tasks,). Default is uniform weights (1 for all tasks).
    beta : float, optional
        Weight of the KL divergence term in the VAE loss (default is 1.0).
    gamma_task : float, optional
        Weight of the task loss term in the total loss (default is 1.0).
    batch_size : int, optional
        Batch size for training (default is 32).
    validation_split : float, optional
        Fraction of the data to use for validation (default is 0.2).
    use_lr_scheduler : bool, optional
        Whether to enable ReduceLROnPlateau learning rate scheduler for both VAE and Multi-Task predictors
        based on validation losses (default is True). If False, no learning rate adjustments will be made.
    lr_scheduler_factor : float, optional
        Factor by which the learning rate will be reduced when the scheduler is triggered (default is 0.1).
    lr_scheduler_patience : int, optional
        Number of validation epochs to wait for improvement before reducing the learning rate (default is 50).
    use_batch_norm : bool, optional
        Use batch normalization (default is True).
    Methods
    -------
    forward(x)
        Forward pass through VAE and Multi-Task Predictor.

    Attributes
    ----------
    vae : VAE
        Variational Autoencoder module consisting of an encoder and decoder.
    predictor : MultiTaskPredictor
        Multi-task prediction module that takes the latent representation as input
        and outputs predictions for multiple tasks.
    """
    def __init__(self, 
                 input_dim=30, 
                 vae_hidden_dim=64, 
                 vae_depth=1,
                 vae_dropout_rate=0.3, 
                 latent_dim=10, 
                 predictor_hidden_dim=64, 
                 predictor_depth=1,
                 predictor_dropout_rate=0.3, 
                 task_count=2,
                 # training related params for param tuning
                 vae_lr=1e-3, 
                 vae_weight_decay=1e-3, 
                 multitask_lr=1e-3, 
                 multitask_weight_decay=1e-3,
                 alphas=None,
                 beta=1.0, 
                 gamma_task=1.0,
                 batch_size=200, 
                 validation_split=0.3, 
                 use_lr_scheduler=True,
                 lr_scheduler_factor=0.1,
                 lr_scheduler_patience=50,
                 use_batch_norm=True, 
                 ):
        super(HybridVAEMultiTaskModel, self).__init__()
        self.vae = VAE(input_dim, depth=vae_depth, hidden_dim=vae_hidden_dim, dropout_rate=vae_dropout_rate, latent_dim=latent_dim, use_batch_norm=use_batch_norm)
        self.predictor = MultiTaskPredictor(latent_dim, depth=predictor_depth, hidden_dim=predictor_hidden_dim, dropout_rate=predictor_dropout_rate, task_count=task_count, use_batch_norm=use_batch_norm)
        
        self.input_dim = input_dim
        self.vae_hidden_dim = vae_hidden_dim
        self.vae_depth = vae_depth
        self.vae_dropout_rate = vae_dropout_rate
        self.latent_dim = latent_dim
        self.predictor_hidden_dim = predictor_hidden_dim
        self.predictor_depth = predictor_depth
        self.predictor_dropout_rate = predictor_dropout_rate
        self.task_count = task_count
        self.vae_lr = vae_lr
        self.vae_weight_decay = vae_weight_decay
        self.multitask_lr = multitask_lr
        self.multitask_weight_decay = multitask_weight_decay
        self.alphas = alphas
        self.beta = beta
        self.gamma_task = gamma_task
        self.batch_size = batch_size
        self.validation_split = validation_split
        self.use_lr_scheduler = use_lr_scheduler
        self.lr_scheduler_factor = lr_scheduler_factor
        self.lr_scheduler_patience = lr_scheduler_patience
        self.use_batch_norm = use_batch_norm

    def reset_parameters(self, seed=19960816):
        for layer in self.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                torch.manual_seed(seed)  # 固定随机种子
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, x):
        """
        Forward pass through the complete model.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor with shape (batch_size, input_dim).

        Returns
        -------
        torch.Tensor
            Reconstructed tensor with shape (batch_size, input_dim).
        torch.Tensor
            Latent mean tensor with shape (batch_size, latent_dim).
        torch.Tensor
            Latent log variance tensor with shape (batch_size, latent_dim).
        torch.Tensor
            Latent representation tensor with shape (batch_size, latent_dim).
        list of torch.Tensor
            List of task-specific prediction tensors, each with shape (batch_size, 1).
        """
        x = self.check_tensor(x).to(DEVICE)
        recon, mu, logvar, z = self.vae(x)
        task_outputs = self.predictor(z)
        return recon, mu, logvar, z, task_outputs


    def check_tensor(self, X: torch.Tensor):
        """
        Ensures the input is a tensor and moves it to the correct device.

        Parameters
        ----------
        X : torch.Tensor or np.ndarray
            Input data to check and convert if necessary.

        Returns
        -------
        torch.Tensor
            Tensor moved to the specified device.

        Notes
        -----
        This method automatically handles input conversion from numpy arrays to tensors.
        """
        if isinstance(X, torch.Tensor):
            return X.to(DEVICE)
        else:
            return torch.from_numpy(np.asarray(X, dtype=np.float32)).to(DEVICE)

    def compute_loss(self, recon, x, mu, logvar, task_outputs, y, beta=1.0, gamma_task=1.0, alpha=None, normalize_loss=False):
        """
        Compute total loss for VAE and multi-task predictor.

        Parameters
        ----------
        recon : torch.Tensor
            Reconstructed input tensor, shape (batch_size, input_dim).
        x : torch.Tensor
            Original input tensor, shape (batch_size, input_dim).
        mu : torch.Tensor
            Latent mean tensor, shape (batch_size, latent_dim).
        logvar : torch.Tensor
            Latent log variance tensor, shape (batch_size, latent_dim).
        task_outputs : list of torch.Tensor
            List of task-specific predictions, each shape (batch_size, 1).
        y : torch.Tensor
            Ground truth target tensor, shape (batch_size, num_tasks).
        beta : float, optional
            Weight of the KL divergence term (default is 1.0).
        gamma_task : float, optional
            Weight of the task loss term (default is 1.0).
        alpha : list or torch.Tensor, optional
            Per-task weights, shape (num_tasks,). Default is uniform weights.
        normalize_loss : bool, optional
            If True, normalize the scale of recon_loss, kl_loss, and task_loss.

        Returns
        -------
        torch.Tensor
            Total loss value.
        torch.Tensor
            Normalized reconstruction loss (if normalize_loss is True).
        torch.Tensor
            Normalized KL divergence loss (if normalize_loss is True).
        torch.Tensor
            Normalized task-specific loss (if normalize_loss is True).
        """
        # Reconstruction loss
        reconstruction_loss_fn = nn.MSELoss(reduction='sum')
        recon_loss = reconstruction_loss_fn(recon, x)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Task-specific losses
        task_loss_fn = nn.BCELoss(reduction='mean')  # Assuming binary tasks
        if alpha is None:
            alpha = torch.ones(len(task_outputs), device=DEVICE)  # Default to uniform weights

        # Clamp task_outputs to [1e-6, 1 - 1e-6] to avoid log(0)
        task_outputs_clamped = [torch.clamp(output, min=1e-9, max=1-1e-9) for output in task_outputs]

        task_losses = [
            alpha[t] * task_loss_fn(task_outputs_clamped[t], y[:, t].unsqueeze(1))
            for t in range(len(task_outputs_clamped))
        ]
        task_loss = sum(task_losses)

        if normalize_loss:
            # Normalize each loss by its scale
            recon_loss_norm = recon_loss / (recon_loss.item() + 1e-8)
            kl_loss_norm = kl_loss / (kl_loss.item() + 1e-8)
            task_loss_norm = task_loss / (task_loss.item() + 1e-8)

            total_loss = beta * (recon_loss_norm + kl_loss_norm) + gamma_task * task_loss_norm
            return total_loss, recon_loss_norm, kl_loss_norm, task_loss_norm
        else:
            # Use raw losses with predefined weights
            total_loss = beta * (recon_loss + kl_loss) + gamma_task * task_loss
            return total_loss, recon_loss, kl_loss, task_loss

    
    def fit(self, X, Y, 
            epochs=2000, 
            early_stopping=True, 
            patience=100,
            verbose=True, 
            plot_path=None,
            save_weights_path=None):
        """
        Fits the VAEMultiTaskModel to the data.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Feature matrix of shape (n_samples, n_features).
        Y : np.ndarray or torch.Tensor
            Target matrix of shape (n_samples, n_tasks).
        epochs : int, optional
            Number of training epochs (default is 2500).
        early_stopping : bool, optional
            Whether to enable early stopping based on validation loss (default is False).
        patience : int, optional
            Number of epochs to wait for improvement in validation loss before stopping (default is 100).
        verbose : bool, optional
            If True, displays tqdm progress bar. If False, suppresses tqdm output (default is True).
        plot_path : str or None, optional
            Directory to save loss plots every 100 epochs. If None, attempt dynamic plotting in a notebook environment.
        save_weights_path : str or None, optional
            Directory to save model weights every 500 epochs and at the end of training. If None, weights are not saved.

        Returns
        -------
        self : VAEMultiTaskModel
            The fitted VAEMultiTaskModel instance.
        """
        self.reset_parameters()
        # Data checks and device transfer
        X = self.check_tensor(X).to(DEVICE)
        Y = self.check_tensor(Y).to(DEVICE)
        self.to(DEVICE)  # Ensure the model is on the correct device
        
        # Data validation
        if len(X) != len(Y):
            raise ValueError("Features and targets must have the same number of samples.")
        if torch.isnan(X).any():
            raise ValueError("Features (X) contain NaNs.")
        if torch.isnan(Y).any():
            raise ValueError("Targets (Y) contain NaNs.")

        # Split data into training and validation sets
        n_samples = len(X)
        n_val = int(n_samples * self.validation_split)
        perm = torch.randperm(n_samples)
        train_idx, val_idx = perm[:-n_val], perm[-n_val:]
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]

        # Separate optimizers for VAE and MultiTask predictor
        vae_optimizer = torch.optim.Adam(self.vae.parameters(), lr=self.vae_lr, weight_decay=self.vae_weight_decay, eps=1e-8)
        multitask_optimizer = torch.optim.Adam(self.predictor.parameters(), lr=self.multitask_lr, weight_decay=self.multitask_weight_decay, eps=1e-8)

        # 初始化调度器（仅当启用时）
        if self.use_lr_scheduler:
            vae_scheduler = ReduceLROnPlateau(vae_optimizer, mode='min', factor=0.1, patience=int(1/3 * patience))
            multitask_scheduler = ReduceLROnPlateau(multitask_optimizer, mode='min', factor=0.1, patience=int(1/3 * patience))

        # Initialize best losses and patience counters
        best_vae_loss = float('inf')
        best_task_loss = float('inf')
        vae_patience_counter = 0
        task_patience_counter = 0

        # Training and validation loss storage
        train_vae_losses, train_task_losses = [], []
        val_vae_losses, val_task_losses = [], []

        # Training loop with tqdm
        iterator = range(epochs)
        if verbose:  # 控制进度条显示
            iterator = tqdm(iterator, desc="Training Progress", unit="epoch")

        for epoch in iterator:
            self.train()
            train_vae_loss = 0.0
            train_task_loss = 0.0
            for i in range(0, len(X_train), self.batch_size):
                X_batch = X_train[i:i + self.batch_size]
                Y_batch = Y_train[i:i + self.batch_size]

                # Reset gradients
                vae_optimizer.zero_grad()
                multitask_optimizer.zero_grad()

                # Forward pass
                recon, mu, logvar, z, task_outputs = self(X_batch)

                # Compute loss
                total_loss, recon_loss, kl_loss, task_loss = self.compute_loss(
                    recon, X_batch, mu, logvar, task_outputs, Y_batch, 
                    beta=self.beta, gamma_task=self.gamma_task, alpha=self.alphas
                )

                # Backward pass and optimization
                total_loss.backward()
                vae_optimizer.step()
                multitask_optimizer.step()

                # Accumulate losses
                train_vae_loss += (recon_loss.item() + kl_loss.item())
                train_task_loss += task_loss.item()

            # Normalize training losses by the number of batches
            train_vae_loss /= len(X_train)
            train_task_loss /= len(X_train)
            train_vae_losses.append(train_vae_loss)
            train_task_losses.append(train_task_loss)

            # Validation phase
            self.eval()
            val_vae_loss = 0.0
            val_task_loss = 0.0
            with torch.no_grad():
                for i in range(0, len(X_val), self.batch_size):
                    X_batch = X_val[i:i + self.batch_size]
                    Y_batch = Y_val[i:i + self.batch_size]

                    # Forward pass
                    recon, mu, logvar, z, task_outputs = self(X_batch)

                    # Compute validation losses
                    total_loss, recon_loss, kl_loss, task_loss = self.compute_loss(
                        recon, X_batch, mu, logvar, task_outputs, Y_batch, 
                        beta=self.beta, gamma_task=self.gamma_task, alpha=self.alphas
                    )

                    # Accumulate losses
                    val_vae_loss += (recon_loss.item() + kl_loss.item())
                    val_task_loss += task_loss.item()

            if self.use_lr_scheduler:
                vae_scheduler.step(val_vae_loss)  # 调用时需传入验证损失
                multitask_scheduler.step(val_task_loss)

            # Normalize validation losses by the number of batches
            val_vae_loss /= len(X_val)
            val_task_loss /= len(X_val)
            val_vae_losses.append(val_vae_loss)
            val_task_losses.append(val_task_loss)
            val_total_loss = val_vae_loss + val_task_loss

            # Update progress bar
            if verbose:
                iterator.set_postfix({
                    "Train VAE Loss": f"{train_vae_loss:.4f}",
                    "Train Task Loss": f"{train_task_loss:.4f}",
                    "Val VAE Loss": f"{val_vae_loss:.4f}",
                    "Val Task Loss": f"{val_task_loss:.4f}"
                })
            
            # Early stopping logic
            if early_stopping:
                if val_vae_loss < best_vae_loss:
                    best_vae_loss = val_vae_loss
                    vae_patience_counter = 0
                else:
                    vae_patience_counter += 1

                if val_task_loss < best_task_loss:
                    best_task_loss = val_task_loss
                    task_patience_counter = 0
                else:
                    task_patience_counter += 1

                # Stop if both counters exceed patience
                if vae_patience_counter >= patience and task_patience_counter >= patience:
                    print("Early stopping triggered due to no improvement in both VAE and task losses.") if verbose > 0 else None
                    if save_weights_path:
                        self.save_model(save_weights_path, "final")
                    break

            # Save loss plot every 100 epochs
            if ((epoch + 1) % 100 == 0) and (is_interactive_environment() or plot_path):
                loss_plot_path = None
                if plot_path:
                    loss_plot_path = os.path.join(plot_path, f"loss_epoch.jpg")
                self.plot_loss(train_vae_losses, train_task_losses, val_vae_losses, val_task_losses, save_path=loss_plot_path)

            # Save weights every 500 epochs
            if (epoch + 1) % 500 == 0 and save_weights_path:
                self.save_model(save_weights_path, epoch + 1)

        # Save final weights
        if save_weights_path:
            self.save_model(save_weights_path, "final")
            print(f'最终模型参数已保存: {os.path.join(save_weights_path, f"epoch_final.pth")}') if verbose > 0 else None

        return self


    def plot_loss(self, train_vae_losses, train_task_losses, val_vae_losses, val_task_losses, save_path=None):
        """
        Plot training and validation loss curves for VAE and task-specific losses.

        Parameters
        ----------
        train_vae_losses : list
            List of VAE losses (reconstruction + KL) for the training set at each epoch.
        train_task_losses : list
            List of task-specific losses (BCE) for the training set at each epoch.
        val_vae_losses : list
            List of VAE losses (reconstruction + KL) for the validation set at each epoch.
        val_task_losses : list
            List of task-specific losses (BCE) for the validation set at each epoch.
        save_path : str or None
            Path to save the plot image. If None, dynamically display in a notebook.
        """
        plt.figure(figsize=(12, 6))

        # Check if log scale is needed
        use_log_scale = len(train_vae_losses) > 10000

        # VAE Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(train_vae_losses, label='Train VAE Loss')
        plt.plot(val_vae_losses, label='Val VAE Loss')
        plt.xlabel('Epochs (Log Scale)' if use_log_scale else 'Epochs')
        plt.ylabel('VAE Loss')
        plt.legend()
        plt.title('VAE Loss (Reconstruction + KL)')
        plt.grid()
        if use_log_scale:
            plt.xscale('log')

        # Task Loss Plot
        plt.subplot(1, 2, 2)
        plt.plot(train_task_losses, label='Train Task Loss')
        plt.plot(val_task_losses, label='Val Task Loss')
        plt.xlabel('Epochs (Log Scale)' if use_log_scale else 'Epochs')
        plt.ylabel('Task Loss (BCE)')
        plt.legend()
        plt.title('Task Loss (BCE)')
        plt.grid()
        if use_log_scale:
            plt.xscale('log')

        if save_path:
            plt.savefig(save_path, dpi=360)

        # Check if running in notebook
        if hasattr(sys, 'ps1') or ('IPython' in sys.modules and hasattr(sys, 'argv') and sys.argv[0].endswith('notebook')):
            try:
                from IPython.display import display, clear_output
                clear_output(wait=True)
                display(plt.gcf())
                plt.pause(0.1)
            except ImportError:
                pass
        
        # Close the plot if it was saved, but avoid closing for interactive use
        plt.close()

    def save_model(self, save_path, epoch):
        """
        Save model weights.

        Parameters
        ----------
        save_path : str
            Directory to save the weights.
        epoch : int
            Current epoch number, used for naming the file.
        """
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            torch.save(self.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pth"))


class HybridVAEMultiTaskSklearn(HybridVAEMultiTaskModel, BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    A hybrid model combining VAE and Multi-Task Predictor with scikit-learn compatible interface.

    Methods
    -------
    fit(X, Y, *args, **kwargs)
        Fits the model to the data.
    transform(X)
        Transforms input samples into latent space.
    inverse_transform(Z)
        Reconstructs samples from latent space parameters.
    predict_proba(X)
        Predicts probabilities for each task.
    predict(X, threshold=0.5)
        Predicts binary classifications for each task based on a threshold.
    score(X, Y)
        Computes AUC scores for each task.

    Attributes
    ----------
    feature_names_in_ : list of str
        Feature names for input data.
    """

    def __init__(self, 
                 input_dim=30, 
                 vae_hidden_dim=64, 
                 vae_depth=3,
                 vae_dropout_rate=0.3, 
                 latent_dim=10, 
                 predictor_hidden_dim=64, 
                 predictor_depth=3,
                 predictor_dropout_rate=0.3, 
                 task_count=2,
                 # training related params for param tuning
                 vae_lr=1e-3, 
                 vae_weight_decay=1e-3, 
                 multitask_lr=1e-3, 
                 multitask_weight_decay=1e-3,
                 alphas=None,
                 beta=1.0, 
                 gamma_task=1.0,
                 batch_size=200, 
                 validation_split=0.3, 
                 use_lr_scheduler=True,
                 lr_scheduler_factor=0.1,
                 lr_scheduler_patience=50,
                 use_batch_norm=True, 
                 ):
        super().__init__(input_dim=input_dim, 
                         vae_hidden_dim=vae_hidden_dim, 
                         vae_depth=vae_depth, 
                         vae_dropout_rate=vae_dropout_rate, 
                         latent_dim=latent_dim, 
                         predictor_hidden_dim=predictor_hidden_dim, 
                         predictor_depth=predictor_depth, 
                         predictor_dropout_rate=predictor_dropout_rate, 
                         task_count=task_count,
                         vae_lr=vae_lr, 
                         vae_weight_decay=vae_weight_decay, 
                         multitask_lr=multitask_lr,
                         multitask_weight_decay=multitask_weight_decay,
                         alphas=alphas,
                         beta=beta,
                         gamma_task=gamma_task,
                         batch_size=batch_size,
                         validation_split=validation_split,
                         use_lr_scheduler=use_lr_scheduler,
                         lr_scheduler_factor=lr_scheduler_factor,
                         lr_scheduler_patience=lr_scheduler_patience,
                         use_batch_norm=use_batch_norm,
                         )
    
    def fit(self, X, y, *args, **kwargs):
        """
        see `HybridVAEMultiTaskModel.fit` 
        """
        # Record feature names if provided (e.g., pandas DataFrame)
        if hasattr(X, "columns"):
            self.feature_names_in_ = X.columns.tolist()
        else:
            self.feature_names_in_ = [f"x{i}" for i in range(X.shape[1])]

        return super().fit(X, y, *args, **kwargs)
    
    def transform(self, X):
        """
        Transforms the input samples into the latent space.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).

        Returns
        -------
        Z : np.ndarray
            Latent space representations with shape (n_samples, latent_dim).
        """
        X = self.check_tensor(X).to(DEVICE)
        self.eval()
        with torch.no_grad():
            mu, _ = self.vae.encoder(X)
        return mu.cpu().numpy()

    def sample_latent(self, X, n_samples=1):
        """
        Sample from the latent space using reparameterization trick.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        n_samples : int, optional
            Number of samples to generate for each input (default is 1).

        Returns
        -------
        Z : np.ndarray
            Sampled latent representations with shape (n_samples, latent_dim).
        """
        X = self.check_tensor(X).to(DEVICE)
        self.eval()
        with torch.no_grad():
            mu, logvar = self.vae.encoder(X)
            Z = [self.vae.reparameterize(mu, logvar) for _ in range(n_samples)]
        return torch.stack(Z, dim=1).cpu().numpy()  # Shape: (input_samples, n_samples, latent_dim)
    
    def inverse_transform(self, Z):
        """
        Reconstructs samples from the latent space.

        Parameters
        ----------
        Z : np.ndarray or torch.Tensor
            Latent space representations with shape (n_samples, latent_dim).

        Returns
        -------
        X_recon : np.ndarray
            Reconstructed samples with shape (n_samples, input_dim).
        """
        Z = self.check_tensor(Z).to(DEVICE)
        self.eval()
        with torch.no_grad():
            recon = self.vae.decoder(Z)
        return recon.cpu().numpy()

    def predict_proba(self, X, deterministic=True):
        """
        Predicts probabilities for each task.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        deterministic : bool, optional
            If True, uses the latent mean (mu) for predictions, avoiding randomness.
            If False, samples from the latent space using reparameterization trick.

        Returns
        -------
        probas : np.ndarray
            Probabilities for each task, shape (n_tasks, n_samples).
        """
        X = self.check_tensor(X).to(DEVICE)
        self.eval()  # Ensure model is in evaluation mode
        with torch.no_grad():
            mu, logvar = self.vae.encoder(X)
            if deterministic:
                z = mu  # Use latent mean for deterministic predictions
            else:
                z = self.vae.reparameterize(mu, logvar)  # Sample from latent space
            task_outputs = self.predictor(z)
        probas = torch.cat([out.unsqueeze(0) for out in task_outputs], dim=0)
        return probas.cpu().numpy()


    def predict(self, X, threshold=0.5):
        """
        Predicts binary classifications for each task.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        threshold : float, optional
            Decision threshold for binary classification (default is 0.5).

        Returns
        -------
        predictions : np.ndarray
            Binary predictions for each task, shape (n_tasks, n_samples).
        """
        self.eval()
        probas = self.predict_proba(X)
        return (probas >= threshold).astype(int)

    def score(self, X, Y, average="weighted", *args, **kwargs):
        """
        Computes AUC scores for each task.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input samples with shape (n_samples, n_features).
        Y : np.ndarray
            Ground truth labels with shape (n_samples, n_tasks).
        average : str, optional
            Averaging method for multi-class AUC computation (default is "weighted").

        Returns
        -------
        scores : np.ndarray
            AUC scores for each task, shape (n_tasks,).
        """
        self.eval()
        probas = self.predict_proba(X)
        Y = self.check_tensor(Y).to(DEVICE).cpu().numpy()
        scores = []
        for t in range(self.task_count):
            auc = roc_auc_score(Y[:, t], probas[t, :], average=average, *args, **kwargs)
            scores.append(auc)
        return np.array(scores)
    
    def eval_loss(self, X, Y):
        X = self.check_tensor(X).to(DEVICE)
        Y = self.check_tensor(Y).to(DEVICE)
        self.eval()
        # Forward pass
        with torch.no_grad():
            recon, mu, logvar, z, task_outputs = self(X)
            total_loss, recon_loss, kl_loss, task_loss = self.compute_loss(
                recon, X, mu, logvar, task_outputs, Y, 
                beta=self.beta, gamma_task=self.gamma_task, alpha=self.alphas, normalize_loss=False
            )
        
        # Convert losses to NumPy arrays
        return (total_loss.item(),  # Convert scalar tensor to Python float
                recon_loss.item(),
                kl_loss.item(),
                task_loss.item())
    
    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names (latent space).

        Returns
        -------
        output_feature_names : list of str
            Output feature names for the latent space.
        """
        return [f"latent_{i}" for i in range(self.vae.encoder.latent_mu.out_features)]

def is_interactive_environment():
    """
    Detect if the code is running in an interactive environment (e.g., Jupyter Notebook or IPython).

    Returns
    -------
    bool
        True if running in an interactive environment, False otherwise.
    """
    try:
        # Check if running in IPython or Jupyter Notebook
        if hasattr(sys, 'ps1'):  # Standard interactive interpreter (Python REPL)
            return True
        if 'IPython' in sys.modules:  # IPython or Jupyter environment
            import IPython
            return IPython.get_ipython() is not None
    except ImportError:
        pass  # IPython not installed

    return False  # Not an interactive environment