# stdlib
from typing import Tuple

import ganite_mod.logger as log

# third party
import numpy as np
import torch
from ganite_mod.utils.random import enable_reproducible_results
from ganite_mod.utils.metrics import RCT_ATE_l1_loss
from torch import nn
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import mean_squared_error

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPS = 1e-10


class CounterfactualGenerator(nn.Module):
    """
    The counterfactual generator, G, uses the feature vector x,
    the treatment vector t, and the factual outcome yf, to generate
    a potential outcome vector, hat_y.
    """

    def __init__(
        self, Dim: int, TreatmentsCnt: int, DimHidden: int, depth: int, binary_y: bool
    ) -> None:
        super(CounterfactualGenerator, self).__init__()
        # Generator Layer
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Dropout(),
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.common = nn.Sequential(
            nn.Linear(
                Dim + 2, DimHidden
            ),  # Inputs: X + Treatment (1) + Factual Outcome (1) + Random Vector      (Z)
            nn.LeakyReLU(),
            *hidden,
        ).to(DEVICE)

        self.binary_y = binary_y
        self.outs = []
        for tidx in range(TreatmentsCnt):
            self.outs.append(
                nn.Sequential(
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                    nn.Linear(DimHidden, 1),
                ).to(DEVICE)
            )

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        inputs = torch.cat([x, t, y], dim=1).to(DEVICE)

        G_h2 = self.common(inputs)

        G_prob1 = self.outs[0](G_h2)
        G_prob2 = self.outs[1](G_h2)

        G_prob = torch.cat([G_prob1, G_prob2], dim=1).to(DEVICE)

        if self.binary_y:
            return torch.sigmoid(G_prob)
        else:
            return G_prob


class CounterfactualDiscriminator(nn.Module):
    """
    The discriminator maps pairs (x, hat_y) to vectors in [0, 1]^2
    representing probabilities that the i-th component of hat_y
    is the factual outcome.
    """

    def __init__(
        self, Dim: int, Treatments: list, DimHidden: int, depth: int, binary_y: bool
    ) -> None:
        super(CounterfactualDiscriminator, self).__init__()
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.Treatments = Treatments

        self.model = nn.Sequential(
            nn.Linear(Dim + len(Treatments), DimHidden),
            nn.LeakyReLU(),
            *hidden,
            nn.Linear(DimHidden, 1),
            nn.Sigmoid(),
        ).to(DEVICE)

    def forward(
        self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, hat_y: torch.Tensor
    ) -> torch.Tensor:
        # Factual & Counterfactual outcomes concatenate
        inp0 = (1.0 - t) * y + t * hat_y[:, 0].reshape([-1, 1])
        inp1 = t * y + (1.0 - t) * hat_y[:, 1].reshape([-1, 1])

        inputs = torch.cat([x, inp0, inp1], dim=1).to(DEVICE)
        return self.model(inputs)


class InferenceNets(nn.Module):
    """
    The ITE generator uses only the feature vector, x, to generate a potential outcome vector hat_y.
    """

    def __init__(
        self, Dim: int, TreatmentsCnt: int, DimHidden: int, depth: int, binary_y: bool
    ) -> None:
        super(InferenceNets, self).__init__()
        hidden = []

        for d in range(depth):
            hidden.extend(
                [
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                ]
            )

        self.common = nn.Sequential(
            nn.Linear(Dim, DimHidden),
            nn.LeakyReLU(),
            *hidden,
        ).to(DEVICE)
        self.binary_y = binary_y

        self.outs = []
        for tidx in range(TreatmentsCnt):
            self.outs.append(
                nn.Sequential(
                    nn.Linear(DimHidden, DimHidden),
                    nn.LeakyReLU(),
                    nn.Linear(DimHidden, 1),
                ).to(DEVICE)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        I_h = self.common(x)

        I_probs = []
        for out in self.outs:
            I_probs.append(out(I_h))

        if self.binary_y:
            return torch.sigmoid(torch.cat(I_probs, dim=1).to(DEVICE))
        else:
            return torch.cat(I_probs, dim=1).to(DEVICE)


class Ganite(nn.Module):
    """
    The GANITE framework generates potential outcomes for a given feature vector X.
    It consists of two components:
        - The Counterfactual Generator block (generator + discriminator)
        - The ITE block (Inference Nets).
    
    Parameters
    ----------
    dim_in : int
        Number of the input features, used to configure network dimensions.
    binary_y : bool
        Indicates whether the observed outcome is binary (True) or continuous (False).
    dim_hidden : int, optional
        The number of hidden units in the neural network (default is 100).
    alpha : float, optional
        Weight for the generator's GAN loss (default is 0.1).
    beta : float, optional
        Weight for the inference loss component (default is 0).
    minibatch_size : int, optional
        Batch size for training (default is 256).
    depth : int, optional
        Depth of the neural network (default is 0).
    num_iterations : int, optional
        Number of training iterations (default is 5000).
    num_discr_iterations : int, optional
        Number of iterations for the discriminator's update in each training step (default is 1).

    Raises
    ------
    ValueError
        If X contains NaNs, or if the sizes of X, Treatments, and Y are inconsistent, or
        if the treatment categories are not exactly two.

    Notes
    -----
    - This implementation assumes binary treatment categories (0 and 1).
    - GPU acceleration is supported if a compatible GPU is available.

    Examples
    --------
    >>> X = torch.rand(100, 10)
    >>> Treatments = torch.randint(0, 2, (100,))
    >>> Y = torch.rand(100)
    >>> model = Ganite(X, Treatments, Y)
    >>> predictions = model(X)
    """

    def __init__(
        self,
        dim_in: int,
        binary_y: bool,
        dim_hidden: int = 100,
        alpha: float = 0.1,
        beta: float = 1.0,
        minibatch_size: int = 256,
        depth: int = 0,
        num_iterations: int = 5000,
        num_discr_iterations: int = 1,
    ) -> None:
        super(Ganite, self).__init__()
        enable_reproducible_results()

        self.treatments = [0, 1]

        # Hyperparameters
        self.minibatch_size = minibatch_size
        self.alpha = alpha
        self.beta = beta
        self.depth = depth
        self.num_iterations = num_iterations
        self.num_discr_iterations = num_discr_iterations

        # Layers
        self.counterfactual_generator = CounterfactualGenerator(
            dim_in, len(self.treatments), dim_hidden, depth, binary_y
        ).to(DEVICE)
        self.counterfactual_discriminator = CounterfactualDiscriminator(
            dim_in, self.treatments, dim_hidden, depth, binary_y
        ).to(DEVICE)
        self.inference_nets = InferenceNets(
            dim_in, len(self.treatments), dim_hidden, depth, binary_y
        ).to(DEVICE)

        # Solvers
        self.DG_solver = torch.optim.Adam(
            list(self.counterfactual_generator.parameters())
            + list(self.counterfactual_discriminator.parameters()),
            lr=1e-3,
            eps=1e-8,
            weight_decay=1e-3,
        )
        self.I_solver = torch.optim.Adam(
            self.inference_nets.parameters(), lr=1e-3, weight_decay=1e-3
        )

    def _sample_minibatch(
        self, X: torch.Tensor, T: torch.tensor, Y: torch.Tensor
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Samples a minibatch of data for training.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (n_samples, n_features).
        T : torch.Tensor
            Treatment assignment vector of shape (n_samples, 1).
        Y : torch.Tensor
            Observed outcome vector of shape (n_samples, 1).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing minibatch samples for X, T, and Y.

        Notes
        -----
        The size of the minibatch is determined by self.minibatch_size.
        """
        idx_mb = np.random.randint(0, X.shape[0], self.minibatch_size)

        X_mb = X[idx_mb, :]
        T_mb = T[idx_mb].reshape([self.minibatch_size, 1])
        Y_mb = Y[idx_mb].reshape([self.minibatch_size, 1])

        return X_mb, T_mb, Y_mb

    def fit(
        self,
        X: torch.Tensor,
        Treatment: torch.Tensor,
        Y: torch.Tensor,
    ) -> "Ganite":
        """
        Trains the GANITE model, including the generator, discriminator, and inference networks.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (n_samples, n_features).
        Treatment : torch.Tensor
            Treatment assignment vector of shape (n_samples, 1).
        Y : torch.Tensor
            Observed outcome vector of shape (n_samples, 1).

        Returns
        -------
        Ganite
            The trained GANITE model.

        Notes
        -----
        - This function includes two training phases: training the generator-discriminator pair
          and training the inference networks.
        - Training logs include generator and inference loss metrics.
        """

        X = self._check_tensor(X)
        Treatment = self._check_tensor(Treatment)
        Y = self._check_tensor(Y)
        
        if np.isnan(np.sum(X.cpu().numpy())):
            raise ValueError("X contains NaNs")
        if len(X) != len(Treatment):
            raise ValueError("Features/Treatments mismatch")
        if len(X) != len(Y):
            raise ValueError("Features/Labels mismatch")

        self.original_treatments = np.sort(np.unique(Treatment.cpu().numpy()))
        if len(self.original_treatments) != 2:
            raise ValueError("Only two treatment categories supported")

        Train_X = self._check_tensor(X).float()
        Train_T = self._check_tensor(Treatment).float().reshape([-1, 1])
        Train_Y = self._check_tensor(Y).float().reshape([-1, 1])
        # Encode
        min_t_val = Train_T.min()
        Train_T = (Train_T > min_t_val).float()

        # Iterations
        # Train G and D first
        self.counterfactual_generator.train()
        self.counterfactual_discriminator.train()
        self.inference_nets.train()

        for it in range(self.num_iterations):
            self.DG_solver.zero_grad()

            X_mb, T_mb, Y_mb = self._sample_minibatch(Train_X, Train_T, Train_Y)

            for kk in range(self.num_discr_iterations):
                self.DG_solver.zero_grad()

                Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb).clone()
                D_out = self.counterfactual_discriminator(X_mb, T_mb, Y_mb, Tilde)

                if torch.isnan(Tilde).any():
                    raise RuntimeError("counterfactual_generator generated NaNs")
                if torch.isnan(D_out).any():
                    raise RuntimeError("counterfactual_discriminator generated NaNs")

                D_loss = nn.BCELoss()(D_out, T_mb)

                D_loss.backward()
                self.DG_solver.step()

            Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)
            D_out = self.counterfactual_discriminator(X_mb, T_mb, Y_mb, Tilde)
            D_loss = nn.BCELoss()(D_out, T_mb)

            G_loss_GAN = D_loss

            G_loss_R = torch.mean(
                nn.MSELoss()(
                    Y_mb,
                    T_mb * Tilde[:, 1].reshape([-1, 1])
                    + (1.0 - T_mb) * Tilde[:, 0].reshape([-1, 1]),
                )
            )

            G_loss = G_loss_R + self.alpha * G_loss_GAN

            if it % 100 == 0:
                log.debug(f"Generator loss epoch {it}: {D_loss} {G_loss}")
                if torch.isnan(D_loss).any():
                    raise RuntimeError("counterfactual_discriminator generated NaNs")

                if torch.isnan(G_loss).any():
                    raise RuntimeError("counterfactual_generator generated NaNs")

            G_loss.backward()
            self.DG_solver.step()

        # Train I and ID
        for it in range(self.num_iterations):
            self.I_solver.zero_grad()

            X_mb, T_mb, Y_mb = self._sample_minibatch(Train_X, Train_T, Train_Y)

            Tilde = self.counterfactual_generator(X_mb, T_mb, Y_mb)

            hat = self.inference_nets(X_mb)

            I_loss1: torch.Tensor = 0
            I_loss2: torch.Tensor = 0

            I_loss1 = torch.mean(
                nn.MSELoss()(
                    T_mb * Y_mb + (1 - T_mb) * Tilde[:, 1].reshape([-1, 1]),
                    hat[:, 1].reshape([-1, 1]),
                )
            )
            I_loss2 = torch.mean(
                nn.MSELoss()(
                    (1 - T_mb) * Y_mb + T_mb * Tilde[:, 0].reshape([-1, 1]),
                    hat[:, 0].reshape([-1, 1]),
                )
            )
            I_loss = I_loss1 + self.beta * I_loss2

            if it % 100 == 0:
                log.debug(f"Inference loss epoch {it}: {I_loss}")

            I_loss.backward()
            self.I_solver.step()

        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Computes the Individual Treatment Effect (ITE) and returns both potential outcomes.

        Parameters
        ----------
        X : torch.Tensor
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            A tuple containing:
            - hat{Y}(1): Predicted outcome under treatment (T=1), shape (n_samples,).
            - hat{Y}(0): Predicted outcome under control (T=0), shape (n_samples,).
            - ITE: Individual Treatment Effect (hat{Y}(1) - hat{Y}(0)), shape (n_samples,).

        Notes
        -----
        The method computes the potential outcomes `hat{Y}(1)` and `=hat{Y}(0)` using the inference networks.
        The Individual Treatment Effect (ITE) is derived as:
        `ITE = hat{Y}(1) - hat{Y}(0)`
        """

        with torch.no_grad():
            X = self._check_tensor(X).float()
            y_hat = self.inference_nets(X).detach()

        return y_hat[:, 1], y_hat[:, 0], y_hat[:, 1] - y_hat[:, 0]

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
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
            return torch.from_numpy(np.asarray(X)).to(DEVICE)


class GaniteRegressor(Ganite, BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible wrapper for the GANITE framework.
    This implementation extends Ganite to integrate with scikit-learn's
    pipeline and model selection utilities.

    Parameters
    ----------
    dim_in : int
        Number of the input features, used to configure network dimensions.
    binary_y : bool
        Indicates whether the observed outcome is binary (True) or continuous (False).
    dim_hidden : int, optional
        The number of hidden units in the neural network (default is 100).
    alpha : float, optional
        Weight for the generator's GAN loss (default is 0.1).
    beta : float, optional
        Weight for the inference loss component (default is 0).
    minibatch_size : int, optional
        Batch size for training (default is 256).
    depth : int, optional
        Depth of the neural network (default is 0).
    num_iterations : int, optional
        Number of training iterations (default is 5000).
    num_discr_iterations : int, optional
        Number of iterations for the discriminator's update in each training step (default is 1).
    """

    def __init__(
        self,
        dim_in,
        binary_y,
        dim_hidden=100,
        alpha=0.1,
        beta=0,
        minibatch_size=256,
        depth=0,
        num_iterations=5000,
        num_discr_iterations=1,
    ):
        super(GaniteRegressor, self).__init__(
            dim_in=dim_in,
            binary_y=binary_y,
            dim_hidden=dim_hidden,
            alpha=alpha,
            beta=beta,
            minibatch_size=minibatch_size,
            depth=depth,
            num_iterations=num_iterations,
            num_discr_iterations=num_discr_iterations,
        )

    def fit(self, X, y):
        """
        Wrapper for the GANITE `fit` method. Assumes `X` is a tuple (features, treatment).

        Parameters
        ----------
        X : tuple of (np.ndarray or torch.Tensor, np.ndarray or torch.Tensor)
            - `X[0]`: Feature matrix of shape (n_samples, n_features).
            - `X[1]`: Treatment assignment vector of shape (n_samples,).
        y : np.ndarray or torch.Tensor
            Observed outcomes vector of shape (n_samples,).

        Returns
        -------
        self : GaniteRegressor
            The fitted model.
        """
        X_features, treatment = X
        return super().fit(X_features, treatment, y)
    
    def predict(self, X):
        """
        Predict potential outcomes for the provided feature matrix.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Feature matrix of shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            A tuple of predicted outcomes:
            - hat{Y}(1): Predicted outcome under treatment (T=1), shape (n_samples,).
            - hat{Y}(0): Predicted outcome under control (T=0), shape (n_samples,).
            - ITE: Individual Treatment Effect (hat{Y}(1) - hat{Y}(0)), shape (n_samples,).
        """
        X = self._check_tensor(X).float()
        hat_y1, hat_y0, ite = super().forward(X)
        return (
            hat_y1.cpu().numpy(),
            hat_y0.cpu().numpy(),
            ite.cpu().numpy(),
        )

    def get_params(self, deep=True):
        """
        Get parameters for this regressor. Required for compatibility with sklearn.

        Parameters
        ----------
        deep : bool, optional
            Whether to return deep parameters (default is True).

        Returns
        -------
        dict
            Dictionary of parameters.
        """
        return {
            "dim_in": self.counterfactual_generator.dim_in,
            "binary_y": self.counterfactual_generator.binary_y,
            "dim_hidden": self.counterfactual_generator.dim_hidden,
            "alpha": self.alpha,
            "beta": self.beta,
            "minibatch_size": self.minibatch_size,
            "depth": self.depth,
            "num_iterations": self.num_iterations,
            "num_discr_iterations": self.num_discr_iterations,
        }

    def set_params(self, **params):
        """
        Set parameters for this regressor. Required for compatibility with sklearn.

        Parameters
        ----------
        **params : dict
            Parameters to set.

        Returns
        -------
        self : GaniteRegressor
            The updated regressor.
        """
        for param, value in params.items():
            setattr(self, param, value)
        return self
    
    def score(self, X, y):
        """
        Returns the negative mean squared error of the model on the given data.

        Parameters
        ----------
        X : tuple of (features, treatment)
            - `X[0]`: Feature matrix of shape (n_samples, n_features).
            - `X[1]`: Treatment assignment vector of shape (n_samples,).
        y : np.ndarray or torch.Tensor
            Observed outcomes vector of shape (n_samples,).

        Returns
        -------
        float
            Negative mean squared error as the score.
        """
        
        X_features, treatment = X
        treatment = treatment.cpu().numpy() if isinstance(treatment, torch.Tensor) else treatment
        y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

        # Predict potential outcomes
        hat_y1, hat_y0, _ = self.predict(X_features)  # hat_y1, hat_y0 are numpy arrays

        # Select predictions based on treatment
        y_pred = np.where(treatment == 1, hat_y1, hat_y0)  # Choose hat_y1 if treated, else hat_y0

        # Ensure y and y_pred are numpy arrays for compatibility with sklearn.metrics
        y_pred = y_pred if isinstance(y_pred, np.ndarray) else y_pred.cpu().numpy()

        # Calculate and return the negative mean squared error
        return -mean_squared_error(y, y_pred)

    def ate_l1_loss(self, X, y, eval_strategy='mean_ITE'):
        """
        Returns the negative mean squared error of the model on the given data.

        Parameters
        ----------
        X : tuple of (features, treatment)
            - `X[0]`: Feature matrix of shape (n_samples, n_features).
            - `X[1]`: Treatment assignment vector of shape (n_samples,).
        y : np.ndarray or torch.Tensor
            Observed outcomes vector of shape (n_samples,).
        eval_strategy : str, optional
            Strategy to calculate the predicted ATE:
            - 'observed_only': Computes ATE_pred using only observed treatment groups.
            - 'mean_ITE': Computes ATE_pred as the mean of the predicted individual treatment effects (default is 'observed_only').

        Returns
        -------
        float
            Negative mean squared error as the score.
        """

        X_features, treatment = X
        X_features = self._check_tensor(X_features).float()
        treatment = self._check_tensor(treatment).float()

        treatment = treatment.cpu().numpy() if isinstance(treatment, torch.Tensor) else treatment
        y = y.cpu().numpy() if isinstance(y, torch.Tensor) else y

        # Predict potential outcomes
        hat_y1, hat_y0, _ = self.predict(X_features)  # hat_y1, hat_y0 are numpy arrays

        # Calculate and return the negative mean squared error
        return -RCT_ATE_l1_loss(treatment, y, hat_y0, hat_y1, eval_strategy=eval_strategy)