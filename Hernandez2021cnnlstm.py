import torch
from torch import nn


class TemporalWeightedMSE(nn.Module):
    """
    Loss MSE pondérée temporellement pour prédictions multi-horizon.

    Permet de donner plus d'importance aux prédictions proches (t+1)
    qu'aux prédictions lointaines (t+H).

    Parfait pour contrôle temps réel où on utilise seulement la première
    prédiction pour l'exosquelette, mais on veut quand même que les
    prédictions futures restent cohérentes.

    Args:
        weights (list or torch.Tensor):
            Poids pour chaque timestep du horizon.
            Longueur = horizon_size (ex: 20)
            Exemple: [1.0, 0.95, 0.90, ..., 0.39]

        normalize (bool):
            Si True, normalise les poids pour que leur somme = 1

    Shape:
        - Input (y_pred): (batch, horizon, n_outputs)
        - Target (y_true): (batch, horizon, n_outputs)
        - Output: scalar
    """

    def __init__(self, weights, normalize=True):
        super().__init__()

        # Convertir en tensor
        if not isinstance(weights, torch.Tensor):
            weights = torch.tensor(weights, dtype=torch.float32)

        # Normalisation (optionnelle)
        if normalize:
            weights = weights / weights.sum()

        # Enregistrer comme buffer (pas un paramètre entraînable)
        self.register_buffer("weights", weights)

    def forward(self, y_pred, y_true):

        # Vérifier dimensions
        assert y_pred.shape == y_true.shape, \
            f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"

        batch_size, horizon, n_outputs = y_pred.shape

        # Vérifier cohérence horizon
        assert len(self.weights) == horizon, \
            f"Nombre de poids ({len(self.weights)}) != horizon_size ({horizon})"

        # Squared error
        se = (y_pred - y_true) ** 2      # (batch, horizon, n_outputs)

        # Moyenne sur les articulations
        se = se.mean(dim=2)              # (batch, horizon)

        # Appliquer poids temporels
        weighted_se = se * self.weights.to(se.device)

        # Moyenne finale
        loss = weighted_se.mean()
        return loss


class TemporalWeightedMSEWithDecay(nn.Module):
    """
    Version avec décroissance exponentielle automatique.

    Args:
        horizon_size (int)
        decay_rate (float): entre 0 et 1
            1.0 = pas de décroissance
            0.95 = douce
            0.90 = moyenne
            0.80 = forte
        normalize (bool)
    """

    def __init__(self, horizon_size, decay_rate=0.95, normalize=True):
        super().__init__()

        assert 0 < decay_rate <= 1.0, "decay_rate doit être entre 0 et 1"

        # Générer poids exponentiels
        weights = torch.tensor(
            [decay_rate ** i for i in range(horizon_size)],
            dtype=torch.float32
        )

        if normalize:
            weights = weights / weights.sum()

        self.register_buffer("weights", weights)

    def forward(self, y_pred, y_true):

        assert y_pred.shape == y_true.shape, \
            f"Shape mismatch: y_pred {y_pred.shape} vs y_true {y_true.shape}"

        batch_size, horizon, n_outputs = y_pred.shape

        assert len(self.weights) == horizon, \
            f"Nombre de poids ({len(self.weights)}) != horizon_size ({horizon})"

        se = (y_pred - y_true) ** 2
        se = se.mean(dim=2)

        weighted_se = se * self.weights.to(se.device)
        return weighted_se.mean()
