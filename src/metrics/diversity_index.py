import math
import torch
import numpy as np
from core.container import DiversityIndexContainer

# from extension.cluster_coef_2 import (
#     compute_cluster_coef_from_distance_matrix_batch,
#     compute_cluster_coef_from_distance_matrix_batch_instance_wise,
# )
from extension.cluster_coef_para import (
    compute_cluster_coef_from_distance_matrix_batch,
    compute_cluster_coef_from_distance_matrix_batch_instance_wise,
)
from extension.cluster_coef_para_nodeweight import (
    compute_cluster_coef_from_distance_matrix_batch as cluster_coef_nodeweight
)


class DiversityIndex(object):
    """Calculate Diversity Index

    Attributes
    ----------
    epsilon : float
        Radius of the feasible reagion (lp-ball).
    sqdim : int
        Square root of the dimension of each search points.
    size_of_feasible_region: float
        The size of feasible region defined as 2 * epsilon * sqdim.
    data_container : DiversityIndexContainer
        Data container which has the search points and their distance matrix.
    """

    def __init__(self, epsilon, num_nodes, *args, **kwargs):
        super().__init__()
        self.epsilon = epsilon

        self.sqdim = None  # sqrt(dimension)
        self.size_of_feasible_region = None  # diagonal length of search space
        self.data_container = DiversityIndexContainer(K=num_nodes)

    def __call__(self, xk: torch.Tensor, eta: torch.Tensor):
        """Compute Diversity Index (local and global)

        Parameters
        ----------
        xk : torch.Tensor
            k-th search point.
        eta : torch.Tensor
            Current step size.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Diversity Index (global, local)
        """
        if self.data_container.push(xk) == True:
            diversity_index_1 = 1.0 - self._get_cluster_coef1()
            diversity_index_2 = None if eta is None else 1.0 - self._get_cluster_coef2(eta)
            return diversity_index_1, diversity_index_2
        else:
            if self.sqdim is None:
                self.sqdim = math.sqrt(xk.view(xk.shape[0], -1).shape[1])
            if self.size_of_feasible_region is None:
                self.size_of_feasible_region = 2 * self.epsilon * self.sqdim
            return None, None

    def _get_cluster_coef1(self):
        """Compute global Diversity Index

        Returns
        -------
        torch.Tensor
            Global Diversity Index
        """
        bs, n, _ = self.data_container.DistanceMatrix.shape
        cluster_coef = np.zeros((bs,), dtype=np.float64)
        compute_cluster_coef_from_distance_matrix_batch(
            bs,
            n,
            self.size_of_feasible_region,
            self.data_container.DistanceMatrix.numpy().astype(np.float32).reshape(-1),
            cluster_coef,
        )
        average_coef = torch.from_numpy(cluster_coef)
        return average_coef

    def _get_cluster_coef2(self, eta: torch.Tensor):
        """Compute local Diversity Index

        Parameters
        ----------
        eta : torch.Tensor
            Current step size

        Returns
        -------
        torch.Tensor
            Local Diversity Index
        """
        bs, n, _ = self.data_container.DistanceMatrix.shape
        unit = eta * self.sqdim
        unit = unit.cpu().squeeze(1).squeeze(1).squeeze(1).numpy().astype(np.float64)
        cluster_coef = np.zeros((bs,), dtype=np.float64)
        compute_cluster_coef_from_distance_matrix_batch_instance_wise(
            bs,
            n,
            unit,
            self.data_container.DistanceMatrix.numpy().astype(np.float32).reshape(-1),
            cluster_coef,
        )
        average_coef = torch.from_numpy(cluster_coef)
        return average_coef

    def clear(self):
        """Reset each attribute to None."""
        self.data_container.clear()
        self.sqdim = None
        self.size_of_feasible_region = None

    @torch.inference_mode()
    def get(self, X, weights=None):
        """Compute Diversity Index of the node set X.

        Parameters
        ----------
        X : torch.Tensor
            The set of points.

        Returns
        -------
        torch.Tensor
            The global Diversity Index
        """
        n, bs = X.shape[:2]
        _X = X.permute(1, 0, 2, 3, 4).reshape((bs, n, -1))
        if self.sqdim is None:
            self.sqdim = math.sqrt(_X.shape[-1])
        if self.size_of_feasible_region is None:
            self.size_of_feasible_region = 2 * self.epsilon * self.sqdim
        D = torch.cdist(_X, _X)

        cluster_coef = np.zeros((bs,), dtype=np.float64)
        if weights is None:
            compute_cluster_coef_from_distance_matrix_batch(
                bs,
                n,
                self.size_of_feasible_region,
                D.cpu().numpy().astype(np.float32).reshape(-1),
                cluster_coef,
            )
        else:
            cluster_coef_nodeweight(
                bs,
                n,
                self.size_of_feasible_region,
                D.cpu().numpy().astype(np.float32).reshape(-1),
                cluster_coef,
                weights,
            )
        average_coef = torch.from_numpy(cluster_coef)
        return 1.0 - average_coef
