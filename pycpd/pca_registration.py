import numbers

import numpy as np

from pycpd.emregistration import EMRegistration
from pycpd.utility import pca_kernel


class PCADeformableRegistration(EMRegistration):
    def __init__(self, X, Y, alpha=None, mean_shape=None, U=None, eigenvalues=None):
        super().__init__(X=X, Y=Y)
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if mean_shape is None or U is None or eigenvalues is None:
            raise ValueError("mean_shape, U, and eigenvalues must be provided")

        self.alpha = 2 if alpha is None else alpha
        self.mean_shape = mean_shape
        self.U = U
        self.eigenvalues = eigenvalues
        self.W = np.zeros((self.M, self.D))
        self.G = pca_kernel(self.Y, self.mean_shape, self.U, self.eigenvalues)

    def update_transform(self):
        """
        Calculate a new estimate of the PCA-based deformable transformation.
        """
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        B = self.PX - np.dot(np.diag(self.P1), self.Y)
        self.W = np.linalg.solve(A, B)

    def transform_point_cloud(self, Y=None):
        """
        Update a point cloud using the new estimate of the PCA-based deformable transformation.
        """
        if Y is not None:
            G = pca_kernel(X=Y, mean_shape=self.mean_shape, U=self.U,
                           eigenvalues=self.eigenvalues, Y=self.Y)
            return Y + np.dot(G, self.W)
        else:
            self.TY = self.Y + np.dot(self.G, self.W)

    def update_variance(self):
        """
        Update the variance of the mixture model using the new estimate of the transformation.
        """
        qprev = self.sigma2

        # Calculate the squared Mahalanobis distance
        xPx = np.dot(np.transpose(self.Pt1), np.sum(
            np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(np.transpose(self.P1), np.sum(
            np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        # Update the variance
        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)

        # Add regularization term from the PCA model
        # Project W onto the shape space and weight by eigenvalues
        W_flat = self.W.reshape(1, -1)  # Flatten W to match U dimensions
        W_proj = np.dot(W_flat, self.U)  # Project onto shape space
        pca_reg = np.sum(W_proj ** 2 / (self.eigenvalues + 1e-8))  # Weight by inverse eigenvalues
        self.sigma2 += self.alpha * pca_reg / (self.Np * self.D)

        # Ensure numerical stability
        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10

        # Update difference for convergence check
        self.diff = np.abs(self.sigma2 - qprev)

    def get_registration_parameters(self):
        """
        Return the current estimate of the PCA-based deformable transformation parameters.
        """
        return self.G, self.W