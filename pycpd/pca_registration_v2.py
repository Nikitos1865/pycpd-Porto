import numbers

import numpy as np

from pycpd.emregistration import EMRegistration


class PCADeformableRegistration2(EMRegistration):
    def __init__(self, X, Y, alpha=None, mean_shape=None, U=None, eigenvalues=None, *args, **kwargs):
        # Initialize EMRegistration first with its parameters
        super().__init__(X=X, Y=Y, *args, **kwargs)

        # Handle PCA-specific parameters
        if alpha is not None and (not isinstance(alpha, numbers.Number) or alpha <= 0):
            raise ValueError(
                "Expected a positive value for regularization parameter alpha. Instead got: {}".format(alpha))

        if mean_shape is None or U is None or eigenvalues is None:
            raise ValueError("mean_shape, U, and eigenvalues must be provided")

        self.alpha = 2 if alpha is None else alpha
        self.mean_shape = mean_shape # shape is N x D
        self.U = U # principal components (N*D x K) i.e. columns represent the eigen vectors.
        self.eigenvalues = eigenvalues # (k length array)
        self.b = np.zeros((len(eigenvalues), 1)) # (k x 1) vector
        self.prev_b = self.b


    def expectation(self): # same as parent - expectation step of EM algorithm
        """
        Compute the expectation step of the EM algorithm.
        """
        # Compute squared Euclidean distances between every pair (X_i, TY_j). Note: X has shape (N, D) and Y has shape (M, D)
        # Note: X[None, :, :] → (1, N, D), Y[:, None, :] → (M, 1, D), (X - Y) then results in (M, N, D).
        # Squaring and summing along axis=2 gives (N, M), which is the pairwise squared distance matrix.
        P = np.sum((self.X[None, :, :] - self.TY[:, None, :])**2, axis=2)
        P = np.exp(-P/(2*self.sigma2)) # numerator

        # Compute normalization factor (accounting for outliers)
        c = (2*np.pi*self.sigma2)**(self.D/2)*self.w/(1. - self.w)*self.M/self.N
        den = np.sum(P, axis = 0, keepdims = True) # (1, N)
        den = np.clip(den, np.finfo(self.X.dtype).eps, None) + c # denominator

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)
        self.PX = np.matmul(self.P, self.X)



    def update_transform(self): # step 1 in Maximization step -> solve for b
        # In SSM-based CPD, instead of solving for W (which is large, M × D),
        # we solve for b (which is much smaller, k × 1, where k is the number of PCA modes):

        dP = np.diag(self.P1)
        # print("dP shape: ", np.shape(dP))
        # print("U shape: ", np.shape(self.U))

        # Expand dP to match full (M*D, M*D) space
        dP_expanded = np.kron(dP, np.eye(self.D))

        B = self.PX.flatten() - np.dot(dP_expanded, self.mean_shape)  # Difference from mean shape

        self.b = np.linalg.solve(np.dot(self.U.T, dP_expanded @ self.U), np.dot(self.U.T, B)) # (U_t dP U)b = Ut B

        # Compute relative change in W
        self.b_diff = np.mean(np.abs(self.b - self.prev_b))
        print(f"Iteration {self.iteration}: b change = {self.b_diff:.6f}")

        self.prev_b = self.b

    def transform_point_cloud(self, Y=None): # use b to get the new TY = Y + Ub
        # print("mean shape: ", self.mean_shape)
        # print("U shape: ", self.U.shape)
        # print("b shape: ", self.b.shape)
        # print("M: ", self.M)
        # print("N: ", self.N)
        # print("D: ", self.D)

        if Y is None:
            self.TY = self.mean_shape.reshape(self.M, self.D) + (self.U @ self.b).reshape(self.M, self.D)
        else:
            Y_transformed = Y.reshape(self.M, self.D) + (self.U @ self.b).reshape(self.M, self.D)
            return Y_transformed

    def update_variance(self): # update variance step of M registration

        qprev = self.sigma2

        xPx = np.dot(self.Pt1.T, np.sum(np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(self.P1.T, np.sum(np.multiply(self.TY, self.TY), axis=1))
        trPXY = np.sum(np.multiply(self.TY, self.PX))

        self.sigma2 = (xPx - 2 * trPXY + yPy) / (self.Np * self.D)
        self.sigma2 = max(self.sigma2, self.tolerance / 10)

        self.diff = np.abs(self.sigma2 - qprev)

        # Update sigma difference
        self.sigma_diff = np.abs(self.sigma2 - qprev)

        print(
            f"Sigma2: {self.sigma2:.6f}, Sigma diff: {self.sigma_diff:.6f}, b diff: {self.b_diff:.6f}, Combined diff: {self.diff:.6f}")

    def get_registration_parameters(self):
        return self.U, self.b