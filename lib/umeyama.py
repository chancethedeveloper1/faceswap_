## License (Modified BSD)
## Copyright (C) 2011, the scikit-image team All rights reserved.
##
## Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
##
## Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
## Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
## Neither the name of skimage nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
## THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# umeyama function from scikit-image/skimage/transform/_geometric.py

import numpy as np
from builtins import super

MEAN_FACE_X = np.array([
    0.000213256, 0.0752622, 0.18113, 0.29077, 0.393397, 0.586856, 0.689483,
    0.799124, 0.904991, 0.98004, 0.490127, 0.490127, 0.490127, 0.490127,
    0.36688, 0.426036, 0.490127, 0.554217, 0.613373, 0.121737, 0.187122,
    0.265825, 0.334606, 0.260918, 0.182743, 0.645647, 0.714428, 0.793132,
    0.858516, 0.79751, 0.719335, 0.254149, 0.340985, 0.428858, 0.490127,
    .551395, 0.639268, 0.726104, 0.642159, 0.556721, 0.490127, 0.423532,
    0.338094, 0.290379, 0.428096, 0.490127, 0.552157, 0.689874, 0.553364,
    0.490127, 0.42689])

MEAN_FACE_Y = np.array([
    0.106454, 0.038915, 0.0187482, 0.0344891, 0.0773906, 0.0773906, 0.0344891,
    0.0187482, 0.038915, 0.106454, 0.203352, 0.307009, 0.409805, 0.515625,
    0.587326, 0.609345, 0.628106, 0.609345, 0.587326, 0.216423, 0.178758,
    0.179852, 0.231733, 0.245099, 0.244077, 0.231733, 0.179852, 0.178758,
    0.216423, 0.244077, 0.245099, 0.780233, 0.745405, 0.727388, 0.742578,
    0.727388, 0.745405, 0.780233, 0.864805, 0.902192, 0.909281, 0.902192,
    0.864805, 0.784792, 0.778746, 0.785343, 0.778746, 0.784792, 0.824182,
    0.831803, 0.824182])

def umeyama(src, use_scale, dst=None):
    """Estimate N-D similarity transformation with or without scaling.
    Parameters
    ----------
    src : (M, N) array
        Source coordinates.
    dst : (M, N) array
        Destination coordinates.
    use_scale : bool
        Whether to estimate scaling factor.
    Returns
    -------
    T : (N + 1, N + 1)
        The homogeneous similarity transformation matrix. The matrix contains
        NaN values only if the problem is not well-conditioned.
    References
    ----------
    .. [1] "Least-squares estimation of transformation parameters between two
            point patterns", Shinji Umeyama, PAMI 1991, DOI: 10.1109/34.88573
    """

    if dst is None:
        dst = np.stack((MEAN_FACE_X, MEAN_FACE_Y), axis=-1)

    num = src.shape[0]
    dim = src.shape[1]

    # Compute mean of src and dst.
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    # Subtract mean from src and dst.
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    # Eq. (38).
    A = (dst_demean.T @ src_demean) / num
    U, S, V = np.linalg.svd(A)

    # Eq. (39).
    d = np.ones((dim,), dtype='float32')
    d[dim - 1] = -1. if np.linalg.det(A) < 0. else 1.
    T = np.eye(dim + 1, dtype='float32')

    # Eq. (41).
    scale = 1. / np.sum(np.var(src_demean, axis=0)) * (S @ d) if use_scale else 1.

    # Eq. (40) and (43).
    rank = np.linalg.matrix_rank(A)

    if rank == 0:
        return np.nan * T
    elif rank == dim - 1:
        if np.linalg.det(U) * np.linalg.det(V) > 0:
            second_term = V
        else:
            d[dim - 1] = -1
            second_term = np.diag(d) @ V
    else:
        second_term = np.diag(d) @ V.T
    T[:dim, :dim] = U @ second_term

    # Eq. (42)
    T[:dim, dim] = (dst_mean - scale * (T[:dim, :dim] @ src_mean.T))
    T[:dim, :dim] *= scale

    return T

class expectation_maximization_registration(object):
    def __init__(self, X, Y, sigma2=None, max_iterations=100, tolerance=0.001, w=0, *args, **kwargs):
        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("The target point cloud (X) must be at a 2D numpy array.")
        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("The source point cloud (Y) must be a 2D numpy array.")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Both point clouds need to have the same number of dimensions.")

        self.X = X
        self.Y = Y
        self.sigma2 = sigma2
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.w = w / (1 - w) * (self.M / self.N)
        self.err = self.tolerance + 1.

    def register(self, callback=lambda **kwargs: None):
        P = np.zeros((self.M, self.N))
        Px = np.zeros((self.N, ))
        Py = np.zeros((self.M, ))
        epsilon = np.finfo(float).eps
        self.transform_Y()
        if self.sigma2 is None:
            err = np.square(self.X[None, :, :] - self.TY[:, None, :])
            self.sigma2 = 2. * np.sum(err) / (self.D * self.M * self.N)
        self.q = -self.err - self.N * self.D * 0.5 * np.log(self.sigma2 * 0.5)

        for i in range(self.max_iterations):
            P, Px, Py = self.expectation(P, Px, Py, epsilon)
            self.update_transform(P, Px, Py)
            self.transform_Y()
            self.update_variance()
            if callable(callback):
                kwargs = {'iteration': i, 'error': self.err, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)
            if self.err <= self.tolerance:
                break
        return self.TY, self.get_registration_parameters()

    def get_registration_parameters(self):
        raise NotImplementedError("Registration parameters should be defined in child classes.")

    def expectation(self):
        numer = self.X[:, None, :] - self.TY        # N by M by D
        numer = np.sum(P**2, axis=2)                # N by M
        numer = np.exp(-numer / (self.sigma2))
        
        denom = np.sum(P, axis=1)
        denom += self.w *(np.pi * self.sigma2) ** (self.D / 2.)
        denom[denom==0] = self.epsilon

        P = numer / denom                           # N by M
        Py = np.sum(P, axis=0, keepdims=True)       # 1 by M
        Px = np.sum(P, axis=1, keepdims=True).T    # 1 by N
        return P, Px, Py

class rigid_registration(expectation_maximization_registration):
    def __init__(self, R=None, t=None, s=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.D != 2 and self.D != 3:
            message = 'Rigid registration only supports 2D or 3D point clouds. Instead got {}.'.format(self.D)
            raise ValueError(message)
        if s == 0:
            raise ValueError('A zero scale factor is not supported.')
        self.R = np.eye(self.D).T if R is None else R.T
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t
        self.s = 1. if s is None else s

    def update_transform(self, P, Px, Py):
        Np = np.sum(Px)
        muX_T = (self.X @ Px) / self.Np
        muY_T = (self.Y @ Py) / self.Np
        self.XX = self.X - muX_T[:, None]
        YY = self.Y - muY_T[:, None]

        self.A = self.XX.T @ Px.T @ YY
        U, SS, V = np.linalg.svd(self.A, full_matrices=True)
        C[self.D-1] = np.linalg.det(U @ V.T)

        self.R = U @ np.diag(C) @ V.T
        self.YPY = self.Py @ np.sum(YY**2, axis=1)
        self.scale = np.trace(self.A.T @ self.R) / self.YPY
        self.t = muX_T.T - self.scale @ self.R @ muY_T.T

    def transform_Y(self):
        self.TY = self.scale * self.Y @ self.R.T + self.t

    def update_variance(self):
        qprev = self.q
        trAR = np.trace(self.A @ self.R.T)
        xPx = self.Pt1.T @ np.sum(self.XX**2, axis=1)
        self.q = (xPx - 2. * self.s * trAR + self.s ** 2 * self.YPY) / (self.sigma2) + self.D * self.Np/2. * np.log(self.sigma2 * 0.5)
        self.err = np.abs(self.q - qprev)
        if self.sigma2 > 0.:
            self.sigma2 = 2. * (xPx - self.s * trAR) / (self.Np * self.D)
        else:
            self.sigma2 = self.tolerance / 5.

    def get_registration_parameters(self):
        return self.s, self.R.T, self.t

class affine_registration(expectation_maximization_registration):
    def __init__(self, B=None, t=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.B = np.eye(self.D) if B is None else B
        self.t = np.atleast_2d(np.zeros((1, self.D))) if t is None else t

    def update_transform(self):
        muX = np.divide(np.sum(np.dot(self.P, self.X), axis=0), self.Np)
        muY = np.divide(np.sum(np.dot(np.transpose(self.P), self.Y), axis=0), self.Np)

        self.XX = self.X - np.tile(muX, (self.N, 1))
        YY = self.Y - np.tile(muY, (self.M, 1))

        self.A = np.dot(np.transpose(self.XX), np.transpose(self.P))
        self.A = np.dot(self.A, YY)

        self.YPY = np.dot(np.transpose(YY), np.diag(self.P1))
        self.YPY = np.dot(self.YPY, YY)

        self.B = np.linalg.solve(np.transpose(self.YPY), np.transpose(self.A))
        self.t = np.transpose(muX) - np.dot(np.transpose(self.B), np.transpose(muY))

    def transform_point_cloud(self):
        self.TY = np.dot(self.Y, self.B) + np.tile(self.t, (self.M, 1))

    def update_variance(self):
        qprev = self.q
        trAB = np.trace(np.dot(self.A, self.B))
        xPx = np.dot(self.Pt1.T, np.sum(np.multiply(self.XX, self.XX), axis =1))
        trBYPYP = np.trace(np.dot(np.dot(self.B, self.YPY), self.B))
        self.q = (xPx - 2 * trAB + trBYPYP) / (self.sigma2) + self.D * self.Np/2 * np.log(self.sigma2 * 0.5)
        self.err = np.abs(self.q - qprev)
        if self.sigma2 > 0.:
            self.sigma2 = 2. * (xPx - trAB) / (self.Np * self.D)
        else:
            self.sigma2 = self.tolerance / 5.

    def get_registration_parameters(self):
        return self.B, self.t