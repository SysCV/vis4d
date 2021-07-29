"""Kalman filter."""
import numpy as np
import scipy.linalg
from detectron2.data import MetadataCatalog

# Table for the 0.95 quantile of the chi-square distribution with N degrees of
# freedom (contains values for N=1, ..., 9). Taken from MATLAB/Octave's chi2inv
# function and used as Mahalanobis gating threshold.

chi2inv95 = {
    1: 3.8415,
    2: 5.9915,
    3: 7.8147,
    4: 9.4877,
    5: 11.070,
    6: 12.592,
    7: 14.067,
    8: 15.507,
    9: 16.919,
}


class KalmanFilter:
    """Kalman filter.

    The 8-dimensional state space

        x, y, a, h, vx, vy, va, vh

    contains the bounding box center position (x, y), aspect ratio a, height h,
    and their respective velocities.
    """

    def __init__(self, dataset_name):
        """Init."""
        ndim, dt = 4, 1.0

        # Create Kalman filter model matrices.
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)

        # Motion and observation uncertainty are chosen relative to the current
        # state estimate, height. These weights control the amount of uncertainty in
        # the model. This is a bit hacky.
        self._std_weight_position = {
            "pedestrian": 1.0 / 20,
            "rider": 1.0 / 20 * 5,
            "car": 1.0 / 20 * 10,
            "truck": 1.0 / 20 * 10,
            "bus": 1.0 / 20 * 10,
            "train": 1.0 / 20 * 10,
            "motorcycle": 1.0 / 20 * 10,
            "bicycle": 1.0 / 20 * 5,
        }
        self._std_weight_velocity = {
            "pedestrian": 1.0 / 160,
            "rider": 1.0 / 160 * 5,
            "car": 1.0 / 160 * 10,
            "truck": 1.0 / 160 * 10,
            "bus": 1.0 / 160 * 10,
            "train": 1.0 / 160 * 10,
            "motorcycle": 1.0 / 160 * 10,
            "bicycle": 1.0 / 160 * 5,
        }

        # a = MetadataCatalog.get(dataset_name)
        # self.idx2cls_mapping = a.idx_to_class_mapping
        self.idx2cls_mapping = {
            0: "pedestrian",
            1: "rider",
            2: "car",
            3: "truck",
            4: "bus",
            5: "train",
            6: "motorcycle",
            7: "bicycle",
        }

    def initiate(self, measurement, class_id):
        """Create track from unassociated measurement.

        Parameters
        ----------
        measurement : ndarray
            Bounding box coordinates (x, y, a, h) with center position (x, y),
            aspect ratio a, and height h.

        Returns
        -------
        Tuple[ndarray, ndarray]
            Returns the mean vector (8 dimensional) and covariance matrix (8x8
            dimensional) of the new track. Unobserved velocities are
            initialized to 0 mean.
        """
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2
            * self._std_weight_position[self.idx2cls_mapping[class_id]]
            * measurement[3],
            2
            * self._std_weight_position[self.idx2cls_mapping[class_id]]
            * measurement[3],
            1e-2,
            2
            * self._std_weight_position[self.idx2cls_mapping[class_id]]
            * measurement[3],
            10
            * self._std_weight_velocity[self.idx2cls_mapping[class_id]]
            * measurement[3],
            10
            * self._std_weight_velocity[self.idx2cls_mapping[class_id]]
            * measurement[3],
            1e-5,
            10
            * self._std_weight_velocity[self.idx2cls_mapping[class_id]]
            * measurement[3],
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance, class_id):
        """Run Kalman filter prediction step.

        Parameters
        ----------
        mean : ndarray
            The 8 dimensional mean vector of the object state at the previous
            time step.
        covariance : ndarray
            The 8x8 dimensional covariance matrix of the object state at the
            previous time step.

        Returns
        -------
        Tuple[ndarray, ndarray]
            Returns the mean vector and covariance matrix of the predicted
            state. Unobserved velocities are initialized to 0 mean.
        """
        std_pos = [
            self._std_weight_position[self.idx2cls_mapping[class_id]]
            * mean[3],
            self._std_weight_position[self.idx2cls_mapping[class_id]]
            * mean[3],
            1e-2,
            self._std_weight_position[self.idx2cls_mapping[class_id]]
            * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity[self.idx2cls_mapping[class_id]]
            * mean[3],
            self._std_weight_velocity[self.idx2cls_mapping[class_id]]
            * mean[3],
            1e-5,
            self._std_weight_velocity[self.idx2cls_mapping[class_id]]
            * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))

        mean = np.dot(self._motion_mat, mean)
        covariance = (
            np.linalg.multi_dot(
                (self._motion_mat, covariance, self._motion_mat.T)
            )
            + motion_cov
        )

        return mean, covariance

    def project(self, mean, covariance, class_id):
        """Project state distribution to measurement space.

        Parameters
        ----------
        mean : ndarray
            The state's mean vector (8 dimensional array).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).

        Returns
        -------
        Tuple[ndarray, ndarray]
            Returns the projected mean and covariance matrix of the given state
            estimate.

        """
        std = [
            self._std_weight_position[self.idx2cls_mapping[class_id]]
            * mean[3],
            self._std_weight_position[self.idx2cls_mapping[class_id]]
            * mean[3],
            1e-1,
            self._std_weight_position[self.idx2cls_mapping[class_id]]
            * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))

        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot(
            (self._update_mat, covariance, self._update_mat.T)
        )
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement, class_id):
        """Run Kalman filter correction step.

        Parameters
        ----------
        mean : ndarray
            The predicted state's mean vector (8 dimensional).
        covariance : ndarray
            The state's covariance matrix (8x8 dimensional).
        measurement : ndarray
            The 4 dimensional measurement vector (x, y, a, h), where (x, y)
            is the center position, a the aspect ratio, and h the height of the
            bounding box.

        Returns
        -------
        Tuple[ndarray, ndarray]
            Returns the measurement-corrected state distribution.

        """
        projected_mean, projected_cov = self.project(
            mean, covariance, class_id
        )

        chol_factor, lower = scipy.linalg.cho_factor(
            projected_cov, lower=True, check_finite=False
        )
        kalman_gain = scipy.linalg.cho_solve(
            (chol_factor, lower),
            np.dot(covariance, self._update_mat.T).T,
            check_finite=False,
        ).T
        innovation = measurement - projected_mean

        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot(
            (kalman_gain, projected_cov, kalman_gain.T)
        )
        return new_mean, new_covariance

    def gating_distance(
        self,
        mean,
        covariance,
        measurements,
        class_id,
        only_position=False,
    ):
        """Compute gating distance between state distribution and measurements.

        A suitable distance threshold can be obtained from `chi2inv95`. If
        `only_position` is False, the chi-square distribution has 4 degrees of
        freedom, otherwise 2.

        Parameters
        ----------
        mean : ndarray
            Mean vector over the state distribution (8 dimensional).
        covariance : ndarray
            Covariance of the state distribution (8x8 dimensional).
        measurements : ndarray
            An Nx4 dimensional matrix of N measurements, each in
            format (x, y, a, h) where (x, y) is the bounding box center
            position, a the aspect ratio, and h the height.
        only_position : Optional[bool]
            If True, distance computation is done with respect to the bounding
            box center position only.

        Returns
        -------
        ndarray
            Returns an array of length N, where the i-th element contains the
            squared Mahalanobis distance between (mean, covariance) and
            `measurements[i]`.

        """
        mean, covariance = self.project(mean, covariance, class_id)
        if only_position:
            mean, covariance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - mean
        z = scipy.linalg.solve_triangular(
            cholesky_factor,
            d.T,
            lower=True,
            check_finite=False,
            overwrite_b=True,
        )
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
