import numpy as np

class UKFStateEstimator:
    """
    UKF για διαφορικό / unicycle ρομπότ με state:
    x = [x, y, theta, v, omega]^T

    - predict(): prediction βήμα με motion model
    - update_vo(z): VO μέτρηση [x, y, theta]
    - update_imu(z): IMU μέτρηση [theta, omega]
    - update_odom(z): odom μέτρηση [v, omega]
    """

    def __init__(self, dt,
                 process_noise_std,  # dict: pos, theta, v, omega
                 vo_noise_std,       # dict: x, y, theta
                 imu_noise_std,      # dict: theta, omega
                 odom_noise_std):    # dict: v, omega
        self.dt = dt

        # State: [x, y, theta, v, omega]^T
        self.n = 5
        self.x = np.zeros((self.n, 1))   # αρχική κατάσταση
        self.P = np.eye(self.n) * 1e-3   # μικρή αρχική διασπορά

        # Process noise covariance Q
        q_pos   = process_noise_std["pos"]**2
        q_theta = process_noise_std["theta"]**2
        q_v     = process_noise_std["v"]**2
        q_omega = process_noise_std["omega"]**2
        self.Q = np.diag([q_pos, q_pos, q_theta, q_v, q_omega])

        # Measurement noise matrices
        self.R_vo = np.diag([
            vo_noise_std["x"]**2,
            vo_noise_std["y"]**2,
            vo_noise_std["theta"]**2,
        ])

        self.R_imu = np.diag([
            imu_noise_std["theta"]**2,
            imu_noise_std["omega"]**2,
        ])

        self.R_odom = np.diag([
            odom_noise_std["v"]**2,
            odom_noise_std["omega"]**2,
        ])

        # UKF parameters (scaled unscented transform)
        self.alpha = 1e-3
        self.beta = 2.0
        self.kappa = 0

        self.lam = self.alpha**2 * (self.n + self.kappa) - self.n

        # Weights for sigma points
        self.Wm = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wc = np.full(2 * self.n + 1, 1.0 / (2 * (self.n + self.lam)))
        self.Wm[0] = self.lam / (self.n + self.lam)
        self.Wc[0] = self.lam / (self.n + self.lam) + (1 - self.alpha**2 + self.beta)

    # ----------------- Μοντέλο κίνησης -----------------

    def _motion_model(self, x, dt):
        """
        Unicycle model:
        x_{k+1} = x_k + v cos(theta) dt
        y_{k+1} = y_k + v sin(theta) dt
        theta_{k+1} = theta_k + omega dt
        v, omega constant
        """
        x_pos, y_pos, theta, v, omega = x.flatten()

        x_pos_new = x_pos + v * np.cos(theta) * dt
        y_pos_new = y_pos + v * np.sin(theta) * dt
        theta_new = theta + omega * dt

        # wrap angle στο [-pi, pi]
        theta_new = (theta_new + np.pi) % (2 * np.pi) - np.pi

        return np.array([[x_pos_new],
                         [y_pos_new],
                         [theta_new],
                         [v],
                         [omega]])

    # --------------- Sigma points generation ---------------

    def _sigma_points(self, x, P):
        n = self.n
        lam = self.lam
        S = np.linalg.cholesky((n + lam) * P)

        sigma_points = np.zeros((2 * n + 1, n))
        sigma_points[0] = x.flatten()
        for i in range(n):
            sigma_points[i + 1]     = x.flatten() + S[:, i]
            sigma_points[i + 1 + n] = x.flatten() - S[:, i]

        return sigma_points

    # ----------------- Prediction step -----------------

    def predict(self):
        sigma_points = self._sigma_points(self.x, self.P)

        # προώθηση sigma points μέσα από το motion model
        sigma_points_pred = np.zeros_like(sigma_points)
        for i, sp in enumerate(sigma_points):
            sigma_points_pred[i] = self._motion_model(sp.reshape(-1, 1), self.dt).flatten()

        # predicted mean
        x_pred = np.sum(self.Wm.reshape(-1, 1) * sigma_points_pred, axis=0).reshape(-1, 1)

        # predicted covariance
        P_pred = np.zeros((self.n, self.n))
        for i, sp in enumerate(sigma_points_pred):
            dx = (sp.reshape(-1, 1) - x_pred)
            # wrap για theta
            dx[2, 0] = (dx[2, 0] + np.pi) % (2 * np.pi) - np.pi
            P_pred += self.Wc[i] * (dx @ dx.T)

        P_pred += self.Q

        self.x = x_pred
        self.P = P_pred

    # ----------------- Generic update -----------------

    def _update_generic(self, z, h_func, R, m):
        """
        Γενικό update για οποιοδήποτε αισθητήρα.
        z: measurement (m x 1)
        h_func: h(x) measurement model
        R: measurement covariance (m x m)
        m: dimension μέτρησης
        """
        sigma_points = self._sigma_points(self.x, self.P)

        # prediction στο measurement space
        Z_sigma = np.zeros((2 * self.n + 1, m))
        for i, sp in enumerate(sigma_points):
            Z_sigma[i] = h_func(sp.reshape(-1, 1)).flatten()

        # predicted measurement mean
        z_pred = np.sum(self.Wm.reshape(-1, 1) * Z_sigma, axis=0).reshape(-1, 1)

        # innovation covariance
        S = np.zeros((m, m))
        for i, zp in enumerate(Z_sigma):
            dz = zp.reshape(-1, 1) - z_pred
            S += self.Wc[i] * (dz @ dz.T)
        S += R

        # cross-covariance
        Pxz = np.zeros((self.n, m))
        for i, sp in enumerate(sigma_points):
            dx = sp.reshape(-1, 1) - self.x
            dz = Z_sigma[i].reshape(-1, 1) - z_pred
            Pxz += self.Wc[i] * (dx @ dz.T)

        # Kalman gain
        K = Pxz @ np.linalg.inv(S)

        # update
        y = z - z_pred
        self.x = self.x + K @ y
        self.P = self.P - K @ S @ K.T

    # ----------------- Ειδικά updates ανά αισθητήρα -----------------

    def update_vo(self, z_vo):
        """
        VO μέτρηση: z_vo = [x, y, theta]
        """
        def h_vo(x):
            return np.array([[x[0, 0]],
                             [x[1, 0]],
                             [x[2, 0]]])
        self._update_generic(z_vo.reshape(3, 1), h_vo, self.R_vo, 3)

    def update_imu(self, z_imu):
        """
        IMU μέτρηση: z_imu = [theta, omega]
        """
        def h_imu(x):
            return np.array([[x[2, 0]],
                             [x[4, 0]]])
        self._update_generic(z_imu.reshape(2, 1), h_imu, self.R_imu, 2)

    def update_odom(self, z_odom):
        """
        Odometry μέτρηση: z_odom = [v, omega]
        """
        def h_odom(x):
            return np.array([[x[3, 0]],
                             [x[4, 0]]])
        self._update_generic(z_odom.reshape(2, 1), h_odom, self.R_odom, 2)
