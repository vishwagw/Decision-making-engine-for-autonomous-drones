class StateEstimator:
    def __init__(self):
        self.current_pose = Pose()
        self.velocity = np.zeros(3)
        self.covariance = np.eye(6)
        self.health_status = {"vision": 1.0, "imu": 1.0}
        
    def update(self, vio_pose, imu_velocity, timestamp):
        """Sensor fusion with Kalman filter"""
        # Hybrid approach:
        # - High-frequency: IMU dead-reckoning (200Hz)
        # - Low-frequency: VIO corrections (30Hz)
        # - Covariance-based reliability weighting
        
    def get_state(self):
        return {
            "timestamp": time.time(),
            "position": self.current_pose.translation(),
            "orientation": self.current_pose.quaternion(),
            "velocity": self.velocity,
            "covariance": self.covariance
        }