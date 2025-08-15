# sensor input interface:
# start point of the program:
# class for input sensor:
class SensorHub:
    def __init__(self):
        self.stereo_buffer = RingBuffer(capacity=5)
        self.imu_buffer = RingBuffer(capacity=200)  # IMU @ 200Hz
        
    def add_stereo_frame(self, left: np.ndarray, right: np.ndarray, timestamp: float):
        """Store synchronized stereo images"""
        self.stereo_buffer.push((left, right, timestamp))
        
    def add_imu_data(self, gyro: np.ndarray, accel: np.ndarray, timestamp: float):
        """Store IMU measurements"""
        self.imu_buffer.push((gyro, accel, timestamp))

# class for feature extraction and tracking:
class FeatureTracker:
    def __init__(self, config):
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.prev_features = None
        
    def process_frame(self, image: np.ndarray):
        # Detect and describe features
        kp, desc = self.orb.detectAndCompute(image, None)
        
        # Track features from previous frame
        if self.prev_features:
            matches = self.matcher.match(self.prev_features['desc'], desc)
            # Apply RANSAC for geometric consistency
            # ...
        
        self.prev_features = {'kp': kp, 'desc': desc}
        return tracked_features
    
# clas for visual odometry:
class VIOEngine:
    def __init__(self, calib_params):
        self.state = np.zeros(15)  # [pos, rot, vel, bg, ba]
        self.covariance = np.eye(15)
        self.imu_preinteg = IMUPreintegration()
        
    def update_visual(self, features, timestamp):
        """Bundle adjustment with visual constraints"""
        # 1. Match features to existing map points
        # 2. Optimize using g2o or Ceres:
        #    min Σ(||reprojection_error||² + ||imu_error||²)
        
    def update_imu(self, gyro, accel, dt):
        """IMU pre-integration"""
        self.imu_preinteg.integrate(gyro, accel, dt)
        
    def optimize(self):
        """Sliding window optimization (10 frames)"""
        # Fixed-lag smoother using factor graphs

# class for local mapping:
class WorldMapper:
    def __init__(self):
        self.sparse_map = ORBMap()  # Keyframes + 3D points
        self.occupancy_grid = OccupancyGrid3D(res=0.2m, size=50x50x20m)
        
    def update_sparse_map(self, keyframe, points_3d):
        """Add new keyframe to pose graph"""
        self.sparse_map.add_keyframe(keyframe)
        # Loop closure detection using DBoW2
        
    def update_occupancy(self, depth_map, pose):
        """Update 3D occupancy grid from stereo depth"""
        point_cloud = stereo_to_pointcloud(depth_map)
        world_points = transform_points(point_cloud, pose)
        self.occupancy_grid.integrate(world_points)
        
    def get_local_grid(self, center: np.ndarray, radius=10m):
        """Extract local slice for path planning"""
        return self.occupancy_grid.crop(center, radius)

# class for state estimation:
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
    
# output:
class WorldModelAPI:
    def __init__(self):
        self.estimator = StateEstimator()
        self.mapper = WorldMapper()
        
    def get_navigation_state(self):
        """For path planner and controller"""
        return self.estimator.get_state()
    
    def get_obstacle_map(self, local_only=True):
        """For obstacle avoidance"""
        if local_only:
            pos = self.estimator.current_pose[:3]
            return self.mapper.get_local_grid(pos)
        return self.mapper.occupancy_grid
    
    def get_sparse_map(self):
        """For global planning"""
        return self.mapper.sparse_map