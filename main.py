import numpy as np
import cv2
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict
import threading
from scipy.spatial.transform import Rotation as R
from scipy.optimize import least_squares
import queue

# ===================== UTILITY CLASSES =====================

class RingBuffer:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, item):
        self.buffer.append(item)
        
    def get_latest(self, n: int = 1):
        if n == 1:
            return self.buffer[-1] if self.buffer else None
        return list(self.buffer)[-n:] if len(self.buffer) >= n else list(self.buffer)

@dataclass
class Pose:
    position: np.ndarray = None
    orientation: np.ndarray = None  # quaternion [w, x, y, z]
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.zeros(3)
        if self.orientation is None:
            self.orientation = np.array([1.0, 0.0, 0.0, 0.0])  # identity quaternion
    
    def translation(self):
        return self.position
    
    def quaternion(self):
        return self.orientation
    
    def rotation_matrix(self):
        return R.from_quat(self.orientation[[1,2,3,0]]).as_matrix()
    
    def transform_point(self, point: np.ndarray):
        """Transform point from local to world coordinates"""
        return self.rotation_matrix() @ point + self.position

# ===================== IMU PROCESSING =====================

class IMUPreintegration:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.delta_t = 0.0
        self.delta_p = np.zeros(3)
        self.delta_v = np.zeros(3)
        self.delta_R = np.eye(3)
        self.bias_gyro = np.zeros(3)
        self.bias_accel = np.zeros(3)
        
    def integrate(self, gyro: np.ndarray, accel: np.ndarray, dt: float):
        """Pre-integrate IMU measurements"""
        # Remove bias
        gyro_corrected = gyro - self.bias_gyro
        accel_corrected = accel - self.bias_accel
        
        # Rotation integration (using small angle approximation)
        omega_dt = gyro_corrected * dt
        omega_norm = np.linalg.norm(omega_dt)
        
        if omega_norm > 1e-8:
            axis = omega_dt / omega_norm
            angle = omega_norm
            delta_R_dt = R.from_rotvec(axis * angle).as_matrix()
        else:
            delta_R_dt = np.eye(3)
            
        # Update integrated values
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ accel_corrected * dt**2
        self.delta_v += self.delta_R @ accel_corrected * dt
        self.delta_R = self.delta_R @ delta_R_dt
        self.delta_t += dt

# ===================== OCCUPANCY GRID =====================

class OccupancyGrid3D:
    def __init__(self, resolution: float = 0.2, size: Tuple[int, int, int] = (250, 250, 100)):
        self.resolution = resolution
        self.size = size
        self.grid = np.zeros(size, dtype=np.float32)  # log-odds
        self.origin = np.array([size[0]//2, size[1]//2, 0]) * resolution
        
    def world_to_grid(self, world_points: np.ndarray):
        """Convert world coordinates to grid indices"""
        grid_points = (world_points + self.origin.reshape(1, -1)) / self.resolution
        return grid_points.astype(int)
    
    def integrate(self, world_points: np.ndarray, occupied_prob: float = 0.7):
        """Update occupancy grid with new observations"""
        grid_indices = self.world_to_grid(world_points)
        
        # Filter valid indices
        valid_mask = ((grid_indices >= 0) & (grid_indices < np.array(self.size))).all(axis=1)
        valid_indices = grid_indices[valid_mask]
        
        if len(valid_indices) > 0:
            # Convert probability to log-odds and update
            log_odds = np.log(occupied_prob / (1 - occupied_prob))
            self.grid[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]] += log_odds
            
    def crop(self, center: np.ndarray, radius: float):
        """Extract local region around center"""
        center_grid = self.world_to_grid(center.reshape(1, -1))[0]
        radius_grid = int(radius / self.resolution)
        
        x_min = max(0, center_grid[0] - radius_grid)
        x_max = min(self.size[0], center_grid[0] + radius_grid)
        y_min = max(0, center_grid[1] - radius_grid)
        y_max = min(self.size[1], center_grid[1] + radius_grid)
        
        return self.grid[x_min:x_max, y_min:y_max, :]

class ORBMap:
    def __init__(self):
        self.keyframes = []
        self.map_points = []
        self.pose_graph = {}
        
    def add_keyframe(self, pose: Pose, features: np.ndarray, descriptors: np.ndarray):
        kf_id = len(self.keyframes)
        keyframe = {
            'id': kf_id,
            'pose': pose,
            'features': features,
            'descriptors': descriptors,
            'timestamp': time.time()
        }
        self.keyframes.append(keyframe)
        return kf_id

# ===================== MAIN CLASSES =====================

class SensorHub:
    def __init__(self):
        self.stereo_buffer = RingBuffer(capacity=5)
        self.imu_buffer = RingBuffer(capacity=200)  # IMU @ 200Hz
        self.running = False
        self.data_lock = threading.Lock()
        
    def add_stereo_frame(self, left: np.ndarray, right: np.ndarray, timestamp: float):
        """Store synchronized stereo images"""
        with self.data_lock:
            self.stereo_buffer.push((left, right, timestamp))
        
    def add_imu_data(self, gyro: np.ndarray, accel: np.ndarray, timestamp: float):
        """Store IMU measurements"""
        with self.data_lock:
            self.imu_buffer.push((gyro, accel, timestamp))
    
    def get_latest_stereo(self):
        with self.data_lock:
            return self.stereo_buffer.get_latest()
    
    def get_latest_imu(self, n: int = 1):
        with self.data_lock:
            return self.imu_buffer.get_latest(n)

class FeatureTracker:
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev_features = None
        self.prev_image = None
        
        # Lucas-Kanade tracker for better tracking
        self.lk_params = dict(winSize=(15, 15),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
    def process_frame(self, image: np.ndarray):
        """Extract and track features"""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect ORB features
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        tracked_features = {
            'keypoints': keypoints,
            'descriptors': descriptors,
            'matches': [],
            'points2d': np.array([kp.pt for kp in keypoints])
        }
        
        # Track features from previous frame
        if self.prev_features is not None and self.prev_image is not None:
            # Use Lucas-Kanade for optical flow
            if len(self.prev_features['points2d']) > 0:
                new_points, status, error = cv2.calcOpticalFlowPyrLK(
                    self.prev_image, gray, 
                    self.prev_features['points2d'].astype(np.float32).reshape(-1, 1, 2),
                    None, **self.lk_params)
                
                # Filter good tracks
                good_tracks = status.ravel() == 1
                if np.any(good_tracks):
                    tracked_features['tracked_points'] = {
                        'prev': self.prev_features['points2d'][good_tracks],
                        'curr': new_points.reshape(-1, 2)[good_tracks]
                    }
        
        # Store for next iteration
        self.prev_features = tracked_features
        self.prev_image = gray.copy()
        
        return tracked_features

class VIOEngine:
    def __init__(self, calib_params: Dict):
        self.calib = calib_params
        
        # State: [position(3), orientation(4), velocity(3), bias_gyro(3), bias_accel(3)]
        self.state = np.zeros(16)
        self.state[3] = 1.0  # Initialize quaternion w component
        
        # Covariance matrix
        self.covariance = np.eye(15) * 0.1  # 15x15 (quaternion has 3 DOF)
        
        self.imu_preinteg = IMUPreintegration()
        self.last_imu_time = None
        self.window_poses = deque(maxlen=10)  # Sliding window
        
        # Camera intrinsics
        self.K = np.array(calib_params.get('K', [[500, 0, 320], [0, 500, 240], [0, 0, 1]]))
        self.baseline = calib_params.get('baseline', 0.12)  # meters
        
    def update_visual(self, features: Dict, timestamp: float):
        """Update state with visual measurements"""
        if 'tracked_points' not in features:
            return
        
        prev_pts = features['tracked_points']['prev']
        curr_pts = features['tracked_points']['curr']
        
        if len(prev_pts) < 8:  # Need minimum points for essential matrix
            return
            
        # Estimate motion using essential matrix
        E, mask = cv2.findEssentialMat(prev_pts, curr_pts, self.K, 
                                       method=cv2.RANSAC, prob=0.999, threshold=1.0)
        
        if E is not None:
            # Recover pose
            _, R_est, t_est, _ = cv2.recoverPose(E, prev_pts, curr_pts, self.K, mask=mask)
            
            # Convert to state update
            if np.linalg.det(R_est) > 0:  # Valid rotation
                # Simple fusion - in production, use proper bundle adjustment
                delta_R = R.from_matrix(R_est)
                current_R = R.from_quat(self.state[3:7][[1,2,3,0]])
                new_R = current_R * delta_R
                
                self.state[3:7] = new_R.as_quat()[[3,0,1,2]]  # Convert back to [w,x,y,z]
                self.state[0:3] += current_R.as_matrix() @ (t_est.flatten() * 0.1)  # Scale factor
        
    def update_imu(self, gyro: np.ndarray, accel: np.ndarray, timestamp: float):
        """Update state with IMU measurements"""
        if self.last_imu_time is None:
            self.last_imu_time = timestamp
            return
            
        dt = timestamp - self.last_imu_time
        if dt <= 0 or dt > 0.1:  # Skip invalid timestamps
            self.last_imu_time = timestamp
            return
        
        # Remove gravity from accelerometer (simplified)
        gravity = np.array([0, 0, -9.81])
        current_R = R.from_quat(self.state[3:7][[1,2,3,0]])
        accel_world = current_R.as_matrix() @ accel + gravity
        
        # Update state (simplified integration)
        self.state[0:3] += self.state[7:10] * dt + 0.5 * accel_world * dt**2  # position
        self.state[7:10] += accel_world * dt  # velocity
        
        # Update orientation
        omega = gyro - self.state[10:13]  # Remove bias
        if np.linalg.norm(omega) > 1e-8:
            delta_q = R.from_rotvec(omega * dt).as_quat()[[3,0,1,2]]
            current_q = self.state[3:7]
            self.state[3:7] = self._quaternion_multiply(current_q, delta_q)
            self.state[3:7] /= np.linalg.norm(self.state[3:7])  # Normalize
        
        self.last_imu_time = timestamp
        
    def _quaternion_multiply(self, q1, q2):
        """Multiply two quaternions [w, x, y, z]"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2
        ])
        
    def get_current_pose(self):
        return Pose(position=self.state[0:3].copy(), 
                   orientation=self.state[3:7].copy())

class WorldMapper:
    def __init__(self):
        self.sparse_map = ORBMap()
        self.occupancy_grid = OccupancyGrid3D(resolution=0.2, size=(250, 250, 100))
        self.last_keyframe_pose = None
        self.keyframe_threshold = 0.1  # meters
        
    def update_sparse_map(self, pose: Pose, features: Dict):
        """Add new keyframe to pose graph"""
        # Check if we need a new keyframe
        if (self.last_keyframe_pose is None or 
            np.linalg.norm(pose.position - self.last_keyframe_pose.position) > self.keyframe_threshold):
            
            kf_id = self.sparse_map.add_keyframe(pose, features.get('points2d', []), 
                                               features.get('descriptors', []))
            self.last_keyframe_pose = pose
            return kf_id
        return None
        
    def update_occupancy(self, left_image: np.ndarray, right_image: np.ndarray, pose: Pose):
        """Update 3D occupancy grid from stereo depth"""
        # Compute stereo disparity
        stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=21)
        
        # Convert to grayscale if needed
        if len(left_image.shape) == 3:
            left_gray = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        else:
            left_gray = left_image
            right_gray = right_image
            
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Convert disparity to depth
        baseline = 0.12  # meters
        focal_length = 500  # pixels
        depth = (baseline * focal_length) / (disparity + 1e-6)
        
        # Generate point cloud
        h, w = depth.shape
        point_cloud = []
        
        for v in range(0, h, 4):  # Subsample for efficiency
            for u in range(0, w, 4):
                if disparity[v, u] > 0 and depth[v, u] < 10:  # Valid depth
                    x = (u - w/2) * depth[v, u] / focal_length
                    y = (v - h/2) * depth[v, u] / focal_length
                    z = depth[v, u]
                    point_cloud.append([x, y, z])
        
        if point_cloud:
            # Transform to world coordinates
            point_cloud = np.array(point_cloud)
            world_points = np.array([pose.transform_point(pt) for pt in point_cloud])
            
            # Update occupancy grid
            self.occupancy_grid.integrate(world_points)
        
    def get_local_grid(self, center: np.ndarray, radius: float = 10.0):
        """Extract local slice for path planning"""
        return self.occupancy_grid.crop(center, radius)

class StateEstimator:
    def __init__(self):
        self.current_pose = Pose()
        self.velocity = np.zeros(3)
        self.covariance = np.eye(6) * 0.1
        self.health_status = {"vision": 1.0, "imu": 1.0}
        self.last_update_time = time.time()
        
    def update(self, vio_pose: Pose, imu_velocity: np.ndarray, timestamp: float):
        """Sensor fusion with simple complementary filter"""
        dt = timestamp - self.last_update_time
        
        if dt > 0:
            # Simple fusion - weight based on health status
            vision_weight = self.health_status["vision"]
            imu_weight = self.health_status["imu"]
            
            # Update pose (weighted average)
            if vision_weight > 0.5:
                self.current_pose.position = (vision_weight * vio_pose.position + 
                                            (1-vision_weight) * self.current_pose.position)
                self.current_pose.orientation = vio_pose.orientation  # Use VIO orientation
            
            # Update velocity
            if imu_weight > 0.5:
                alpha = 0.9  # Complementary filter coefficient
                self.velocity = alpha * self.velocity + (1-alpha) * imu_velocity
        
        self.last_update_time = timestamp
        
    def get_state(self):
        return {
            "timestamp": time.time(),
            "position": self.current_pose.translation(),
            "orientation": self.current_pose.quaternion(),
            "velocity": self.velocity,
            "covariance": self.covariance,
            "health": self.health_status
        }

class WorldModelAPI:
    def __init__(self, calib_params: Dict = None):
        # Default calibration parameters
        default_calib = {
            'K': [[500, 0, 320], [0, 500, 240], [0, 0, 1]],
            'baseline': 0.12,
            'imu_noise': {'gyro': 0.01, 'accel': 0.1}
        }
        self.calib = calib_params or default_calib
        
        # Initialize components
        self.sensor_hub = SensorHub()
        self.feature_tracker = FeatureTracker()
        self.vio_engine = VIOEngine(self.calib)
        self.mapper = WorldMapper()
        self.estimator = StateEstimator()
        
        # Processing thread
        self.processing_thread = None
        self.running = False
        
    def start(self):
        """Start the VIO processing pipeline"""
        self.running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        
    def stop(self):
        """Stop the VIO processing pipeline"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
            
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                # Process latest stereo frame
                stereo_data = self.sensor_hub.get_latest_stereo()
                if stereo_data:
                    left, right, timestamp = stereo_data
                    
                    # Extract and track features
                    features = self.feature_tracker.process_frame(left)
                    
                    # Update VIO with visual measurements
                    self.vio_engine.update_visual(features, timestamp)
                    
                    # Update mapping
                    current_pose = self.vio_engine.get_current_pose()
                    self.mapper.update_sparse_map(current_pose, features)
                    self.mapper.update_occupancy(left, right, current_pose)
                    
                    # Update state estimator
                    self.estimator.update(current_pose, self.vio_engine.state[7:10], timestamp)
                
                # Process IMU data
                imu_data = self.sensor_hub.get_latest_imu(10)  # Process last 10 IMU samples
                for gyro, accel, timestamp in imu_data:
                    self.vio_engine.update_imu(gyro, accel, timestamp)
                
                time.sleep(0.01)  # 100Hz processing rate
                
            except Exception as e:
                print(f"Processing error: {e}")
                time.sleep(0.1)
    
    def add_stereo_frame(self, left: np.ndarray, right: np.ndarray):
        """Add stereo camera frame"""
        timestamp = time.time()
        self.sensor_hub.add_stereo_frame(left, right, timestamp)
        
    def add_imu_data(self, gyro: np.ndarray, accel: np.ndarray):
        """Add IMU measurement"""
        timestamp = time.time()
        self.sensor_hub.add_imu_data(gyro, accel, timestamp)
        
    def get_navigation_state(self):
        """For path planner and controller"""
        return self.estimator.get_state()
    
    def get_obstacle_map(self, local_only: bool = True):
        """For obstacle avoidance"""
        if local_only:
            state = self.estimator.get_state()
            pos = state["position"]
            return self.mapper.get_local_grid(pos)
        return self.mapper.occupancy_grid.grid
    
    def get_sparse_map(self):
        """For global planning"""
        return self.mapper.sparse_map
    
    def get_system_health(self):
        """Get system health status"""
        return {
            "vio_initialized": len(self.mapper.sparse_map.keyframes) > 0,
            "feature_count": len(self.feature_tracker.prev_features.get('keypoints', [])) if self.feature_tracker.prev_features else 0,
            "processing_rate": "~100Hz",
            "health_status": self.estimator.health_status
        }

# ===================== DEMO/TEST CODE =====================

def create_demo_system():
    """Create a demo VIO system for testing"""
    # Camera calibration parameters (example)
    calib_params = {
        'K': [[500, 0, 320], [0, 500, 240], [0, 0, 1]],  # Camera intrinsics
        'baseline': 0.12,  # Stereo baseline in meters
        'imu_noise': {'gyro': 0.01, 'accel': 0.1}
    }
    
    # Create VIO system
    vio_system = WorldModelAPI(calib_params)
    
    return vio_system

def simulate_sensor_data():
    """Generate simulated sensor data for testing"""
    # Create dummy stereo images (640x480 grayscale)
    left_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    right_img = np.random.randint(0, 255, (480, 640), dtype=np.uint8)
    
    # Add some features (circles) to both images
    for i in range(10):
        center = (np.random.randint(50, 590), np.random.randint(50, 430))
        cv2.circle(left_img, center, 20, 255, -1)
        # Simulate disparity by shifting right image
        right_center = (max(10, center[0] - np.random.randint(5, 15)), center[1])
        cv2.circle(right_img, right_center, 20, 255, -1)
    
    # Simulate IMU data
    gyro = np.random.normal(0, 0.01, 3)  # rad/s
    accel = np.array([0, 0, -9.81]) + np.random.normal(0, 0.1, 3)  # m/s^2
    
    return left_img, right_img, gyro, accel

if __name__ == "__main__":
    # Demo usage
    print("Creating VIO system...")
    vio = create_demo_system()
    
    print("Starting VIO processing...")
    vio.start()
    
    try:
        # Simulate sensor data for 10 seconds
        for i in range(1000):  # 10 seconds at 100Hz
            left, right, gyro, accel = simulate_sensor_data()
            
            # Add sensor data
            if i % 3 == 0:  # Add stereo frames at ~30Hz
                vio.add_stereo_frame(left, right)
            
            vio.add_imu_data(gyro, accel)  # Add IMU at 100Hz
            
            # Print status every 100 iterations
            if i % 100 == 0:
                nav_state = vio.get_navigation_state()
                health = vio.get_system_health()
                print(f"Step {i}: Position: {nav_state['position']}, Features: {health['feature_count']}")
            
            time.sleep(0.01)  # 100Hz simulation
            
    except KeyboardInterrupt:
        print("Stopping...")
    
    finally:
        vio.stop()
        print("VIO system stopped.")