# Sensor input interface:
# start point of the program:

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