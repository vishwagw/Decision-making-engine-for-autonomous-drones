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