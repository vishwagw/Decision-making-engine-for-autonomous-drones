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