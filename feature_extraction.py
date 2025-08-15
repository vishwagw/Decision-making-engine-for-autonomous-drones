# second part of the model:
# feature extraction and tracking
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