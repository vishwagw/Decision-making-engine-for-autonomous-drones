octomap::OcTree tree(0.1);  // 10cm resolution
tree.updateNode(point, true);  // Integrate points
tree.insertRay(sensor_origin, point); // Free space