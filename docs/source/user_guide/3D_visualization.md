# Visualizing 3D Data

```python
import pickle
from vis4d.vis.pointcloud.functional import show_points
import numpy as np

# First, lets load the pointcloud data so we can visualize it.
pc_data = pickle.load(open("data/pc_data.pkl", "rb"))

xyz = pc_data["points3d"] - np.mean(pc_data["points3d"], 0)
colors = pc_data["colors3d"]
classes = pc_data["semantics3d"]
instances = pc_data["instances3d"]

# Lets remove the top part of the pointcloud to see it better
lower_mask = xyz[:, -1] < 1.4
xyz = xyz[lower_mask, :]
colors = colors[lower_mask, :]
classes = classes[lower_mask]
instances = instances[lower_mask]

print(f"Loaded pointcloud with {colors.shape[0]} points")

# Visualize the point clound with color
show_points(xyz, colors)

# Lets look at some other predictions. Class and Instances
show_points(xyz, classes = classes)

show_points(xyz, instances = instances)

# Finally, lets show all predictions in one window. 
# This will copy the room three times and color it differently each time, based on 'color', 'classes' and 'instances'
show_points(xyz, colors =colors, classes=classes, instances = instances)
```