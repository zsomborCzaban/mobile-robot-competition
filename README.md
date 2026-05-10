# TurtleBot 4 Barrel Detection

This package detects barrel candidates without AprilTags or fiducial markers. The
main localization signal is LiDAR. The camera can still be used separately for
visual confirmation or documentation.

The package name is `barrel_lidar_detector`.

## What It Runs

The system has four executable nodes:

- `lidar_cluster_detector`: detects compact object-sized clusters from `/scan`.
- `map_shape_detector`: searches `/map` for compact round occupied blobs and
  fuses them with the live LiDAR result.
- `mission_controller`: converts the detected barrel pose into a Nav2 approach
  goal and sends it to Nav2.
- `ui_remote`: Tkinter button UI for starting detection and triggering mission
  actions.

## Detection Logic

The LiDAR node subscribes to `/scan` with sensor-data QoS. It converts valid
laser ranges into 2D points in the scan frame, groups nearby points into
Euclidean clusters, filters by physical width, and selects the nearest valid
object-sized cluster.

Default LiDAR cluster width:

```text
min_cluster_width = 0.40 m
max_cluster_width = 1.00 m
```

The selected LiDAR candidate is transformed into `map` and published as
`/barrel_pose`.

The map-shape node subscribes to `/map`, groups connected occupied cells, and
keeps blobs that are approximately round and within the same 0.40 m to 1.00 m
diameter range. It publishes the best map-only result as `/barrel_map_pose`.

If the live LiDAR pose is close to a round map blob, the node publishes a fused
result:

```text
/barrel_confirmed_pose
/barrel_confirmed_marker
```

This is still a candidate detector, not semantic object recognition. A chair,
post, bin, or pillar can look similar to a barrel in 2D LiDAR. The combined
method improves confidence by requiring both a live object-sized LiDAR cluster
and a round-looking map shape.

## Build

From the workspace root that contains `mobile-robot-competition`:

```bash
source /opt/ros/humble/setup.bash
colcon build --packages-select barrel_lidar_detector --symlink-install
source install/setup.bash
```

If you are inside this repository directly, `colcon build` also works because
the package is at the repository root.

## Run Manually

Start the normal TurtleBot 4 stack first: robot bringup, RPLiDAR, TF,
localization or SLAM, `/map`, and Nav2.

Then run the detector nodes:

```bash
ros2 run barrel_lidar_detector lidar_cluster_detector
ros2 run barrel_lidar_detector map_shape_detector
```

To calculate a Nav2 approach pose and start navigation:

```bash
ros2 run barrel_lidar_detector mission_controller
ros2 service call /calculate_target std_srvs/srv/Trigger
ros2 service call /start_navigation std_srvs/srv/Trigger
```

The mission controller prefers target sources in this order:

```text
/barrel_confirmed_pose
/barrel_pose
/barrel_map_pose
```

It places the navigation goal about `0.60 m` away from the barrel, on the side
closest to the robot, and orients the robot toward the barrel.

## Run With Buttons

Run the UI:

```bash
ros2 run barrel_lidar_detector ui_remote
```

Use the buttons in this order:

1. `Start Mission Controller`
2. `Start LiDAR + Map Detection`
3. `Calculate Target`
4. `START NAVIGATION`

The UI starts the detector processes for convenience. Closing the UI stops only
the child processes that were started by that UI instance.

If Tkinter is missing on the Raspberry Pi, install it:

```bash
sudo apt install python3-tk
```

## RViz Setup

Set RViz `Fixed Frame` to:

```text
map
```

Add the normal robot displays first:

- `TF`
- `Map` using topic `/map`
- `LaserScan` using topic `/scan`
- Nav2 displays if you are using the TurtleBot/Nav2 RViz config

To add the barrel displays:

1. Click `Add` in the Displays panel.
2. Use `By topic`.
3. Add these topics:

```text
/barrel_candidate_markers       visualization_msgs/msg/MarkerArray
/barrel_map_candidate_markers   visualization_msgs/msg/MarkerArray
/barrel_marker                  visualization_msgs/msg/Marker
/barrel_map_marker              visualization_msgs/msg/Marker
/barrel_confirmed_marker        visualization_msgs/msg/Marker
/barrel_pose                    geometry_msgs/msg/PoseStamped
/barrel_map_pose                geometry_msgs/msg/PoseStamped
/barrel_confirmed_pose          geometry_msgs/msg/PoseStamped
```

Recommended interpretation:

- `/barrel_candidate_markers`: all live LiDAR object-sized clusters.
- `/barrel_marker`: nearest live LiDAR candidate.
- `/barrel_map_candidate_markers`: round occupied blobs found in the map.
- `/barrel_map_marker`: best map-shape candidate.
- `/barrel_confirmed_marker`: candidate confirmed by both LiDAR and map shape.
- `/barrel_confirmed_pose`: best pose to use for navigation.

If the markers do not appear, check:

- `/scan` is publishing.
- `/map` is publishing.
- TF contains `map -> odom -> base_link/base_footprint -> rplidar_link`.
- RViz fixed frame is `map`.
- The detector terminals are not printing TF transform warnings.

## Useful Parameters

LiDAR detector:

```bash
ros2 run barrel_lidar_detector lidar_cluster_detector --ros-args \
  -p min_cluster_width:=0.40 \
  -p max_cluster_width:=1.00 \
  -p cluster_gap:=0.15 \
  -p max_range:=3.0
```

Map-shape detector:

```bash
ros2 run barrel_lidar_detector map_shape_detector --ros-args \
  -p min_blob_diameter:=0.40 \
  -p max_blob_diameter:=1.00 \
  -p confirm_distance:=0.75
```

Mission controller:

```bash
ros2 run barrel_lidar_detector mission_controller --ros-args \
  -p approach_offset:=0.60 \
  -p base_frame:=base_link
```

Use `base_footprint` instead of `base_link` if that is the frame available in
your TurtleBot TF tree.
