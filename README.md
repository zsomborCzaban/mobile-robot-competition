# TurtleBot 4 Barrel Detection

ROS 2 Humble package: `barrel_lidar_detector`

It detects barrel candidates without AprilTags:

- `lidar_cluster_detector`: finds object-sized clusters from `/scan`.
- `map_shape_detector`: finds round blobs in `/map` and confirms them with LiDAR.
- `mission_controller`: converts the detected barrel pose into a Nav2 goal.
- `ui_remote`: button UI for starting detection and navigation.

Main target output:

```text
/barrel_confirmed_pose
```

## Raspberry Pi

Run only the robot stack on the Raspberry Pi. The UI and barrel detector package
are meant to run on the computer.

Start TurtleBot bringup, RPLiDAR, localization or SLAM, and Nav2 so these exist:

```text
/scan
/map
map -> odom -> base_link/base_footprint -> rplidar_link
Nav2
```

The Raspberry Pi and computer must use the same ROS domain:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=<same_as_computer>
```

## Computer

Run the UI, detector nodes, mission controller, and RViz on the computer. The
nodes will read `/scan`, `/map`, and TF from the Raspberry Pi over ROS.

Build and source:

```bash
cd ~/Desktop/studies/autonomous_mobile_robots/practical_work/code
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=<same_as_raspberry_pi>
colcon build --packages-select barrel_lidar_detector --symlink-install
source install/setup.bash
```

Start the button UI:

```bash
ros2 run barrel_lidar_detector ui_remote
```

Press buttons in this order:

1. `Start Mission Controller`
2. `Start LiDAR + Map Detection`
3. `Calculate Target`
4. `START NAVIGATION`

The UI starts detector/controller processes on the computer. That is expected:
the Raspberry Pi runs the robot stack, while the computer runs this package.

Manual workflow instead:

```bash
ros2 run barrel_lidar_detector lidar_cluster_detector
ros2 run barrel_lidar_detector map_shape_detector
ros2 run barrel_lidar_detector mission_controller
```

Then call:

```bash
ros2 service call /calculate_target std_srvs/srv/Trigger
ros2 service call /start_navigation std_srvs/srv/Trigger
```

Start RViz on the computer:

```bash
rviz2
```

If the UI is missing Tkinter:

```bash
sudo apt install python3-tk
```

## RViz

Set `Fixed Frame`:

```text
map
```

Add normal robot displays:

- `TF`
- `Map` topic `/map`
- `LaserScan` topic `/scan`

Add barrel displays with `Add -> By topic`:

```text
/barrel_candidate_markers       MarkerArray
/barrel_map_candidate_markers   MarkerArray
/barrel_marker                  Marker
/barrel_map_marker              Marker
/barrel_confirmed_marker        Marker
/barrel_pose                    PoseStamped
/barrel_map_pose                PoseStamped
/barrel_confirmed_pose          PoseStamped
```

Most useful displays:

- `/barrel_candidate_markers`: all LiDAR object candidates.
- `/barrel_map_candidate_markers`: round objects found in the map.
- `/barrel_confirmed_marker`: candidate confirmed by LiDAR and map shape.
- `/barrel_confirmed_pose`: pose used by the mission controller.

## Tuning

Default object size:

```text
0.40 m to 1.00 m
```

Useful parameters:

```bash
ros2 run barrel_lidar_detector lidar_cluster_detector --ros-args \
  -p min_cluster_width:=0.40 \
  -p max_cluster_width:=1.00 \
  -p cluster_gap:=0.15 \
  -p max_range:=3.0
```

```bash
ros2 run barrel_lidar_detector map_shape_detector --ros-args \
  -p min_blob_diameter:=0.40 \
  -p max_blob_diameter:=1.00 \
  -p confirm_distance:=0.75
```
