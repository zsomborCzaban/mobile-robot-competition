# TurtleBot 4 Barrel Detection

ROS 2 Humble package: `barrel_lidar_detector`

It detects barrel candidates without AprilTags:

- `lidar_cluster_detector`: tracks curved LiDAR clusters over multiple scans and
  publishes `/barrel_pose` only after the accumulated points fit a circle better
  than a straight line.
- `map_shape_detector`: finds round blobs in `/map` and confirms them with LiDAR.
- `mission_controller`: reads the barrel YAML and sends all approach goals to Nav2.
- `ui_remote`: button UI for starting detection and navigation.

Main target output:

```text
/barrel_confirmed_pose
~/turtlebot4_ws/barrel_target.yaml
```

`map_shape_detector` updates the YAML file while SLAM is running and the robot
drives around. `mission_controller` reads the same YAML file, computes an A*
visit order through every valid barrel entry, then sends each approach goal to
Nav2.

Route calculation starts from the robot's current TF pose in `map`. If
`map -> odom -> base_link` or `map -> odom -> base_footprint` is unavailable,
the controller refuses to calculate instead of using the map origin as a fake
start point.

## Raspberry Pi

Run only the robot stack on the Raspberry Pi. The UI and barrel detector package
are meant to run on the computer.

Start TurtleBot bringup and RPLiDAR so these exist:

```text
/scan
odom -> base_link/base_footprint -> rplidar_link
```

The Raspberry Pi and computer must use the same ROS domain:

```bash
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=<same_as_computer>
```

## Mapping vs Navigation Launches

`view_robot.launch.py` is only RViz. It lets you see the robot, map, scan, TF,
and markers, but it does not start Nav2. The `mission_controller` sends goals
through `nav2_simple_commander`, so Nav2 must be running before pressing
`START NAVIGATION`.

Use this rule:

- Discovering/mapping barrels: run SLAM plus `view_robot.launch.py`.
- Executing the route: run Nav2 bringup/localization. Keep RViz open if useful.

On TurtleBot 4 Humble, the normal physical-robot Nav2 command is:

```bash
ros2 launch turtlebot4_navigation nav_bringup.launch.py \
  slam:=off localization:=true map:=/full/path/to/map.yaml
```

Some older TurtleBot 4 setups split that into:

```bash
ros2 launch turtlebot4_navigation localization.launch.py map:=/full/path/to/map.yaml
ros2 launch turtlebot4_navigation nav2.launch.py
```

If your install has a `nav.launch.py` file instead, use it only if it is your
Nav2 bringup launch file. Do not replace it with `view_robot.launch.py`;
`view_robot.launch.py` is still just the RViz viewer.

For the navigation phase these must exist:

```text
/scan
/map
map -> odom -> base_link/base_footprint -> rplidar_link
Nav2 action servers
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

Recommended full workflow:

1. On the robot/Raspberry Pi, start the normal TurtleBot 4 robot bringup.
2. On the computer, start SLAM:

   ```bash
   ros2 launch turtlebot4_navigation slam.launch.py
   ```

3. On the computer, start RViz:

   ```bash
   ros2 launch turtlebot4_viz view_robot.launch.py
   ```

4. Start the button UI:

   ```bash
   ros2 run barrel_lidar_detector ui_remote
   ```

5. Set `Expected barrels` in the UI. Use `0` for unlimited, or enter the
   exact number of barrels you expect in the arena.
6. Press `Start Mission Controller`.
7. Press `Start LiDAR + Map Detection`.
8. Drive around during SLAM until the full arena is mapped and the barrels are
   detected. Confirm magenta markers appear and
   `~/turtlebot4_ws/barrel_target.yaml` contains the barrel entries. If
   `Expected barrels` is greater than `0`, only the strongest confirmed
   candidates are kept.
9. Save the map:

   ```bash
   ros2 service call /slam_toolbox/save_map slam_toolbox/srv/SaveMap "name:
     data: 'arena_map'"
   ```

10. Stop SLAM, then start Nav2 with the saved map:

    ```bash
    ros2 launch turtlebot4_navigation nav_bringup.launch.py \
      slam:=off localization:=true map:=/full/path/to/arena_map.yaml
    ```

11. In RViz, use `2D Pose Estimate` if localization needs the initial pose.
12. Press `Calculate Target Path`.
13. Press `START NAVIGATION`.

The expected barrel count is passed to both `map_shape_detector` and
`mission_controller`. The detector ranks confirmed barrels by confirmations,
score, and roundness, then keeps only the strongest N entries. The mission
controller applies the same limit before planning, so stale extra YAML entries
are ignored.

Older quick button order:

1. Set `Expected barrels`.
2. `Start Mission Controller`
3. `Start LiDAR + Map Detection`
4. Drive around during SLAM until the barrels are detected and written to YAML.
5. Start Nav2 with the saved map.
6. `Calculate Target Path`
7. `START NAVIGATION`

The UI also exposes pause, resume, and stop/reset controls for the active
multi-barrel mission.

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
ros2 service call /pause_navigation std_srvs/srv/Trigger
ros2 service call /resume_navigation std_srvs/srv/Trigger
ros2 service call /stop_navigation std_srvs/srv/Trigger
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
/mission_status                 String
```

Most useful displays:

- `/barrel_candidate_markers`: single-scan curved LiDAR candidates.
- `/barrel_marker`: stable multi-scan LiDAR barrel track.
- `/barrel_map_candidate_markers`: round objects found in the map.
- `/barrel_confirmed_marker`: candidate confirmed by LiDAR and map shape.
- `/barrel_confirmed_pose`: stable pose written into the barrel YAML.
- `/mission_status`: current mission text, such as `Going to barrel 1/3`.

## Barrel YAML

Default path:

```text
~/turtlebot4_ws/barrel_target.yaml
```

The detector creates or updates entries like this:

```yaml
barrels:
  - id: Barrel_001
    map_x: 1.234
    map_y: 2.345
    width: 0.45
    surface_x: 1.021
    surface_y: 2.292
    normal_x: -0.9701
    normal_y: -0.2425
    roundness: 0.91
    score: 0.86
    confirmations: 6
    last_seen_unix_sec: 1710000000.0
```

`map_x` and `map_y` are the confirmed barrel center. When enough map boundary
data is available, `surface_x` and `surface_y` are the measured occupied edge
of the barrel on the confirmed side, and `normal_x`/`normal_y` point outward
from the barrel center through that edge. The mission controller uses the
surface fields first, then falls back to center plus `width / 2` for older YAML
files.

Use the same path for both nodes if you override it:

```bash
ros2 run barrel_lidar_detector map_shape_detector --ros-args \
  -p barrel_yaml_path:=/home/ubuntu/turtlebot4_ws/barrel_target.yaml \
  -p expected_barrel_count:=3

ros2 run barrel_lidar_detector mission_controller --ros-args \
  -p barrel_yaml_path:=/home/ubuntu/turtlebot4_ws/barrel_target.yaml \
  -p expected_barrel_count:=3
```

## Tuning

Default object size:

```text
Map blobs: 0.20 m to 1.20 m
LiDAR clusters: 0.20 m to 1.20 m
```

Useful parameters:

```bash
ros2 run barrel_lidar_detector lidar_cluster_detector --ros-args \
  -p min_cluster_width:=0.20 \
  -p max_cluster_width:=1.20 \
  -p require_curved_cluster:=true \
  -p min_cluster_arc_depth:=0.035 \
  -p min_cluster_range_depth:=0.055 \
  -p min_cluster_circle_radius:=0.08 \
  -p max_cluster_circle_radius:=0.80 \
  -p single_scan_min_line_circle_ratio:=1.15 \
  -p reject_straight_segments:=true \
  -p straight_segment_min_length:=0.35 \
  -p track_min_observations:=3 \
  -p track_min_view_bins:=1 \
  -p track_max_circle_rmse:=0.10 \
  -p track_min_line_circle_ratio:=1.10 \
  -p track_min_confidence:=0.50 \
  -p cluster_gap:=0.15 \
  -p max_range:=3.0
```

```bash
ros2 run barrel_lidar_detector map_shape_detector --ros-args \
  -p min_blob_diameter:=0.20 \
  -p max_blob_diameter:=1.20 \
  -p max_corner_fill_ratio:=0.35 \
  -p max_bounding_box_fill_ratio:=0.88 \
  -p allow_spatial_fallback:=false \
  -p marker_max_diameter:=0.45 \
  -p confirm_distance:=0.65 \
  -p stable_confirmations:=4
```

The mission controller default `approach_offset` is `0.15`. With the generated
surface fields, the requested goal is `0.15 m` outside the measured barrel edge.
For older YAML entries without surface fields, it falls back to `width / 2 +
approach_offset` from the barrel center. If Nav2 still refuses to get that
close, reduce the Nav2 costmap inflation radius in your Nav2 config.

These size limits are intentionally broad because the challenge barrel size may
change. The main rejection logic is now:

- a curved single-scan LiDAR candidate,
- a persistent LiDAR track with enough observations,
- accumulated track points fitting a circle better than a straight line,
- map roundness/corner-fill checks,
- repeated map/LiDAR confirmation before YAML writes.

If magenta confirmed markers appear on non-barrels, increase
`stable_confirmations`, `min_confirmed_roundness`, or
`min_confirmed_minor_major_ratio`. If rectangular boxes are marked as barrels,
lower `max_corner_fill_ratio` or `max_bounding_box_fill_ratio`. If flat objects
still appear as green LiDAR candidates, increase `min_cluster_arc_depth` or
`min_cluster_range_depth`, or increase `single_scan_min_line_circle_ratio`.
If a box with rounded corners still appears green, lower
`straight_segment_min_length` slightly. If flat objects reach the yellow/orange
LiDAR track, increase `track_min_observations`, `track_min_view_bins`, or
`track_min_line_circle_ratio`. If real barrels are missed, lower those values
slightly, lower `track_min_confidence`, or increase `confirm_distance`.
