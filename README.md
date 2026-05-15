# TurtleBot 4 Barrel Detection

ROS 2 Humble package: `barrel_lidar_detector`

It detects barrel targets from the map, without AprilTags or LiDAR barrel
classification:

- `ground_truth_barrel_detector`: reads `/map`, detects circular barrel marks
  with the same Hough-circle classifier used by the offline marker, writes the
  barrel YAML, and publishes RViz markers.
- `mission_controller`: reads the barrel YAML and sends all approach goals to Nav2.
- `ui_remote`: button UI for starting detection and navigation.

Main target output:

```text
/barrel_ground_truth_markers
/barrel_markers
/barrel_ground_truth_poses
/barrel_poses
~/turtlebot4_ws/barrel_target.yaml
```

`ground_truth_barrel_detector` updates the YAML file whenever `/map` changes.
`mission_controller` reads the same YAML file, computes an A* visit order
through every valid barrel entry, then sends each approach goal to Nav2.

If you already have a saved map, `offline_barrel_marker` can create the same
barrel YAML without running the robot. It reads the map YAML/PGM, uses an
OpenCV Hough-circle classifier to find circular barrel outlines, and can also
accept manually typed map-frame barrel centers.

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
7. Press `Start Map Barrel Detection`.
8. Drive around during SLAM until the full arena is mapped and the barrels are
   detected. Confirm red markers appear and
   `~/turtlebot4_ws/barrel_target.yaml` contains the barrel entries. If
   `Expected barrels` is greater than `0`, only the strongest detected
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

The expected barrel count is passed to both `ground_truth_barrel_detector` and
`mission_controller`. The detector ranks barrels by classifier score and keeps
only the strongest N entries when the count is greater than `0`. The mission
controller applies the same limit before planning.

Older quick button order:

1. Set `Expected barrels`.
2. `Start Mission Controller`
3. `Start Map Barrel Detection`
4. Drive around during SLAM until the barrels are detected and written to YAML.
5. Start Nav2 with the saved map.
6. `Calculate Target Path`
7. `START NAVIGATION`

The UI also exposes pause, resume, fallback, and stop/reset controls for the
active multi-barrel mission. `FALLBACK: TURN + BACK UP + REPLAN` cancels the
current Nav2 goal, turns the robot around, backs up `1.0 m`, recalculates the
remaining barrel route from the new robot pose, and starts navigation again.

The `Barrel detection sensitivity` slider controls the running
`ground_truth_barrel_detector`. `50` is the default setting that matches the
blue-marked example maps. Higher values accept weaker/partial circles; lower
values make the detector more conservative.

The UI starts detector/controller processes on the computer. That is expected:
the Raspberry Pi runs the robot stack, while the computer runs this package.

Manual workflow instead:

```bash
ros2 run barrel_lidar_detector ground_truth_barrel_detector
ros2 run barrel_lidar_detector mission_controller
```

Then call:

```bash
ros2 service call /calculate_target std_srvs/srv/Trigger
ros2 service call /start_navigation std_srvs/srv/Trigger
ros2 service call /pause_navigation std_srvs/srv/Trigger
ros2 service call /resume_navigation std_srvs/srv/Trigger
ros2 service call /fallback_recovery std_srvs/srv/Trigger
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
/barrel_ground_truth_markers    MarkerArray
/barrel_markers                 MarkerArray
/barrel_ground_truth_poses      PoseArray
/barrel_poses                   PoseArray
/barrel_confirmed_pose          PoseStamped
/mission_status                 String
```

Most useful displays:

- `/barrel_markers`: red cylinders and labels for all map-detected barrels.
- `/barrel_ground_truth_markers`: same marker array with the original full name.
- `/barrel_ground_truth_poses`: pose array for all map-detected barrels.
- `/barrel_poses`: same pose array with a shorter name.
- `/barrel_confirmed_pose`: first detected barrel pose for compatibility.
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
ros2 run barrel_lidar_detector ground_truth_barrel_detector --ros-args \
  -p barrel_yaml_path:=/home/ubuntu/turtlebot4_ws/barrel_target.yaml \
  -p expected_barrel_count:=3

ros2 run barrel_lidar_detector mission_controller --ros-args \
  -p barrel_yaml_path:=/home/ubuntu/turtlebot4_ws/barrel_target.yaml \
  -p expected_barrel_count:=3
```

## Offline Barrel Marking

Use this after saving a map if you want to mark barrels without driving the
robot again:

```bash
ros2 run barrel_lidar_detector offline_barrel_marker arena_map.yaml \
  -o ~/turtlebot4_ws/barrel_target.yaml \
  --annotated-image arena_map_barrels.ppm
```

The script keeps the map files unchanged. It writes a compatible
`barrel_target.yaml`; `--annotated-image` is only a color preview image with
red rings at the selected barrel centers. The script automatically keeps every
candidate that passes the Hough-circle, square-corner rejection, border
clearance, confidence, and surrounding free-space filters. It closes/fills
enclosed black outlines before proposing circles, so a gray or white interior
inside a black barrel ring can still be detected. It does not assume a fixed
barrel count or barrel size; use `--count` only when you explicitly want to cap
the final total including manual entries.

To process every saved map YAML in the current directory and create one barrel
YAML per map:

```bash
ros2 run barrel_lidar_detector offline_barrel_marker --all-maps
```

That writes barrel files to `barrel_targets/` and red preview maps to
`marked_maps/`.

If automatic map-shape detection misses a barrel, add known map-frame centers
manually:

```bash
ros2 run barrel_lidar_detector offline_barrel_marker arena_map.yaml \
  -o ~/turtlebot4_ws/barrel_target.yaml \
  --no-auto \
  --barrel 1.25,0.80 \
  --barrel 2.10,-1.35
```

You can mix automatic and manual entries by omitting `--no-auto`. Entries closer
than `--merge-distance` are merged so a manual correction can replace a nearby
automatic detection.

## Tuning

The detector does not require a fixed barrel size. By default
`min_diameter:=0.0` and `max_diameter:=0.0`, so the Hough radius range is
derived from the map image size. Override those only if the map contains many
non-barrel circles.

The offline saved-map script and the online `/map` detector use the same
automatic detection defaults from `offline_barrel_marker.py`. Tune a parameter
in one path with the same value in the other path when comparing results.

Useful detector parameters:

```bash
ros2 run barrel_lidar_detector ground_truth_barrel_detector --ros-args \
  -p detection_sensitivity:=50 \
  -p occupied_threshold:=50 \
  -p min_radius_pixels:=3 \
  -p hough_param2:=10.0 \
  -p min_score:=0.55 \
  -p min_free_ring_ratio:=0.65 \
  -p min_circle_support_ratio:=0.75 \
  -p max_unknown_ring_ratio:=0.35
```

The mission controller default `approach_offset` is `0.15`. With the generated
surface fields, the requested goal is `0.15 m` outside the measured barrel edge.
For older YAML entries without surface fields, it falls back to `width / 2 +
approach_offset` from the barrel center. If Nav2 still refuses to get that
close, reduce the Nav2 costmap inflation radius in your Nav2 config.

Fallback behavior can be tuned with `fallback_turn_radians`,
`fallback_backup_distance`, `fallback_backup_speed`, and
`fallback_time_allowance_sec` on `mission_controller`.

If red markers appear on non-barrels, increase `min_score`, lower
`max_square_corner_ratio`, or lower `max_context_occupied_ratio`. If real
barrels are missed on a live SLAM map, lower `min_radius_pixels`,
`hough_param2`, `min_circle_support_ratio`, or `min_free_ring_ratio`.
The same coarse adjustment is available as `detection_sensitivity` in ROS and
`--sensitivity` in the offline marker script.

Barrels tangent to a wall are handled by `allow_wall_touching:=true`, which is
enabled by default. This only relaxes the wall-side ring/context checks when
the occupied pixels look like a narrow tangent contact.

At high sensitivity, or when `Expected barrels` is greater than the number of
Hough detections, the detector also runs auxiliary contour/template classifiers
after subtracting wall-like components. These are intended for weak or partial
barrel marks, but they are less conservative than the default Hough path.
