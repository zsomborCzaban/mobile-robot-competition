import math
import os
import yaml
from typing import Dict, Optional, Tuple, List

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rcl_interfaces.msg import SetParametersResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.time import Time
from std_msgs.msg import Bool
from std_srvs.srv import Trigger
import tf2_geometry_msgs  # noqa: F401  Registers geometry_msgs transforms with tf2.
from tf2_ros import Buffer, TransformException, TransformListener


class BarrelMissionController(Node):
    def __init__(self) -> None:
        super().__init__('barrel_mission_controller')

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('approach_offset', 0.60)
        self.declare_parameter('pose_timeout_sec', 5.0)
        self.declare_parameter('barrel_pose_topic', '/barrel_pose')
        self.declare_parameter('camera_barrel_confirm_topic', '/camera_barrel_confirmed')
        self.declare_parameter('lidar_target_fresh_sec', 1.0)
        self.declare_parameter('use_strict_camera', False)
        self.declare_parameter('strict_camera_validation_topic', '/strict_camera_validation')

        self.target_frame = self.get_parameter('target_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        
        # --- State Machine Variables ---
        self.state = "IDLE" # IDLE, READY, NAVIGATING, PAUSED
        self.waypoints: List[PoseStamped] = []
        self.current_waypoint_index = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.navigator = BasicNavigator()

        self.camera_barrel_confirmed: bool = False
        self.last_lidar_target_time: Optional[Time] = None
        self.use_strict_camera: bool = bool(self.get_parameter('use_strict_camera').value)
        # When True, the next LiDAR-active / camera-false cycle under strict mode may trigger a pause.
        self._strict_pause_edge_armed: bool = True

        barrel_pose_topic = str(self.get_parameter('barrel_pose_topic').value)
        camera_topic = str(self.get_parameter('camera_barrel_confirm_topic').value)
        strict_topic = str(self.get_parameter('strict_camera_validation_topic').value)
        self.create_subscription(
            Bool,
            camera_topic,
            self._camera_barrel_confirm_callback,
            10,
        )
        self.create_subscription(
            PoseStamped,
            barrel_pose_topic,
            self._barrel_pose_callback,
            10,
        )
        self.create_subscription(
            Bool,
            strict_topic,
            self._strict_camera_validation_callback,
            10,
        )

        self.add_on_set_parameters_callback(self._on_set_parameters)

        # --- Services ---
        self.create_service(Trigger, 'calculate_target', self.calc_callback)
        self.create_service(Trigger, 'start_navigation', self.nav_callback)
        self.create_service(Trigger, 'pause_navigation', self.pause_callback)
        self.create_service(Trigger, 'resume_navigation', self.resume_callback)
        self.create_service(Trigger, 'stop_navigation', self.stop_callback)

        # --- Heartbeat Timer ---
        self.timer = self.create_timer(0.5, self.control_loop)

        self.get_logger().info('Advanced Multi-Barrel Mission Controller ready.')

    def _camera_barrel_confirm_callback(self, msg: Bool) -> None:
        self.camera_barrel_confirmed = bool(msg.data)

    def _barrel_pose_callback(self, msg: PoseStamped) -> None:
        self.last_lidar_target_time = self.get_clock().now()

    def _strict_camera_validation_callback(self, msg: Bool) -> None:
        """Safety switch from the UI (or any publisher) on strict_camera_validation_topic."""
        self.use_strict_camera = bool(msg.data)
        self.get_logger().info(
            f'Strict camera validation {"enabled" if self.use_strict_camera else "disabled"} '
            f'(via topic).',
            throttle_duration_sec=1.0,
        )

    def _on_set_parameters(self, params: List[Parameter]) -> SetParametersResult:
        for param in params:
            if param.name == 'use_strict_camera' and param.type_ == Parameter.Type.BOOL:
                self.use_strict_camera = bool(param.value)
                self.get_logger().info(
                    f'Strict camera validation {"enabled" if self.use_strict_camera else "disabled"} '
                    f'(via parameter).',
                )
        return SetParametersResult(successful=True)

    def _lidar_target_is_fresh(self) -> bool:
        if self.last_lidar_target_time is None:
            return False
        age_sec = (
            self.get_clock().now() - self.last_lidar_target_time
        ).nanoseconds * 1e-9
        fresh_sec = float(self.get_parameter('lidar_target_fresh_sec').value)
        return age_sec <= fresh_sec

    def calc_callback(self, request, response):
        """Reads YAML, finds shortest path, calculates dynamic offsets."""
        self.get_logger().info('Calculating Multi-Barrel Shortest Path...')
        yaml_path = os.path.expanduser('~/turtlebot4_ws/barrel_target.yaml')
        
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                unvisited = data['barrels']
        except Exception as e:
            response.success = False
            response.message = f"YAML Error: {e}. Please ensure barrel_target.yaml exists."
            self.get_logger().warn(response.message)
            return response

        robot_xy = self.robot_position()
        if robot_xy is None:
            robot_xy = (0.0, 0.0) # Fallback if TF is missing

        current_x, current_y = robot_xy
        self.waypoints = []
        approach_offset = float(self.get_parameter('approach_offset').value)
        
        while unvisited:
            # 1. Greedy Shortest Path (Nearest Neighbor)
            closest_barrel = min(unvisited, key=lambda b: math.hypot(b['map_x'] - current_x, b['map_y'] - current_y))
            unvisited.remove(closest_barrel)
            
            bx, by = closest_barrel['map_x'], closest_barrel['map_y']
            b_width = closest_barrel['width']
            
            # 2. Dynamic Offset Math (Width / 2 + approach_offset)
            dx = current_x - bx
            dy = current_y - by
            distance = math.hypot(dx, dy)
            
            total_offset = (b_width / 2.0) + approach_offset
            
            if distance < 1e-3:
                # Fallback if robot is exactly on center
                goal_x = bx - total_offset
                goal_y = by
            else:
                goal_x = bx + total_offset * (dx / distance)
                goal_y = by + total_offset * (dy / distance)
                
            yaw = math.atan2(by - goal_y, bx - goal_x)
            
            pose = PoseStamped()
            pose.header.frame_id = self.target_frame
            pose.header.stamp = self.get_clock().now().to_msg()
            pose.pose.position.x = goal_x
            pose.pose.position.y = goal_y
            pose.pose.position.z = 0.0
            pose.pose.orientation = self.yaw_to_quaternion(yaw)
            
            self.waypoints.append(pose)
            current_x, current_y = goal_x, goal_y 

        self.current_waypoint_index = 0
        self.state = "READY"
        response.success = True
        response.message = f"Calculated optimized path for {len(self.waypoints)} barrels."
        self.get_logger().info(response.message)
        return response

    def nav_callback(self, request, response):
        if self.state != "READY":
            response.success = False
            response.message = 'Calculate the target before starting navigation.'
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info('Starting multi-barrel navigation!')
        self.navigator.waitUntilNav2Active()
        self.state = "NAVIGATING"
        self._strict_pause_edge_armed = True
        self.send_current_waypoint()

        response.success = True
        response.message = 'Navigation started.'
        return response

    def pause_callback(self, request, response):
        if self.state == "NAVIGATING":
            self.navigator.cancelTask()
            self.state = "PAUSED"
            response.success, response.message = True, "Robot Paused."
        else:
            response.success, response.message = False, "Not currently navigating."
        return response

    def resume_callback(self, request, response):
        if self.state == "PAUSED":
            self.state = "NAVIGATING"
            self._strict_pause_edge_armed = True
            self.send_current_waypoint()
            response.success, response.message = True, "Resuming Navigation."
        else:
            response.success, response.message = False, "Robot is not paused."
        return response

    def stop_callback(self, request, response):
        self.navigator.cancelTask()
        self.state = "IDLE"
        self.waypoints = []
        self._strict_pause_edge_armed = True
        response.success, response.message = True, "Mission Stopped and Reset."
        return response

    def send_current_waypoint(self):
        goal = self.waypoints[self.current_waypoint_index]
        goal.header.stamp = self.get_clock().now().to_msg()
        self.navigator.goToPose(goal)
        self.get_logger().info(f"Going to barrel {self.current_waypoint_index + 1}/{len(self.waypoints)}")

    def control_loop(self):
        """Timer loop tracking robot progress"""
        if self.state != "NAVIGATING":
            return

        self._apply_strict_camera_policy_if_needed()

        if self.state != "NAVIGATING":
            return

        self._log_visual_confirmation_if_lidar_active()

        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.current_waypoint_index += 1
                if self.current_waypoint_index >= len(self.waypoints):
                    self.get_logger().info("All barrels reached! Mission Complete.")
                    self.state = "IDLE"
                else:
                    self.send_current_waypoint()
            elif result == TaskResult.CANCELED:
                self.get_logger().info("Task canceled by user.")
            elif result == TaskResult.FAILED:
                self.get_logger().warn("Failed to reach barrel. Skipping to next.")
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.waypoints):
                    self.send_current_waypoint()
                else:
                    self.state = "IDLE"

    def _apply_strict_camera_policy_if_needed(self) -> None:
        """If strict camera validation is on, pause navigation when LiDAR is active but the camera disagrees."""
        violation = (
            self.use_strict_camera
            and self._lidar_target_is_fresh()
            and not self.camera_barrel_confirmed
        )
        if not violation:
            self._strict_pause_edge_armed = True
            return

        if not self._strict_pause_edge_armed:
            return

        self._strict_pause_edge_armed = False
        self.get_logger().error(
            'Strict camera validation: LiDAR reports an active barrel target but '
            'camera_barrel_confirmed is False. Pausing navigation for safety.'
        )
        self.navigator.cancelTask()
        self.state = "PAUSED"

    def _log_visual_confirmation_if_lidar_active(self) -> None:
        """While navigating, compare LiDAR barrel pose activity with the camera validator topic."""
        if not self._lidar_target_is_fresh():
            return
        if self.camera_barrel_confirmed:
            self.get_logger().info(
                'Visual Confirmation Success: LiDAR target present and camera_barrel_confirmed is True.',
                throttle_duration_sec=2.0,
            )
            return

        if self.use_strict_camera:
            self.get_logger().warn(
                'Strict camera validation is enabled: LiDAR target is active without camera confirmation. '
                'Navigation is paused when this condition is detected (see error log).',
                throttle_duration_sec=2.0,
            )
        else:
            self.get_logger().warn(
                'Visual confirmation mismatch: LiDAR target is active but camera_barrel_confirmed is False '
                '(logging only; strict camera validation is off).',
                throttle_duration_sec=2.0,
            )

    def robot_position(self) -> Optional[Tuple[float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            self.get_logger().warn(f'Could not look up robot pose: {exc}', throttle_duration_sec=2.0)
            return None
        return (transform.transform.translation.x, transform.transform.translation.y)

    @staticmethod
    def yaw_to_quaternion(yaw: float) -> Quaternion:
        quaternion = Quaternion()
        quaternion.z = math.sin(yaw * 0.5)
        quaternion.w = math.cos(yaw * 0.5)
        return quaternion


def main(args=None) -> None:
    rclpy.init(args=args)
    node = BarrelMissionController()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()