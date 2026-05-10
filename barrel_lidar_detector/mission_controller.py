import math
import os
import yaml
from typing import Dict, Optional, Tuple, List

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
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

        self.target_frame = self.get_parameter('target_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        
        # --- State Machine Variables ---
        self.state = "IDLE" # IDLE, READY, NAVIGATING, PAUSED
        self.waypoints: List[PoseStamped] = []
        self.current_waypoint_index = 0

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.navigator = BasicNavigator()

        # --- Services ---
        self.create_service(Trigger, 'calculate_target', self.calc_callback)
        self.create_service(Trigger, 'start_navigation', self.nav_callback)
        self.create_service(Trigger, 'pause_navigation', self.pause_callback)
        self.create_service(Trigger, 'resume_navigation', self.resume_callback)
        self.create_service(Trigger, 'stop_navigation', self.stop_callback)

        # --- Heartbeat Timer ---
        self.timer = self.create_timer(0.5, self.control_loop)

        self.get_logger().info('Advanced Multi-Barrel Mission Controller ready.')

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
            self.send_current_waypoint()
            response.success, response.message = True, "Resuming Navigation."
        else:
            response.success, response.message = False, "Robot is not paused."
        return response

    def stop_callback(self, request, response):
        self.navigator.cancelTask()
        self.state = "IDLE"
        self.waypoints = []
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