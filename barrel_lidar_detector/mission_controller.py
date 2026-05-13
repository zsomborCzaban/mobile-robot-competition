import heapq
import math
import os
from typing import Dict, List, Optional, Tuple

import yaml

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
        self.declare_parameter(
            'barrel_yaml_path',
            '~/turtlebot4_ws/barrel_target.yaml',
        )

        self.target_frame = self.get_parameter('target_frame').value
        self.base_frame = self.get_parameter('base_frame').value

        self.state = 'IDLE'
        self.waypoints: List[PoseStamped] = []
        self.current_waypoint_index = 0
        self.ordered_barrels: List[Dict] = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.navigator = BasicNavigator()

        self.create_service(Trigger, 'calculate_target', self.calc_callback)
        self.create_service(Trigger, 'start_navigation', self.nav_callback)
        self.create_service(Trigger, 'pause_navigation', self.pause_callback)
        self.create_service(Trigger, 'resume_navigation', self.resume_callback)
        self.create_service(Trigger, 'stop_navigation', self.stop_callback)

        self.timer = self.create_timer(0.5, self.control_loop)

        self.get_logger().info(
            'Multi-barrel mission controller ready. Barrel YAML: '
            f'{self.barrel_yaml_path()}'
        )

    def calc_callback(self, request, response):
        self.get_logger().info('Calculating A* order for all barrels in YAML.')
        yaml_path = self.barrel_yaml_path()

        try:
            barrels = self.load_barrels(yaml_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            response.success = False
            response.message = f'YAML error for {yaml_path}: {exc}'
            self.get_logger().warn(response.message)
            return response

        if not barrels:
            response.success = False
            response.message = f'No valid barrels found in {yaml_path}.'
            self.get_logger().warn(response.message)
            return response

        robot_xy = self.robot_position()
        if robot_xy is None:
            robot_xy = (0.0, 0.0)
            self.get_logger().warn(
                'Robot pose unavailable; using map origin as A* start.'
            )

        self.ordered_barrels = self.astar_barrel_order(robot_xy, barrels)
        self.waypoints = self.build_waypoints(robot_xy, self.ordered_barrels)

        self.current_waypoint_index = 0
        self.state = 'READY'
        route = ', '.join(str(barrel['id']) for barrel in self.ordered_barrels)
        response.success = True
        response.message = (
            f'Calculated A* route for {len(self.waypoints)} barrels: {route}'
        )
        self.get_logger().info(response.message)
        return response

    def nav_callback(self, request, response):
        if self.state != 'READY' or not self.waypoints:
            response.success = False
            response.message = 'Calculate the target before starting navigation.'
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info('Starting navigation through all barrel waypoints.')
        self.navigator.waitUntilNav2Active()
        self.state = 'NAVIGATING'
        self.send_current_waypoint()

        response.success = True
        response.message = f'Navigation started for {len(self.waypoints)} barrels.'
        return response

    def pause_callback(self, request, response):
        if self.state == 'NAVIGATING':
            self.navigator.cancelTask()
            self.state = 'PAUSED'
            response.success, response.message = True, 'Robot paused.'
        else:
            response.success, response.message = False, 'Not currently navigating.'
        return response

    def resume_callback(self, request, response):
        if self.state == 'PAUSED' and self.waypoints:
            self.state = 'NAVIGATING'
            self.send_current_waypoint()
            response.success, response.message = True, 'Resuming navigation.'
        else:
            response.success, response.message = False, 'Robot is not paused.'
        return response

    def stop_callback(self, request, response):
        self.navigator.cancelTask()
        self.state = 'IDLE'
        self.waypoints = []
        self.ordered_barrels = []
        self.current_waypoint_index = 0
        response.success, response.message = True, 'Mission stopped and reset.'
        return response

    def send_current_waypoint(self):
        if self.current_waypoint_index >= len(self.waypoints):
            self.get_logger().info('No remaining barrel waypoints.')
            self.state = 'IDLE'
            return

        goal = self.waypoints[self.current_waypoint_index]
        goal.header.stamp = self.get_clock().now().to_msg()
        self.navigator.goToPose(goal)
        barrel = self.ordered_barrels[self.current_waypoint_index]
        self.get_logger().info(
            f'Going to barrel {self.current_waypoint_index + 1}/'
            f'{len(self.waypoints)}: {barrel["id"]}'
        )

    def control_loop(self):
        if self.state != 'NAVIGATING':
            return

        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.current_waypoint_index += 1
                if self.current_waypoint_index >= len(self.waypoints):
                    self.get_logger().info('All barrels reached. Mission complete.')
                    self.state = 'IDLE'
                else:
                    self.send_current_waypoint()
            elif result == TaskResult.CANCELED:
                self.get_logger().info('Navigation task canceled.')
                if self.state == 'NAVIGATING':
                    self.state = 'IDLE'
            elif result == TaskResult.FAILED:
                barrel = self.ordered_barrels[self.current_waypoint_index]
                self.get_logger().warn(
                    f'Failed to reach {barrel["id"]}; continuing to next barrel.'
                )
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.waypoints):
                    self.send_current_waypoint()
                else:
                    self.state = 'IDLE'

    def barrel_yaml_path(self) -> str:
        return os.path.expanduser(
            str(self.get_parameter('barrel_yaml_path').value)
        )

    def load_barrels(self, yaml_path: str) -> List[Dict]:
        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file) or {}

        raw_barrels = data.get('barrels')
        if not isinstance(raw_barrels, list):
            raise ValueError('expected a top-level "barrels" list')

        barrels: List[Dict] = []
        for index, raw_barrel in enumerate(raw_barrels, start=1):
            if not isinstance(raw_barrel, dict):
                self.get_logger().warn(f'Skipping barrel #{index}: not a mapping.')
                continue

            try:
                map_x = float(raw_barrel['map_x'])
                map_y = float(raw_barrel['map_y'])
                width = float(raw_barrel.get('width', 0.6))
            except (KeyError, TypeError, ValueError) as exc:
                self.get_logger().warn(f'Skipping barrel #{index}: {exc}')
                continue

            if not math.isfinite(map_x) or not math.isfinite(map_y):
                self.get_logger().warn(
                    f'Skipping barrel #{index}: coordinates are not finite.'
                )
                continue

            if not math.isfinite(width) or width <= 0.0:
                width = 0.6

            barrels.append(
                {
                    'id': str(raw_barrel.get('id', f'Barrel_{index:03d}')),
                    'map_x': map_x,
                    'map_y': map_y,
                    'width': width,
                }
            )

        return barrels

    def astar_barrel_order(
        self,
        start_xy: Tuple[float, float],
        barrels: List[Dict],
    ) -> List[Dict]:
        goal_mask = (1 << len(barrels)) - 1
        start_state = (-1, 0)
        best_cost = {start_state: 0.0}
        queue = [
            (
                self.route_heuristic(-1, 0, start_xy, barrels),
                0.0,
                -1,
                0,
                (),
            )
        ]

        while queue:
            _, cost, current_index, visited_mask, path = heapq.heappop(queue)
            if visited_mask == goal_mask:
                return [barrels[index] for index in path]

            state = (current_index, visited_mask)
            if cost > best_cost.get(state, math.inf):
                continue

            for next_index in range(len(barrels)):
                bit = 1 << next_index
                if visited_mask & bit:
                    continue

                step_cost = self.route_distance(
                    current_index,
                    next_index,
                    start_xy,
                    barrels,
                )
                next_cost = cost + step_cost
                next_mask = visited_mask | bit
                next_state = (next_index, next_mask)
                if next_cost >= best_cost.get(next_state, math.inf):
                    continue

                best_cost[next_state] = next_cost
                heuristic = self.route_heuristic(
                    next_index,
                    next_mask,
                    start_xy,
                    barrels,
                )
                heapq.heappush(
                    queue,
                    (
                        next_cost + heuristic,
                        next_cost,
                        next_index,
                        next_mask,
                        path + (next_index,),
                    ),
                )

        return barrels[:]

    def route_heuristic(
        self,
        current_index: int,
        visited_mask: int,
        start_xy: Tuple[float, float],
        barrels: List[Dict],
    ) -> float:
        distances = [
            self.route_distance(current_index, next_index, start_xy, barrels)
            for next_index in range(len(barrels))
            if not visited_mask & (1 << next_index)
        ]
        return min(distances) if distances else 0.0

    @staticmethod
    def route_distance(
        current_index: int,
        next_index: int,
        start_xy: Tuple[float, float],
        barrels: List[Dict],
    ) -> float:
        if current_index < 0:
            current_x, current_y = start_xy
        else:
            current_x = barrels[current_index]['map_x']
            current_y = barrels[current_index]['map_y']

        next_x = barrels[next_index]['map_x']
        next_y = barrels[next_index]['map_y']
        return math.hypot(next_x - current_x, next_y - current_y)

    def build_waypoints(
        self,
        start_xy: Tuple[float, float],
        barrels: List[Dict],
    ) -> List[PoseStamped]:
        current_xy = start_xy
        waypoints: List[PoseStamped] = []

        for barrel in barrels:
            waypoint, current_xy = self.barrel_waypoint(barrel, current_xy)
            waypoints.append(waypoint)

        return waypoints

    def barrel_waypoint(
        self,
        barrel: Dict,
        current_xy: Tuple[float, float],
    ) -> Tuple[PoseStamped, Tuple[float, float]]:
        barrel_x = float(barrel['map_x'])
        barrel_y = float(barrel['map_y'])
        barrel_width = float(barrel.get('width', 0.6))
        approach_offset = float(self.get_parameter('approach_offset').value)
        total_offset = max(barrel_width, 0.0) * 0.5 + approach_offset

        current_x, current_y = current_xy
        dx = current_x - barrel_x
        dy = current_y - barrel_y
        distance = math.hypot(dx, dy)

        if distance < 1e-3:
            goal_x = barrel_x - total_offset
            goal_y = barrel_y
        else:
            goal_x = barrel_x + total_offset * dx / distance
            goal_y = barrel_y + total_offset * dy / distance

        yaw = math.atan2(barrel_y - goal_y, barrel_x - goal_x)
        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.0
        pose.pose.orientation = self.yaw_to_quaternion(yaw)
        return pose, (goal_x, goal_y)

    def robot_position(self) -> Optional[Tuple[float, float]]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                self.base_frame,
                Time(),
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Could not look up robot pose: {exc}',
                throttle_duration_sec=2.0,
            )
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
