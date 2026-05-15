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
from std_msgs.msg import String
from std_srvs.srv import Trigger
import tf2_geometry_msgs  # noqa: F401  Registers geometry_msgs transforms with tf2.
from tf2_ros import Buffer, TransformException, TransformListener


class BarrelMissionController(Node):
    def __init__(self) -> None:
        super().__init__('barrel_mission_controller')

        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('approach_offset', 0.15)
        self.declare_parameter('expected_barrel_count', 0)
        self.declare_parameter('pose_timeout_sec', 1.0)
        self.declare_parameter('fallback_turn_radians', math.pi)
        self.declare_parameter('fallback_backup_distance', 1.0)
        self.declare_parameter('fallback_backup_speed', 0.15)
        self.declare_parameter('fallback_time_allowance_sec', 20.0)
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
        self.recovery_phase = ''
        self.recovery_visited_ids = set()

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.navigator = BasicNavigator()

        self.create_service(Trigger, 'calculate_target', self.calc_callback)
        self.create_service(Trigger, 'start_navigation', self.nav_callback)
        self.create_service(Trigger, 'pause_navigation', self.pause_callback)
        self.create_service(Trigger, 'resume_navigation', self.resume_callback)
        self.create_service(Trigger, 'stop_navigation', self.stop_callback)
        self.create_service(Trigger, 'fallback_recovery', self.fallback_callback)
        self.status_pub = self.create_publisher(String, 'mission_status', 10)

        self.timer = self.create_timer(0.5, self.control_loop)

        self.get_logger().info(
            'Multi-barrel mission controller ready. Barrel YAML: '
            f'{self.barrel_yaml_path()}, expected barrels: '
            f'{self.expected_barrel_count_label()}'
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

        success, message = self.calculate_route_from_barrels(barrels)
        response.success = success
        response.message = message
        self.get_logger().info(response.message)
        self.publish_status(response.message)
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
        waypoint_message = self.send_current_waypoint()

        response.success = True
        response.message = waypoint_message or (
            f'Navigation started for {len(self.waypoints)} barrels.'
        )
        return response

    def pause_callback(self, request, response):
        if self.state == 'NAVIGATING':
            self.navigator.cancelTask()
            self.state = 'PAUSED'
            response.success, response.message = True, 'Robot paused.'
            self.publish_status(response.message)
        else:
            response.success, response.message = False, 'Not currently navigating.'
        return response

    def resume_callback(self, request, response):
        if self.state == 'PAUSED' and self.waypoints:
            self.state = 'NAVIGATING'
            waypoint_message = self.send_current_waypoint()
            response.success = True
            response.message = waypoint_message or 'Resuming navigation.'
        else:
            response.success, response.message = False, 'Robot is not paused.'
        return response

    def stop_callback(self, request, response):
        self.navigator.cancelTask()
        self.state = 'IDLE'
        self.waypoints = []
        self.ordered_barrels = []
        self.current_waypoint_index = 0
        self.recovery_phase = ''
        self.recovery_visited_ids = set()
        response.success, response.message = True, 'Mission stopped and reset.'
        self.publish_status(response.message)
        return response

    def fallback_callback(self, request, response):
        if self.state == 'RECOVERING':
            response.success = False
            response.message = 'Fallback recovery is already running.'
            return response

        if self.state not in ('NAVIGATING', 'PAUSED', 'READY'):
            response.success = False
            response.message = 'Fallback requires an active or ready mission.'
            return response

        self.recovery_visited_ids = {
            str(barrel['id'])
            for barrel in self.ordered_barrels[:self.current_waypoint_index]
        }
        self.navigator.cancelTask()
        self.navigator.waitUntilNav2Active()
        self.state = 'RECOVERING'
        self.recovery_phase = 'TURNING'
        turn_radians = float(self.get_parameter('fallback_turn_radians').value)
        time_allowance = float(
            self.get_parameter('fallback_time_allowance_sec').value
        )
        self.navigator.spin(
            spin_dist=turn_radians,
            time_allowance=time_allowance,
        )

        response.success = True
        response.message = (
            'Fallback started: turning around, backing up, then recalculating route.'
        )
        self.publish_status(response.message)
        return response

    def send_current_waypoint(self) -> Optional[str]:
        if self.current_waypoint_index >= len(self.waypoints):
            message = 'No remaining barrel waypoints.'
            self.get_logger().info(message)
            self.publish_status(message)
            self.state = 'IDLE'
            return message

        goal = self.waypoints[self.current_waypoint_index]
        goal.header.stamp = self.get_clock().now().to_msg()
        self.navigator.goToPose(goal)
        barrel = self.ordered_barrels[self.current_waypoint_index]
        message = (
            f'Going to barrel {self.current_waypoint_index + 1}/'
            f'{len(self.waypoints)}: {barrel["id"]}'
        )
        self.get_logger().info(message)
        self.publish_status(message)
        return message

    def control_loop(self):
        if self.state == 'RECOVERING':
            self.recovery_loop()
            return

        if self.state != 'NAVIGATING':
            return

        if self.navigator.isTaskComplete():
            result = self.navigator.getResult()
            if result == TaskResult.SUCCEEDED:
                self.current_waypoint_index += 1
                if self.current_waypoint_index >= len(self.waypoints):
                    message = 'All barrels reached. Mission complete.'
                    self.get_logger().info(message)
                    self.publish_status(message)
                    self.state = 'IDLE'
                else:
                    self.send_current_waypoint()
            elif result == TaskResult.CANCELED:
                message = 'Navigation task canceled.'
                self.get_logger().info(message)
                self.publish_status(message)
                if self.state == 'NAVIGATING':
                    self.state = 'IDLE'
            elif result == TaskResult.FAILED:
                barrel = self.ordered_barrels[self.current_waypoint_index]
                message = (
                    f'Failed to reach {barrel["id"]}; continuing to next barrel.'
                )
                self.get_logger().warn(message)
                self.publish_status(message)
                self.current_waypoint_index += 1
                if self.current_waypoint_index < len(self.waypoints):
                    self.send_current_waypoint()
                else:
                    self.publish_status('Mission ended after final failed waypoint.')
                    self.state = 'IDLE'

    def recovery_loop(self) -> None:
        if not self.navigator.isTaskComplete():
            return

        result = self.navigator.getResult()
        if result != TaskResult.SUCCEEDED:
            message = f'Fallback {self.recovery_phase.lower()} failed.'
            self.get_logger().warn(message)
            self.publish_status(message)
            self.state = 'PAUSED' if self.waypoints else 'IDLE'
            self.recovery_phase = ''
            return

        if self.recovery_phase == 'TURNING':
            self.recovery_phase = 'BACKING_UP'
            backup_distance = float(
                self.get_parameter('fallback_backup_distance').value
            )
            backup_speed = float(self.get_parameter('fallback_backup_speed').value)
            time_allowance = float(
                self.get_parameter('fallback_time_allowance_sec').value
            )
            self.navigator.backup(
                backup_dist=backup_distance,
                backup_speed=backup_speed,
                time_allowance=time_allowance,
            )
            self.publish_status(f'Fallback: backing up {backup_distance:.1f} m.')
            return

        if self.recovery_phase == 'BACKING_UP':
            self.recovery_phase = ''
            success, message = self.recalculate_remaining_route()
            self.publish_status(message)
            if not success:
                self.state = 'IDLE'
                return

            self.state = 'NAVIGATING'
            self.send_current_waypoint()

    def publish_status(self, message: str) -> None:
        status = String()
        status.data = message
        self.status_pub.publish(status)

    def calculate_route_from_barrels(self, barrels: List[Dict]) -> Tuple[bool, str]:
        robot_xy = self.robot_position()
        if robot_xy is None:
            message = (
                'Robot pose unavailable; cannot calculate route. Check TF '
                'map->odom->base_link or map->odom->base_footprint.'
            )
            self.get_logger().warn(message)
            return False, message

        self.ordered_barrels = self.astar_barrel_order(robot_xy, barrels)
        self.waypoints = self.build_waypoints(robot_xy, self.ordered_barrels)
        self.current_waypoint_index = 0
        self.state = 'READY'

        route = ', '.join(str(barrel['id']) for barrel in self.ordered_barrels)
        first_barrel = str(self.ordered_barrels[0]['id'])
        message = (
            f'Calculated shortest route for {len(self.waypoints)} barrels '
            f'from robot ({robot_xy[0]:.2f}, {robot_xy[1]:.2f}). '
            f'First: {first_barrel}. Route: {route}'
        )
        return True, message

    def recalculate_remaining_route(self) -> Tuple[bool, str]:
        yaml_path = self.barrel_yaml_path()
        try:
            barrels = self.load_barrels(yaml_path)
        except (OSError, ValueError, yaml.YAMLError) as exc:
            message = f'Fallback route recalculation failed: {exc}'
            self.get_logger().warn(message)
            return False, message

        remaining_barrels = [
            barrel
            for barrel in barrels
            if str(barrel['id']) not in self.recovery_visited_ids
        ]
        if not remaining_barrels:
            message = 'Fallback found no remaining barrels to navigate to.'
            self.get_logger().warn(message)
            return False, message

        success, message = self.calculate_route_from_barrels(remaining_barrels)
        if success:
            message = f'Fallback complete. {message}'
        return success, message

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

            barrel = {
                'id': str(raw_barrel.get('id', f'Barrel_{index:03d}')),
                'map_x': map_x,
                'map_y': map_y,
                'width': width,
                'roundness': raw_barrel.get('roundness', 0.0),
                'score': raw_barrel.get('score', 0.0),
                'confirmations': raw_barrel.get('confirmations', 0),
            }
            surface_fields = self.surface_fields(raw_barrel)
            if surface_fields is not None:
                barrel.update(surface_fields)

            barrels.append(barrel)

        barrels.sort(key=self.barrel_strength, reverse=True)
        expected_count = self.expected_barrel_count()
        if expected_count > 0:
            barrels = barrels[:expected_count]

        return barrels

    @staticmethod
    def surface_fields(raw_barrel: Dict) -> Optional[Dict[str, float]]:
        try:
            surface_x = float(raw_barrel['surface_x'])
            surface_y = float(raw_barrel['surface_y'])
            normal_x = float(raw_barrel['normal_x'])
            normal_y = float(raw_barrel['normal_y'])
        except (KeyError, TypeError, ValueError):
            return None

        if not all(
            math.isfinite(value)
            for value in (surface_x, surface_y, normal_x, normal_y)
        ):
            return None

        normal_length = math.hypot(normal_x, normal_y)
        if normal_length <= 1e-6:
            return None

        return {
            'surface_x': surface_x,
            'surface_y': surface_y,
            'normal_x': normal_x / normal_length,
            'normal_y': normal_y / normal_length,
        }

    @staticmethod
    def barrel_strength(barrel: Dict) -> Tuple[float, float, float]:
        try:
            confirmations = float(barrel.get('confirmations', 0.0))
        except (TypeError, ValueError):
            confirmations = 0.0

        try:
            score = float(barrel.get('score', 0.0))
        except (TypeError, ValueError):
            score = 0.0

        try:
            roundness = float(barrel.get('roundness', 0.0))
        except (TypeError, ValueError):
            roundness = 0.0

        return confirmations, score, roundness

    def expected_barrel_count(self) -> int:
        try:
            count = int(self.get_parameter('expected_barrel_count').value)
        except (TypeError, ValueError):
            count = 0
        return max(count, 0)

    def expected_barrel_count_label(self) -> str:
        count = self.expected_barrel_count()
        return str(count) if count > 0 else 'unlimited'

    def astar_barrel_order(
        self,
        start_xy: Tuple[float, float],
        barrels: List[Dict],
    ) -> List[Dict]:
        route_points = [self.barrel_route_xy(barrel) for barrel in barrels]
        goal_mask = (1 << len(barrels)) - 1
        start_state = (-1, 0)
        best_cost = {start_state: 0.0}
        queue = [
            (
                self.route_heuristic(-1, 0, start_xy, route_points),
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
                    route_points,
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
                    route_points,
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
        route_points: List[Tuple[float, float]],
    ) -> float:
        distances = [
            self.route_distance(current_index, next_index, start_xy, route_points)
            for next_index in range(len(route_points))
            if not visited_mask & (1 << next_index)
        ]
        return min(distances) if distances else 0.0

    @staticmethod
    def route_distance(
        current_index: int,
        next_index: int,
        start_xy: Tuple[float, float],
        route_points: List[Tuple[float, float]],
    ) -> float:
        if current_index < 0:
            current_x, current_y = start_xy
        else:
            current_x, current_y = route_points[current_index]

        next_x, next_y = route_points[next_index]
        return math.hypot(next_x - current_x, next_y - current_y)

    def barrel_route_xy(self, barrel: Dict) -> Tuple[float, float]:
        approach_offset = max(
            float(self.get_parameter('approach_offset').value),
            0.0,
        )
        surface = self.surface_fields(barrel)
        if surface is not None:
            return (
                surface['surface_x'] + surface['normal_x'] * approach_offset,
                surface['surface_y'] + surface['normal_y'] * approach_offset,
            )

        return float(barrel['map_x']), float(barrel['map_y'])

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
        approach_offset = max(
            float(self.get_parameter('approach_offset').value),
            0.0,
        )

        surface_waypoint = self.surface_waypoint(barrel, approach_offset)
        if surface_waypoint is not None:
            return surface_waypoint

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

    def surface_waypoint(
        self,
        barrel: Dict,
        approach_offset: float,
    ) -> Optional[Tuple[PoseStamped, Tuple[float, float]]]:
        surface = self.surface_fields(barrel)
        if surface is None:
            return None

        surface_x = surface['surface_x']
        surface_y = surface['surface_y']
        normal_x = surface['normal_x']
        normal_y = surface['normal_y']

        goal_x = surface_x + normal_x * approach_offset
        goal_y = surface_y + normal_y * approach_offset
        yaw = math.atan2(-normal_y, -normal_x)

        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = goal_x
        pose.pose.position.y = goal_y
        pose.pose.position.z = 0.0
        pose.pose.orientation = self.yaw_to_quaternion(yaw)
        return pose, (goal_x, goal_y)

    def robot_position(self) -> Optional[Tuple[float, float]]:
        timeout_sec = max(float(self.get_parameter('pose_timeout_sec').value), 0.1)
        last_error = None

        for base_frame in self.base_frame_candidates():
            try:
                transform = self.tf_buffer.lookup_transform(
                    self.target_frame,
                    base_frame,
                    Time(),
                    timeout=Duration(seconds=timeout_sec),
                )
                if base_frame != self.base_frame:
                    self.get_logger().info(
                        f'Using fallback base frame {base_frame} for route planning.',
                        throttle_duration_sec=5.0,
                    )
                return (
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                )
            except TransformException as exc:
                last_error = exc

        self.get_logger().warn(
            f'Could not look up robot pose in {self.target_frame} using '
            f'{self.base_frame_candidates()}: {last_error}',
            throttle_duration_sec=2.0,
        )
        return None

    def base_frame_candidates(self) -> List[str]:
        candidates = [self.base_frame]
        for frame in ('base_footprint', 'base_link'):
            if frame not in candidates:
                candidates.append(frame)
        return candidates

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
