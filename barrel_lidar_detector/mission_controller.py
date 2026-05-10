import math
from typing import Dict, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, Quaternion
from nav2_simple_commander.robot_navigator import BasicNavigator
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
        self.declare_parameter('confirmed_pose_topic', '/barrel_confirmed_pose')
        self.declare_parameter('lidar_pose_topic', '/barrel_pose')
        self.declare_parameter('map_pose_topic', '/barrel_map_pose')

        self.target_frame = self.get_parameter('target_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.latest_poses: Dict[str, Tuple[PoseStamped, Time]] = {}
        self.target_pose: Optional[PoseStamped] = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.navigator = BasicNavigator()

        self.create_subscription(
            PoseStamped,
            self.get_parameter('confirmed_pose_topic').value,
            lambda msg: self.pose_callback('confirmed', msg),
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('lidar_pose_topic').value,
            lambda msg: self.pose_callback('lidar', msg),
            10,
        )
        self.create_subscription(
            PoseStamped,
            self.get_parameter('map_pose_topic').value,
            lambda msg: self.pose_callback('map', msg),
            10,
        )

        self.create_service(Trigger, 'calculate_target', self.calc_callback)
        self.create_service(Trigger, 'start_navigation', self.nav_callback)

        self.get_logger().info(
            'Mission controller ready. Preferred target source order: '
            'confirmed, lidar, map.'
        )

    def pose_callback(self, source: str, pose: PoseStamped) -> None:
        pose_in_target_frame = self.transform_pose(pose, self.target_frame)
        if pose_in_target_frame is None:
            return

        self.latest_poses[source] = (pose_in_target_frame, self.get_clock().now())

    def calc_callback(self, request, response):
        barrel_pose, source = self.best_recent_barrel_pose()
        if barrel_pose is None:
            response.success = False
            response.message = (
                'No recent barrel pose available. Start the LiDAR detector and '
                'map-shape detector, then wait for /barrel_pose or '
                '/barrel_confirmed_pose.'
            )
            self.get_logger().warn(response.message)
            return response

        robot_xy = self.robot_position()
        if robot_xy is None:
            goal_x, goal_y = self.fallback_goal_position(barrel_pose)
        else:
            goal_x, goal_y = self.offset_goal_toward_robot(barrel_pose, robot_xy)

        barrel_x = barrel_pose.pose.position.x
        barrel_y = barrel_pose.pose.position.y
        yaw = math.atan2(barrel_y - goal_y, barrel_x - goal_x)

        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = self.target_frame
        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.target_pose.pose.position.x = goal_x
        self.target_pose.pose.position.y = goal_y
        self.target_pose.pose.position.z = 0.0
        self.target_pose.pose.orientation = self.yaw_to_quaternion(yaw)

        response.success = True
        response.message = (
            f'Calculated goal from {source} barrel pose: '
            f'x={goal_x:.2f}, y={goal_y:.2f}'
        )
        self.get_logger().info(response.message)
        return response

    def nav_callback(self, request, response):
        if self.target_pose is None:
            response.success = False
            response.message = 'Calculate the target before starting navigation.'
            self.get_logger().warn(response.message)
            return response

        self.target_pose.header.stamp = self.get_clock().now().to_msg()
        self.get_logger().info('Starting navigation to calculated target.')
        self.navigator.waitUntilNav2Active()
        self.navigator.goToPose(self.target_pose)

        response.success = True
        response.message = 'Navigation command sent to Nav2.'
        return response

    def best_recent_barrel_pose(self) -> Tuple[Optional[PoseStamped], Optional[str]]:
        for source in ('confirmed', 'lidar', 'map'):
            pose_info = self.latest_poses.get(source)
            if pose_info is None:
                continue

            pose, stamp = pose_info
            age = self.get_clock().now() - stamp
            timeout_sec = float(self.get_parameter('pose_timeout_sec').value)
            if age.nanoseconds <= Duration(seconds=timeout_sec).nanoseconds:
                return pose, source

        return None, None

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
                f'Could not look up robot pose in {self.target_frame}: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

        return (
            transform.transform.translation.x,
            transform.transform.translation.y,
        )

    def offset_goal_toward_robot(
        self,
        barrel_pose: PoseStamped,
        robot_xy: Tuple[float, float],
    ) -> Tuple[float, float]:
        barrel_x = barrel_pose.pose.position.x
        barrel_y = barrel_pose.pose.position.y
        robot_x, robot_y = robot_xy
        dx = robot_x - barrel_x
        dy = robot_y - barrel_y
        distance = math.hypot(dx, dy)

        if distance < 1e-3:
            return self.fallback_goal_position(barrel_pose)

        approach_offset = float(self.get_parameter('approach_offset').value)
        return (
            barrel_x + approach_offset * dx / distance,
            barrel_y + approach_offset * dy / distance,
        )

    def fallback_goal_position(self, barrel_pose: PoseStamped) -> Tuple[float, float]:
        approach_offset = float(self.get_parameter('approach_offset').value)
        return (
            barrel_pose.pose.position.x - approach_offset,
            barrel_pose.pose.position.y,
        )

    def transform_pose(
        self,
        pose: PoseStamped,
        target_frame: str,
    ) -> Optional[PoseStamped]:
        if pose.header.frame_id == target_frame:
            return pose

        try:
            return self.tf_buffer.transform(
                pose,
                target_frame,
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Could not transform pose from {pose.header.frame_id} '
                f'to {target_frame}: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

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
