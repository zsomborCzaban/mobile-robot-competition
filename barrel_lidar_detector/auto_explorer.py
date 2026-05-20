import math
import signal
from typing import List, Optional

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger


class AutoExplorer(Node):
    def __init__(self) -> None:
        super().__init__('auto_explorer')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('cmd_vel_topic', '/cmd_vel')
        self.declare_parameter('forward_speed', 0.11)
        self.declare_parameter('turn_speed', 0.45)
        self.declare_parameter('front_stop_distance', 0.42)
        self.declare_parameter('side_target_distance', 0.45)
        self.declare_parameter('side_open_distance', 0.80)
        self.declare_parameter('side_too_close_distance', 0.26)
        self.declare_parameter('front_sector_deg', 36.0)
        self.declare_parameter('side_sector_deg', 34.0)
        self.declare_parameter('corridor_sector_deg', 26.0)
        self.declare_parameter('turn_degrees', 86.0)
        self.declare_parameter('scan_timeout_sec', 1.0)
        self.declare_parameter('control_rate_hz', 10.0)
        self.declare_parameter('wall_follow_side', 'right')
        self.declare_parameter('wall_gain', 0.9)
        self.declare_parameter('max_correction_turn_speed', 0.28)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.active = False
        self.state = 'IDLE'
        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_time = None
        self.turn_end_time = None
        self.turn_direction = -1.0

        self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.status_pub = self.create_publisher(String, 'mission_status', 10)
        self.create_service(Trigger, 'start_auto_exploration', self.start_callback)
        self.create_service(Trigger, 'stop_auto_exploration', self.stop_callback)

        control_rate = max(float(self.get_parameter('control_rate_hz').value), 1.0)
        self.timer = self.create_timer(1.0 / control_rate, self.control_loop)

        self.get_logger().info(
            f'Auto explorer ready. Subscribing to {self.scan_topic}, '
            f'publishing velocity commands to {self.cmd_vel_topic}.'
        )

    def start_callback(self, _request, response):
        if self.active:
            response.success = True
            response.message = 'Auto exploration is already running.'
            return response

        self.active = True
        self.state = 'DRIVING'
        self.turn_end_time = None
        self.turn_direction = -1.0 if self.following_right_wall() else 1.0
        message = 'Maze auto exploration started.'
        self.get_logger().info(message)
        self.publish_status(message)
        response.success = True
        response.message = message
        return response

    def stop_callback(self, _request, response):
        self.stop_robot()
        self.active = False
        self.state = 'IDLE'
        self.turn_end_time = None
        message = 'Maze auto exploration stopped.'
        self.get_logger().info(message)
        self.publish_status(message)
        response.success = True
        response.message = message
        return response

    def scan_callback(self, scan: LaserScan) -> None:
        self.latest_scan = scan
        self.latest_scan_time = self.get_clock().now()

    def control_loop(self) -> None:
        if not self.active:
            return

        if not self.scan_is_fresh():
            self.stop_robot()
            self.publish_status('Auto exploration waiting for fresh LiDAR scan.')
            return

        scan = self.latest_scan
        front = self.sector_distance(scan, 0.0, 'front_sector_deg')
        left = self.sector_distance(scan, 90.0, 'side_sector_deg')
        right = self.sector_distance(scan, -90.0, 'side_sector_deg')
        front_left = self.sector_distance(scan, 35.0, 'corridor_sector_deg')
        front_right = self.sector_distance(scan, -35.0, 'corridor_sector_deg')

        if front is None:
            self.stop_robot()
            self.publish_status('Auto exploration has no usable front LiDAR ranges.')
            return

        if self.state == 'TURNING':
            if self.turn_end_time is not None and self.get_clock().now() >= self.turn_end_time:
                self.state = 'DRIVING'
                self.publish_status('Auto exploration following maze wall.')
                self.drive_with_wall_following(left, right, front_left, front_right)
            else:
                self.turn_in_place()
            return

        stop_distance = float(self.get_parameter('front_stop_distance').value)
        if front <= stop_distance:
            self.start_turn(self.blocked_turn_direction(left, right), 'wall ahead')
            return

        open_distance = float(self.get_parameter('side_open_distance').value)
        if self.following_right_wall() and self.distance_is_open(right, open_distance):
            self.start_turn(-1.0, 'right corridor opening')
            return
        if not self.following_right_wall() and self.distance_is_open(left, open_distance):
            self.start_turn(1.0, 'left corridor opening')
            return

        self.drive_with_wall_following(left, right, front_left, front_right)

    def scan_is_fresh(self) -> bool:
        if self.latest_scan is None or self.latest_scan_time is None:
            return False

        age = self.get_clock().now() - self.latest_scan_time
        timeout = float(self.get_parameter('scan_timeout_sec').value)
        return age.nanoseconds / 1e9 <= timeout

    def sector_distance(
        self,
        scan: Optional[LaserScan],
        center_degrees: float,
        width_parameter: str,
    ) -> Optional[float]:
        if scan is None:
            return None

        half_width = math.radians(
            max(float(self.get_parameter(width_parameter).value), 1.0)
        ) * 0.5
        center = math.radians(center_degrees)
        ranges: List[float] = []

        for index, range_m in enumerate(scan.ranges):
            if not math.isfinite(range_m):
                continue

            angle = scan.angle_min + index * scan.angle_increment
            offset = math.atan2(math.sin(angle - center), math.cos(angle - center))
            if abs(offset) > half_width:
                continue

            valid_min = max(scan.range_min, 0.01)
            valid_max = scan.range_max if scan.range_max > 0.0 else math.inf
            if valid_min <= range_m <= valid_max:
                ranges.append(range_m)

        if not ranges:
            return None

        ranges.sort()
        return ranges[len(ranges) // 2]

    @staticmethod
    def distance_is_open(distance: Optional[float], threshold: float) -> bool:
        return distance is None or distance >= threshold

    def following_right_wall(self) -> bool:
        side = str(self.get_parameter('wall_follow_side').value).strip().lower()
        return side != 'left'

    def blocked_turn_direction(
        self,
        left: Optional[float],
        right: Optional[float],
    ) -> float:
        if left is None and right is None:
            return 1.0
        if left is None:
            return 1.0
        if right is None:
            return -1.0
        return 1.0 if left > right else -1.0

    def start_turn(self, direction: float, reason: str) -> None:
        turn_speed = max(float(self.get_parameter('turn_speed').value), 0.01)
        turn_radians = math.radians(
            max(abs(float(self.get_parameter('turn_degrees').value)), 1.0)
        )
        self.turn_end_time = self.get_clock().now() + Duration(
            seconds=turn_radians / turn_speed
        )
        self.turn_direction = 1.0 if direction >= 0.0 else -1.0
        self.state = 'TURNING'
        self.stop_robot()
        side = 'left' if self.turn_direction > 0.0 else 'right'
        self.publish_status(f'Auto exploration turning {side}: {reason}.')

    def drive_with_wall_following(
        self,
        left: Optional[float],
        right: Optional[float],
        front_left: Optional[float],
        front_right: Optional[float],
    ) -> None:
        twist = Twist()
        twist.linear.x = float(self.get_parameter('forward_speed').value)

        target = float(self.get_parameter('side_target_distance').value)
        too_close = float(self.get_parameter('side_too_close_distance').value)
        gain = float(self.get_parameter('wall_gain').value)
        max_turn = abs(float(self.get_parameter('max_correction_turn_speed').value))

        if self.following_right_wall():
            wall_distance = right
            direction_sign = 1.0
            near_corner_distance = front_right
        else:
            wall_distance = left
            direction_sign = -1.0
            near_corner_distance = front_left

        if wall_distance is not None:
            error = target - wall_distance
            twist.angular.z = max(-max_turn, min(max_turn, direction_sign * gain * error))

        if near_corner_distance is not None and near_corner_distance < too_close:
            twist.angular.z = max(
                -max_turn,
                min(max_turn, direction_sign * max_turn),
            )

        self.cmd_pub.publish(twist)

    def turn_in_place(self) -> None:
        twist = Twist()
        twist.angular.z = self.turn_direction * float(
            self.get_parameter('turn_speed').value
        )
        self.cmd_pub.publish(twist)

    def stop_robot(self) -> None:
        self.cmd_pub.publish(Twist())

    def publish_status(self, message: str) -> None:
        status = String()
        status.data = message
        self.status_pub.publish(status)

    def destroy_node(self) -> bool:
        if rclpy.ok():
            self.stop_robot()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = AutoExplorer()

    def stop_on_signal(_signum, _frame) -> None:
        node.stop_robot()
        if rclpy.ok():
            rclpy.shutdown()

    signal.signal(signal.SIGTERM, stop_on_signal)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
