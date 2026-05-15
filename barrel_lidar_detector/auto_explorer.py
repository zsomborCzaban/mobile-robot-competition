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
        self.declare_parameter('forward_speed', 0.12)
        self.declare_parameter('turn_speed', 0.45)
        self.declare_parameter('front_stop_distance', 0.38)
        self.declare_parameter('unexpected_obstacle_distance', 0.20)
        self.declare_parameter('front_sector_deg', 35.0)
        self.declare_parameter('normal_turn_degrees', 165.0)
        self.declare_parameter('collision_turn_degrees', 15.0)
        self.declare_parameter('scan_timeout_sec', 1.0)
        self.declare_parameter('control_rate_hz', 10.0)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.cmd_vel_topic = self.get_parameter('cmd_vel_topic').value

        self.active = False
        self.state = 'IDLE'
        self.latest_scan: Optional[LaserScan] = None
        self.latest_scan_time = None
        self.turn_end_time = None
        self.turn_direction = 1.0

        self.scan_sub = self.create_subscription(
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

    def start_callback(self, request, response):
        if self.active:
            response.success = True
            response.message = 'Auto exploration is already running.'
            return response

        self.active = True
        self.state = 'DRIVING'
        self.turn_end_time = None
        message = 'Auto exploration started.'
        self.get_logger().info(message)
        self.publish_status(message)
        response.success = True
        response.message = message
        return response

    def stop_callback(self, request, response):
        self.stop_robot()
        self.active = False
        self.state = 'IDLE'
        self.turn_end_time = None
        message = 'Auto exploration stopped.'
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

        front_distance = self.front_distance(self.latest_scan)
        if front_distance is None:
            self.stop_robot()
            self.publish_status('Auto exploration has no usable front LiDAR ranges.')
            return

        unexpected_distance = float(
            self.get_parameter('unexpected_obstacle_distance').value
        )
        stop_distance = float(self.get_parameter('front_stop_distance').value)

        if front_distance <= unexpected_distance:
            self.start_turn(
                float(self.get_parameter('collision_turn_degrees').value),
                reason='unexpected obstacle',
            )
        elif self.state == 'DRIVING' and front_distance <= stop_distance:
            self.start_turn(
                float(self.get_parameter('normal_turn_degrees').value),
                reason='wall ahead',
            )

        if self.state == 'TURNING':
            if self.turn_end_time is not None and self.get_clock().now() >= self.turn_end_time:
                self.state = 'DRIVING'
                self.publish_status('Auto exploration driving forward.')
                self.drive_forward()
            else:
                self.turn_in_place()
            return

        self.drive_forward()

    def scan_is_fresh(self) -> bool:
        if self.latest_scan is None or self.latest_scan_time is None:
            return False

        age = self.get_clock().now() - self.latest_scan_time
        timeout = float(self.get_parameter('scan_timeout_sec').value)
        return age.nanoseconds / 1e9 <= timeout

    def front_distance(self, scan: Optional[LaserScan]) -> Optional[float]:
        if scan is None:
            return None

        half_sector = math.radians(
            max(float(self.get_parameter('front_sector_deg').value), 1.0)
        ) * 0.5
        ranges: List[float] = []

        for index, range_m in enumerate(scan.ranges):
            if not math.isfinite(range_m):
                continue

            angle = scan.angle_min + index * scan.angle_increment
            normalized_angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(normalized_angle) > half_sector:
                continue

            valid_min = max(scan.range_min, 0.01)
            valid_max = scan.range_max if scan.range_max > 0.0 else math.inf
            if valid_min <= range_m <= valid_max:
                ranges.append(range_m)

        if not ranges:
            return None

        return min(ranges)

    def start_turn(self, turn_degrees: float, reason: str) -> None:
        if self.state == 'TURNING':
            return

        turn_speed = max(float(self.get_parameter('turn_speed').value), 0.01)
        turn_radians = math.radians(max(abs(turn_degrees), 1.0))
        duration_sec = turn_radians / turn_speed
        self.turn_end_time = self.get_clock().now() + Duration(seconds=duration_sec)
        self.turn_direction *= -1.0
        self.state = 'TURNING'
        self.stop_robot()
        self.publish_status(
            f'Auto exploration turning {turn_degrees:.0f} deg: {reason}.'
        )

    def drive_forward(self) -> None:
        twist = Twist()
        twist.linear.x = float(self.get_parameter('forward_speed').value)
        self.cmd_pub.publish(twist)

    def turn_in_place(self) -> None:
        twist = Twist()
        twist.angular.z = (
            self.turn_direction * float(self.get_parameter('turn_speed').value)
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

    def stop_on_signal(signum, frame) -> None:
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
