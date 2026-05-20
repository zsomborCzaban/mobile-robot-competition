import math
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformException, TransformListener


class ScanTFRepair(Node):
    def __init__(self) -> None:
        super().__init__('scan_tf_repair')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('parent_frame', 'base_link')
        self.declare_parameter('x', 0.0)
        self.declare_parameter('y', 0.0)
        self.declare_parameter('z', 0.10)
        self.declare_parameter('roll', 0.0)
        self.declare_parameter('pitch', 0.0)
        self.declare_parameter('yaw', 0.0)
        self.declare_parameter('repair_after_sec', 2.0)
        self.declare_parameter('status_period_sec', 1.0)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.parent_frame = self.clean_frame(
            str(self.get_parameter('parent_frame').value)
        )
        self.repair_after_sec = max(
            float(self.get_parameter('repair_after_sec').value),
            0.0,
        )
        status_period_sec = max(
            float(self.get_parameter('status_period_sec').value),
            0.2,
        )

        self.scan_frame: Optional[str] = None
        self.first_scan_monotonic: Optional[float] = None
        self.last_status = 'waiting for scan'
        self.repair_published = False

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.static_broadcaster = StaticTransformBroadcaster(self)
        self.status_pub = self.create_publisher(String, '/scan_tf_repair_status', 10)

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.create_timer(status_period_sec, self.check_and_publish_status)

        self.get_logger().info(
            f'Watching {self.scan_topic}; will repair missing '
            f'{self.parent_frame}-><scan frame> static TF if needed.'
        )

    @staticmethod
    def clean_frame(frame: str) -> str:
        return frame.strip().lstrip('/')

    def scan_callback(self, message: LaserScan) -> None:
        frame = self.clean_frame(message.header.frame_id)
        if not frame:
            self.last_status = f'WAIT: {self.scan_topic} has empty frame_id'
            return

        if self.scan_frame != frame:
            self.scan_frame = frame
            self.first_scan_monotonic = time.monotonic()
            self.repair_published = False
            self.get_logger().info(f'Detected scan frame: {frame}')

    def check_and_publish_status(self) -> None:
        self.last_status = self.status_text()
        message = String()
        message.data = self.last_status
        self.status_pub.publish(message)

    def status_text(self) -> str:
        if not self.scan_frame or self.first_scan_monotonic is None:
            return f'WAIT: no scan received on {self.scan_topic}'

        if self.scan_frame == self.parent_frame:
            return f'OK: scan frame is already {self.parent_frame}'

        if self.transform_exists():
            return f'OK: TF {self.parent_frame}->{self.scan_frame} exists'

        age = time.monotonic() - self.first_scan_monotonic
        if age < self.repair_after_sec:
            return (
                f'WAIT: missing TF {self.parent_frame}->{self.scan_frame}; '
                f'waiting {self.repair_after_sec - age:.1f}s before repair'
            )

        if not self.repair_published:
            self.publish_repair_transform()
            self.repair_published = True

        return f'REPAIRED: published static TF {self.parent_frame}->{self.scan_frame}'

    def transform_exists(self) -> bool:
        try:
            self.tf_buffer.lookup_transform(
                self.parent_frame,
                self.scan_frame,
                rclpy.time.Time(),
            )
            return True
        except TransformException:
            return False

    def publish_repair_transform(self) -> None:
        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self.parent_frame
        transform.child_frame_id = self.scan_frame
        transform.transform.translation.x = float(self.get_parameter('x').value)
        transform.transform.translation.y = float(self.get_parameter('y').value)
        transform.transform.translation.z = float(self.get_parameter('z').value)
        transform.transform.rotation = self.quaternion_from_rpy(
            float(self.get_parameter('roll').value),
            float(self.get_parameter('pitch').value),
            float(self.get_parameter('yaw').value),
        )
        self.static_broadcaster.sendTransform(transform)
        self.get_logger().warn(
            f'Published repair static TF {self.parent_frame}->{self.scan_frame}. '
            'If this fixes SLAM, make the same frame relationship permanent in '
            'the robot description or lidar driver config.'
        )

    @staticmethod
    def quaternion_from_rpy(roll: float, pitch: float, yaw: float):
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        from geometry_msgs.msg import Quaternion

        q = Quaternion()
        q.w = cr * cp * cy + sr * sp * sy
        q.x = sr * cp * cy - cr * sp * sy
        q.y = cr * sp * cy + sr * cp * sy
        q.z = cr * cp * sy - sr * sp * cy
        return q


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScanTFRepair()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
