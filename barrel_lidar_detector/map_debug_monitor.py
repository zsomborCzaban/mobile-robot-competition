import time
from typing import Optional

import rclpy
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


class MapDebugMonitor(Node):
    def __init__(self) -> None:
        super().__init__('map_debug_monitor')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('stale_after_sec', 5.0)
        self.declare_parameter('status_period_sec', 1.0)

        self.map_topic = str(self.get_parameter('map_topic').value)
        self.stale_after_sec = max(
            float(self.get_parameter('stale_after_sec').value),
            0.1,
        )
        status_period_sec = max(
            float(self.get_parameter('status_period_sec').value),
            0.2,
        )

        self.last_map: Optional[OccupancyGrid] = None
        self.last_map_monotonic: Optional[float] = None
        self.last_qos_label = 'none'
        self.map_messages = 0

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        best_effort_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        marker_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )

        self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            lambda message: self.map_callback(message, 'reliable/transient'),
            reliable_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            lambda message: self.map_callback(message, 'best_effort/volatile'),
            best_effort_qos,
        )

        self.status_pub = self.create_publisher(String, '/map_debug_status', 10)
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/map_debug_markers',
            marker_qos,
        )
        self.create_timer(status_period_sec, self.publish_status)

        self.get_logger().info(
            f'Map debug monitor watching {self.map_topic}. '
            'RViz topics: /map_debug_status, /map_debug_markers'
        )

    def map_callback(self, grid: OccupancyGrid, qos_label: str) -> None:
        self.last_map = grid
        self.last_map_monotonic = time.monotonic()
        self.last_qos_label = qos_label
        self.map_messages += 1

    def publish_status(self) -> None:
        status = self.status_text()

        message = String()
        message.data = status
        self.status_pub.publish(message)

        self.publish_marker(status)
        self.log_status(status)

    def status_text(self) -> str:
        publisher_count = self.count_publishers(self.map_topic)
        subscriber_count = self.count_subscribers(self.map_topic)
        endpoint_summary = self.endpoint_summary()

        if self.last_map is None or self.last_map_monotonic is None:
            return (
                f'NO MAP on {self.map_topic}. '
                f'publishers={publisher_count}, subscribers={subscriber_count}. '
                f'{endpoint_summary}'
            )

        age = time.monotonic() - self.last_map_monotonic
        grid = self.last_map
        status = 'STALE' if age > self.stale_after_sec else 'OK'
        origin = grid.info.origin.position
        occupied, free, unknown = self.cell_counts(grid)

        return (
            f'{status} map age={age:.1f}s via {self.last_qos_label}; '
            f'size={grid.info.width}x{grid.info.height}, '
            f'res={grid.info.resolution:.3f}, frame={grid.header.frame_id or "<empty>"}, '
            f'origin=({origin.x:.2f},{origin.y:.2f}); '
            f'cells free={free}, occupied={occupied}, unknown={unknown}; '
            f'messages={self.map_messages}; '
            f'publishers={publisher_count}, subscribers={subscriber_count}. '
            f'{endpoint_summary}'
        )

    def endpoint_summary(self) -> str:
        publishers = self.get_publishers_info_by_topic(self.map_topic)
        if not publishers:
            return 'No /map publishers discovered.'

        descriptions = []
        for endpoint in publishers:
            qos = endpoint.qos_profile
            descriptions.append(
                f'pub={endpoint.node_name} '
                f'reliability={qos.reliability.name} '
                f'durability={qos.durability.name}'
            )
        return '; '.join(descriptions)

    @staticmethod
    def cell_counts(grid: OccupancyGrid) -> tuple[int, int, int]:
        occupied = 0
        free = 0
        unknown = 0
        for value in grid.data:
            if value < 0:
                unknown += 1
            elif value == 0:
                free += 1
            else:
                occupied += 1
        return occupied, free, unknown

    def publish_marker(self, text: str) -> None:
        marker_array = MarkerArray()
        marker = Marker()
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.header.frame_id = self.marker_frame()
        marker.ns = 'map_debug'
        marker.id = 1
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = self.marker_x()
        marker.pose.position.y = self.marker_y()
        marker.pose.position.z = 0.25
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.18
        marker.color.r = 0.0
        marker.color.g = 1.0 if text.startswith('OK') else 0.2
        marker.color.b = 0.0
        marker.color.a = 1.0
        marker.text = self.marker_text(text)
        marker_array.markers.append(marker)
        self.marker_pub.publish(marker_array)

    def marker_frame(self) -> str:
        if self.last_map is None or not self.last_map.header.frame_id:
            return 'map'
        return self.last_map.header.frame_id

    def marker_x(self) -> float:
        if self.last_map is None:
            return 0.0
        origin_x = self.last_map.info.origin.position.x
        width = self.last_map.info.width * self.last_map.info.resolution
        return origin_x + min(max(width * 0.5, 0.0), 1.0)

    def marker_y(self) -> float:
        if self.last_map is None:
            return 0.0
        origin_y = self.last_map.info.origin.position.y
        height = self.last_map.info.height * self.last_map.info.resolution
        return origin_y + min(max(height * 0.5, 0.0), 1.0)

    @staticmethod
    def marker_text(text: str) -> str:
        max_length = 220
        if len(text) <= max_length:
            return text
        return text[: max_length - 3] + '...'

    def log_status(self, status: str) -> None:
        if status.startswith('OK'):
            self.get_logger().info(status, throttle_duration_sec=5.0)
        else:
            self.get_logger().warn(status, throttle_duration_sec=2.0)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapDebugMonitor()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
