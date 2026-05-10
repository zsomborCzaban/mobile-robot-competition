import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import LaserScan
import tf2_geometry_msgs  # noqa: F401  Registers geometry_msgs transforms with tf2.
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class ScanPoint:
    x: float
    y: float
    range_m: float
    angle: float


@dataclass
class ClusterCandidate:
    points: List[ScanPoint]
    center_x: float
    center_y: float
    width: float
    distance: float


class LidarClusterDetector(Node):
    def __init__(self) -> None:
        super().__init__('lidar_cluster_detector')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('min_range', 0.25)
        self.declare_parameter('max_range', 3.0)
        self.declare_parameter('cluster_gap', 0.15)
        self.declare_parameter('min_cluster_points', 4)
        self.declare_parameter('min_cluster_width', 0.40)
        self.declare_parameter('max_cluster_width', 1.00)
        self.declare_parameter('front_only', False)
        self.declare_parameter('front_angle_deg', 90.0)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.candidate_markers_pub = self.create_publisher(
            MarkerArray,
            '/barrel_candidate_markers',
            10,
        )
        self.selected_marker_pub = self.create_publisher(
            Marker,
            '/barrel_marker',
            10,
        )
        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/barrel_pose',
            10,
        )

        self.get_logger().info(
            f'LiDAR cluster detector listening on {self.scan_topic}, '
            f'target frame: {self.target_frame}'
        )

    def scan_callback(self, scan: LaserScan) -> None:
        points = self.scan_to_points(scan)
        clusters = self.cluster_points(points)
        candidates = self.filter_clusters(clusters)

        selected = min(candidates, key=lambda c: c.distance) if candidates else None
        self.publish_candidate_markers(scan, candidates, selected)

        if selected is None:
            self.publish_delete_selected_marker(scan.header.frame_id)
            return

        selected_pose = self.make_pose(scan, selected.center_x, selected.center_y)
        selected_map_pose = self.transform_pose(selected_pose, self.target_frame)

        if selected_map_pose is None:
            self.publish_delete_selected_marker(scan.header.frame_id)
            return

        selected_map_pose.pose.position.z = 0.0
        selected_map_pose.pose.orientation.x = 0.0
        selected_map_pose.pose.orientation.y = 0.0
        selected_map_pose.pose.orientation.z = 0.0
        selected_map_pose.pose.orientation.w = 1.0
        self.pose_pub.publish(selected_map_pose)
        self.publish_selected_marker(selected_map_pose, selected.width)

    def scan_to_points(self, scan: LaserScan) -> List[ScanPoint]:
        min_range = float(self.get_parameter('min_range').value)
        max_range = float(self.get_parameter('max_range').value)
        front_only = bool(self.get_parameter('front_only').value)
        front_angle_rad = math.radians(float(self.get_parameter('front_angle_deg').value))
        front_half_angle = front_angle_rad * 0.5

        valid_min = max(min_range, scan.range_min)
        valid_max = min(max_range, scan.range_max)
        points: List[ScanPoint] = []

        for index, range_m in enumerate(scan.ranges):
            if not math.isfinite(range_m):
                points.append(self.invalid_point())
                continue

            angle = scan.angle_min + index * scan.angle_increment
            normalized_angle = math.atan2(math.sin(angle), math.cos(angle))

            if range_m < valid_min or range_m > valid_max:
                points.append(self.invalid_point())
                continue

            if front_only and abs(normalized_angle) > front_half_angle:
                points.append(self.invalid_point())
                continue

            points.append(
                ScanPoint(
                    x=range_m * math.cos(angle),
                    y=range_m * math.sin(angle),
                    range_m=range_m,
                    angle=normalized_angle,
                )
            )

        return points

    @staticmethod
    def invalid_point() -> ScanPoint:
        return ScanPoint(
            x=math.nan,
            y=math.nan,
            range_m=math.nan,
            angle=math.nan,
        )

    def cluster_points(self, points: List[ScanPoint]) -> List[List[ScanPoint]]:
        cluster_gap = float(self.get_parameter('cluster_gap').value)
        clusters: List[List[ScanPoint]] = []
        current_cluster: List[ScanPoint] = []
        previous_point: Optional[ScanPoint] = None

        for point in points:
            if not math.isfinite(point.range_m):
                if current_cluster:
                    clusters.append(current_cluster)
                    current_cluster = []
                previous_point = None
                continue

            if previous_point is None:
                current_cluster = [point]
            else:
                gap = math.hypot(point.x - previous_point.x, point.y - previous_point.y)
                if gap <= cluster_gap:
                    current_cluster.append(point)
                else:
                    if current_cluster:
                        clusters.append(current_cluster)
                    current_cluster = [point]

            previous_point = point

        if current_cluster:
            clusters.append(current_cluster)

        return clusters

    def filter_clusters(self, clusters: List[List[ScanPoint]]) -> List[ClusterCandidate]:
        min_points = int(self.get_parameter('min_cluster_points').value)
        min_width = float(self.get_parameter('min_cluster_width').value)
        max_width = float(self.get_parameter('max_cluster_width').value)
        candidates: List[ClusterCandidate] = []

        for cluster in clusters:
            if len(cluster) < min_points:
                continue

            width = self.cluster_width(cluster)
            if width < min_width or width > max_width:
                continue

            center_x, center_y = self.cluster_center(cluster)
            distance = math.hypot(center_x, center_y)
            candidates.append(
                ClusterCandidate(
                    points=cluster,
                    center_x=center_x,
                    center_y=center_y,
                    width=width,
                    distance=distance,
                )
            )

        return candidates

    @staticmethod
    def cluster_width(cluster: List[ScanPoint]) -> float:
        first = cluster[0]
        last = cluster[-1]
        return math.hypot(last.x - first.x, last.y - first.y)

    @staticmethod
    def cluster_center(cluster: List[ScanPoint]) -> Tuple[float, float]:
        center_x = sum(point.x for point in cluster) / len(cluster)
        center_y = sum(point.y for point in cluster) / len(cluster)
        return center_x, center_y

    @staticmethod
    def make_pose(scan: LaserScan, x: float, y: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header = scan.header
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    def transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        try:
            return self.tf_buffer.transform(
                pose,
                target_frame,
                timeout=Duration(seconds=0.2),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'Could not transform candidate from {pose.header.frame_id} '
                f'to {target_frame}: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

    def publish_candidate_markers(
        self,
        scan: LaserScan,
        candidates: List[ClusterCandidate],
        selected: Optional[ClusterCandidate],
    ) -> None:
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = scan.header.frame_id
        delete_all.header.stamp = scan.header.stamp
        delete_all.ns = 'barrel_candidates'
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for marker_id, candidate in enumerate(candidates):
            pose = self.make_pose(scan, candidate.center_x, candidate.center_y)
            marker_pose = self.transform_pose(pose, self.target_frame)
            if marker_pose is None:
                marker_pose = pose

            marker = Marker()
            marker.header = marker_pose.header
            marker.ns = 'barrel_candidates'
            marker.id = marker_id
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose = marker_pose.pose
            marker.pose.position.z = 0.12
            marker.scale.x = max(candidate.width, 0.12)
            marker.scale.y = max(candidate.width, 0.12)
            marker.scale.z = 0.24
            marker.color.a = 0.75

            if candidate is selected:
                marker.color.r = 1.0
                marker.color.g = 0.55
                marker.color.b = 0.0
            else:
                marker.color.r = 0.0
                marker.color.g = 0.8
                marker.color.b = 0.35

            marker.lifetime = Duration(seconds=0.5).to_msg()
            marker_array.markers.append(marker)

        self.candidate_markers_pub.publish(marker_array)

    def publish_selected_marker(self, pose: PoseStamped, width: float) -> None:
        marker = Marker()
        marker.header = pose.header
        marker.ns = 'barrel_selected'
        marker.id = 0
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose = pose.pose
        marker.pose.position.z = 0.25
        marker.scale.x = max(width, 0.16)
        marker.scale.y = max(width, 0.16)
        marker.scale.z = 0.5
        marker.color.r = 1.0
        marker.color.g = 0.25
        marker.color.b = 0.0
        marker.color.a = 0.9
        marker.lifetime = Duration(seconds=0.5).to_msg()
        self.selected_marker_pub.publish(marker)

    def publish_delete_selected_marker(self, frame_id: str) -> None:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'barrel_selected'
        marker.id = 0
        marker.action = Marker.DELETE
        self.selected_marker_pub.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LidarClusterDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
