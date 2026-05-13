import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
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


@dataclass
class CircleFit:
    center_x: float
    center_y: float
    radius: float
    rmse: float


@dataclass
class LineFit:
    rmse: float


@dataclass
class LidarTrack:
    track_id: int
    center_x: float
    center_y: float
    radius: float
    first_seen_ns: int
    last_seen_ns: int
    points: List[Tuple[float, float]] = field(default_factory=list)
    view_bins: Set[int] = field(default_factory=set)
    observations: int = 0
    circle_rmse: float = math.inf
    line_rmse: float = math.inf
    confidence: float = 0.0


@dataclass
class PlanarTransform:
    x: float
    y: float
    cos_yaw: float
    sin_yaw: float


class LidarClusterDetector(Node):
    def __init__(self) -> None:
        super().__init__('lidar_cluster_detector')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('min_range', 0.25)
        self.declare_parameter('max_range', 3.0)
        self.declare_parameter('cluster_gap', 0.15)
        self.declare_parameter('min_cluster_points', 4)
        self.declare_parameter('min_cluster_width', 0.20)
        self.declare_parameter('max_cluster_width', 1.20)
        self.declare_parameter('require_curved_cluster', True)
        self.declare_parameter('min_cluster_arc_depth', 0.035)
        self.declare_parameter('min_cluster_range_depth', 0.055)
        self.declare_parameter('min_cluster_circle_radius', 0.08)
        self.declare_parameter('max_cluster_circle_radius', 0.80)
        self.declare_parameter('max_cluster_circle_fit_error', 0.06)
        self.declare_parameter('min_closest_point_fraction', 0.20)
        self.declare_parameter('max_closest_point_fraction', 0.80)
        self.declare_parameter('front_only', False)
        self.declare_parameter('front_angle_deg', 90.0)
        self.declare_parameter('transform_timeout_sec', 0.5)
        self.declare_parameter('use_latest_transform', True)
        self.declare_parameter('track_match_distance', 0.70)
        self.declare_parameter('track_timeout_sec', 12.0)
        self.declare_parameter('track_publish_timeout_sec', 1.0)
        self.declare_parameter('track_max_points', 360)
        self.declare_parameter('track_min_points', 24)
        self.declare_parameter('track_min_observations', 5)
        self.declare_parameter('track_min_view_bins', 2)
        self.declare_parameter('track_view_bin_deg', 45.0)
        self.declare_parameter('track_max_circle_rmse', 0.08)
        self.declare_parameter('track_min_line_circle_ratio', 1.25)
        self.declare_parameter('track_min_confidence', 0.65)

        self.scan_topic = self.get_parameter('scan_topic').value
        self.target_frame = self.get_parameter('target_frame').value
        self.tracks: List[LidarTrack] = []
        self.next_track_id = 1

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
        now_ns = self.get_clock().now().nanoseconds

        self.publish_candidate_markers(scan, candidates, None)
        self.prune_tracks(now_ns)

        scan_transform = self.lookup_scan_transform(scan)
        if scan_transform is None:
            self.publish_delete_selected_marker(scan.header.frame_id)
            return

        for candidate in candidates:
            map_points = self.transform_cluster_points(candidate, scan_transform)
            if map_points:
                self.update_track(map_points, scan_transform, now_ns)

        selected_track = self.best_confirmed_track(now_ns)
        if selected_track is None:
            self.publish_delete_selected_marker(self.target_frame)
            self.get_logger().info(
                'No /barrel_pose published: waiting for a persistent '
                'multi-scan circular LiDAR track.',
                throttle_duration_sec=3.0,
            )
            return

        selected_map_pose = self.make_map_pose(
            selected_track.center_x,
            selected_track.center_y,
        )
        selected_map_pose.pose.position.z = 0.0
        self.pose_pub.publish(selected_map_pose)
        self.publish_selected_marker(
            selected_map_pose,
            max(selected_track.radius * 2.0, 0.16),
        )

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

            if not self.is_single_scan_barrel_candidate(cluster):
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

    def is_single_scan_barrel_candidate(self, cluster: List[ScanPoint]) -> bool:
        if not bool(self.get_parameter('require_curved_cluster').value):
            return True

        range_depth = max(point.range_m for point in cluster) - min(
            point.range_m for point in cluster
        )
        min_range_depth = float(
            self.get_parameter('min_cluster_range_depth').value
        )
        if range_depth < min_range_depth:
            return False

        arc_depth = self.cluster_arc_depth(cluster)
        min_arc_depth = float(self.get_parameter('min_cluster_arc_depth').value)
        if arc_depth < min_arc_depth:
            return False

        closest_index = min(
            range(len(cluster)),
            key=lambda index: cluster[index].range_m,
        )
        closest_fraction = closest_index / max(len(cluster) - 1, 1)
        min_fraction = float(
            self.get_parameter('min_closest_point_fraction').value
        )
        max_fraction = float(
            self.get_parameter('max_closest_point_fraction').value
        )
        if closest_fraction < min_fraction or closest_fraction > max_fraction:
            return False

        circle_fit = self.fit_circle(cluster)
        if circle_fit is None:
            return False

        radius, fit_error = circle_fit
        min_radius = float(self.get_parameter('min_cluster_circle_radius').value)
        max_radius = float(self.get_parameter('max_cluster_circle_radius').value)
        max_fit_error = float(
            self.get_parameter('max_cluster_circle_fit_error').value
        )

        return (
            radius >= min_radius
            and radius <= max_radius
            and fit_error <= max_fit_error
        )

    def lookup_scan_transform(self, scan: LaserScan) -> Optional[PlanarTransform]:
        if scan.header.frame_id == self.target_frame:
            return PlanarTransform(0.0, 0.0, 1.0, 0.0)

        lookup_time = Time()
        if not bool(self.get_parameter('use_latest_transform').value):
            lookup_time = Time.from_msg(scan.header.stamp)

        timeout_sec = float(self.get_parameter('transform_timeout_sec').value)
        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                scan.header.frame_id,
                lookup_time,
                timeout=Duration(seconds=timeout_sec),
            )
        except TransformException as exc:
            self.get_logger().warn(
                f'No LiDAR track update: cannot transform scan from '
                f'{scan.header.frame_id} to {self.target_frame}: {exc}',
                throttle_duration_sec=2.0,
            )
            return None

        yaw = self.quaternion_to_yaw(transform.transform.rotation)
        return PlanarTransform(
            x=transform.transform.translation.x,
            y=transform.transform.translation.y,
            cos_yaw=math.cos(yaw),
            sin_yaw=math.sin(yaw),
        )

    @staticmethod
    def transform_cluster_points(
        candidate: ClusterCandidate,
        transform: PlanarTransform,
    ) -> List[Tuple[float, float]]:
        return [
            (
                transform.x
                + point.x * transform.cos_yaw
                - point.y * transform.sin_yaw,
                transform.y
                + point.x * transform.sin_yaw
                + point.y * transform.cos_yaw,
            )
            for point in candidate.points
        ]

    def update_track(
        self,
        map_points: List[Tuple[float, float]],
        scan_transform: PlanarTransform,
        now_ns: int,
    ) -> None:
        center_x = sum(point[0] for point in map_points) / len(map_points)
        center_y = sum(point[1] for point in map_points) / len(map_points)
        track = self.closest_track(center_x, center_y)

        if track is None:
            track = LidarTrack(
                track_id=self.next_track_id,
                center_x=center_x,
                center_y=center_y,
                radius=0.0,
                first_seen_ns=now_ns,
                last_seen_ns=now_ns,
            )
            self.next_track_id += 1
            self.tracks.append(track)

        track.points.extend(map_points)
        max_points = int(self.get_parameter('track_max_points').value)
        if max_points > 0 and len(track.points) > max_points:
            del track.points[:len(track.points) - max_points]

        track.observations += 1
        track.last_seen_ns = now_ns
        track.view_bins.add(
            self.view_bin(scan_transform.x, scan_transform.y, center_x, center_y)
        )
        self.recompute_track_shape(track)

    def closest_track(self, center_x: float, center_y: float) -> Optional[LidarTrack]:
        if not self.tracks:
            return None

        match_distance = float(self.get_parameter('track_match_distance').value)
        closest = min(
            self.tracks,
            key=lambda track: math.hypot(track.center_x - center_x, track.center_y - center_y),
        )
        if math.hypot(closest.center_x - center_x, closest.center_y - center_y) > match_distance:
            return None

        return closest

    def view_bin(
        self,
        sensor_x: float,
        sensor_y: float,
        center_x: float,
        center_y: float,
    ) -> int:
        bin_size = math.radians(float(self.get_parameter('track_view_bin_deg').value))
        if bin_size <= 0.0:
            bin_size = math.radians(45.0)

        angle = math.atan2(sensor_y - center_y, sensor_x - center_x)
        return int(math.floor((angle + math.pi) / bin_size))

    def recompute_track_shape(self, track: LidarTrack) -> None:
        circle = self.fit_circle_points(track.points)
        line = self.fit_line_points(track.points)
        if circle is None or line is None:
            track.confidence = 0.0
            return

        track.center_x = circle.center_x
        track.center_y = circle.center_y
        track.radius = circle.radius
        track.circle_rmse = circle.rmse
        track.line_rmse = line.rmse

        max_circle_rmse = float(self.get_parameter('track_max_circle_rmse').value)
        min_line_ratio = float(
            self.get_parameter('track_min_line_circle_ratio').value
        )
        min_observations = max(
            int(self.get_parameter('track_min_observations').value),
            1,
        )
        min_view_bins = max(int(self.get_parameter('track_min_view_bins').value), 1)

        circle_score = 1.0 - min(circle.rmse / max(max_circle_rmse, 1e-6), 1.0)
        line_ratio = line.rmse / max(circle.rmse, 0.005)
        line_score = min(max((line_ratio - 1.0) / max(min_line_ratio - 1.0, 1e-6), 0.0), 1.0)
        observation_score = min(track.observations / min_observations, 1.0)
        view_score = min(len(track.view_bins) / min_view_bins, 1.0)
        track.confidence = (
            0.35 * circle_score
            + 0.30 * line_score
            + 0.20 * observation_score
            + 0.15 * view_score
        )

    def best_confirmed_track(self, now_ns: int) -> Optional[LidarTrack]:
        publish_timeout = Duration(
            seconds=float(self.get_parameter('track_publish_timeout_sec').value)
        ).nanoseconds
        confirmed_tracks = [
            track
            for track in self.tracks
            if now_ns - track.last_seen_ns <= publish_timeout
            and self.is_confirmed_track(track)
        ]
        if not confirmed_tracks:
            return None

        return max(
            confirmed_tracks,
            key=lambda track: (track.confidence, track.observations),
        )

    def is_confirmed_track(self, track: LidarTrack) -> bool:
        min_points = int(self.get_parameter('track_min_points').value)
        min_observations = int(self.get_parameter('track_min_observations').value)
        min_view_bins = int(self.get_parameter('track_min_view_bins').value)
        max_circle_rmse = float(self.get_parameter('track_max_circle_rmse').value)
        min_line_ratio = float(
            self.get_parameter('track_min_line_circle_ratio').value
        )
        min_confidence = float(self.get_parameter('track_min_confidence').value)
        min_radius = float(self.get_parameter('min_cluster_circle_radius').value)
        max_radius = float(self.get_parameter('max_cluster_circle_radius').value)
        line_ratio = track.line_rmse / max(track.circle_rmse, 0.005)

        return (
            len(track.points) >= max(min_points, 3)
            and track.observations >= max(min_observations, 1)
            and len(track.view_bins) >= max(min_view_bins, 1)
            and track.radius >= min_radius
            and track.radius <= max_radius
            and track.circle_rmse <= max_circle_rmse
            and line_ratio >= min_line_ratio
            and track.confidence >= min_confidence
        )

    def prune_tracks(self, now_ns: int) -> None:
        timeout_ns = Duration(
            seconds=float(self.get_parameter('track_timeout_sec').value)
        ).nanoseconds
        self.tracks = [
            track
            for track in self.tracks
            if now_ns - track.last_seen_ns <= timeout_ns
        ]

    @staticmethod
    def cluster_arc_depth(cluster: List[ScanPoint]) -> float:
        first = cluster[0]
        last = cluster[-1]
        line_dx = last.x - first.x
        line_dy = last.y - first.y
        line_length = math.hypot(line_dx, line_dy)
        if line_length <= 1e-6:
            return 0.0

        return max(
            abs(
                line_dy * point.x
                - line_dx * point.y
                + last.x * first.y
                - last.y * first.x
            )
            / line_length
            for point in cluster[1:-1]
        ) if len(cluster) > 2 else 0.0

    @staticmethod
    def fit_circle(cluster: List[ScanPoint]) -> Optional[Tuple[float, float]]:
        circle = LidarClusterDetector.fit_circle_points(
            [(point.x, point.y) for point in cluster]
        )
        if circle is None:
            return None
        return circle.radius, circle.rmse

    @staticmethod
    def fit_circle_points(points: List[Tuple[float, float]]) -> Optional[CircleFit]:
        if len(points) < 3:
            return None

        normal = [[0.0 for _ in range(4)] for _ in range(3)]
        for x, y in points:
            row = [x, y, 1.0]
            rhs = -(x * x + y * y)
            for row_index in range(3):
                for col_index in range(3):
                    normal[row_index][col_index] += row[row_index] * row[col_index]
                normal[row_index][3] += row[row_index] * rhs

        solution = LidarClusterDetector.solve_3x3(normal)
        if solution is None:
            return None

        circle_a, circle_b, circle_c = solution
        center_x = -0.5 * circle_a
        center_y = -0.5 * circle_b
        radius_squared = center_x * center_x + center_y * center_y - circle_c
        if radius_squared <= 0.0:
            return None

        radius = math.sqrt(radius_squared)
        rmse = math.sqrt(
            sum(
                (
                    math.hypot(x - center_x, y - center_y)
                    - radius
                ) ** 2
                for x, y in points
            )
            / len(points)
        )
        return CircleFit(center_x, center_y, radius, rmse)

    @staticmethod
    def fit_line_points(points: List[Tuple[float, float]]) -> Optional[LineFit]:
        if len(points) < 2:
            return None

        mean_x = sum(point[0] for point in points) / len(points)
        mean_y = sum(point[1] for point in points) / len(points)
        var_x = sum((point[0] - mean_x) ** 2 for point in points) / len(points)
        var_y = sum((point[1] - mean_y) ** 2 for point in points) / len(points)
        cov_xy = (
            sum((point[0] - mean_x) * (point[1] - mean_y) for point in points)
            / len(points)
        )

        trace = var_x + var_y
        discriminant = math.sqrt((var_x - var_y) ** 2 + 4.0 * cov_xy * cov_xy)
        minor_variance = 0.5 * (trace - discriminant)
        return LineFit(math.sqrt(max(minor_variance, 0.0)))

    @staticmethod
    def solve_3x3(matrix: List[List[float]]) -> Optional[Tuple[float, float, float]]:
        for pivot_index in range(3):
            pivot_row = max(
                range(pivot_index, 3),
                key=lambda row_index: abs(matrix[row_index][pivot_index]),
            )
            if abs(matrix[pivot_row][pivot_index]) <= 1e-9:
                return None

            if pivot_row != pivot_index:
                matrix[pivot_index], matrix[pivot_row] = (
                    matrix[pivot_row],
                    matrix[pivot_index],
                )

            pivot = matrix[pivot_index][pivot_index]
            for col_index in range(pivot_index, 4):
                matrix[pivot_index][col_index] /= pivot

            for row_index in range(3):
                if row_index == pivot_index:
                    continue

                factor = matrix[row_index][pivot_index]
                for col_index in range(pivot_index, 4):
                    matrix[row_index][col_index] -= (
                        factor * matrix[pivot_index][col_index]
                    )

        return matrix[0][3], matrix[1][3], matrix[2][3]

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

    def make_map_pose(self, x: float, y: float) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    @staticmethod
    def quaternion_to_yaw(orientation) -> float:
        siny_cosp = 2.0 * (
            orientation.w * orientation.z + orientation.x * orientation.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            orientation.y * orientation.y + orientation.z * orientation.z
        )
        return math.atan2(siny_cosp, cosy_cosp)

    def transform_pose(self, pose: PoseStamped, target_frame: str) -> Optional[PoseStamped]:
        if pose.header.frame_id == target_frame:
            return pose

        pose_for_transform = pose
        if bool(self.get_parameter('use_latest_transform').value):
            pose_for_transform = PoseStamped()
            pose_for_transform.header.frame_id = pose.header.frame_id
            pose_for_transform.header.stamp = Time().to_msg()
            pose_for_transform.pose = pose.pose

        timeout_sec = float(self.get_parameter('transform_timeout_sec').value)
        try:
            return self.tf_buffer.transform(
                pose_for_transform,
                target_frame,
                timeout=Duration(seconds=timeout_sec),
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
