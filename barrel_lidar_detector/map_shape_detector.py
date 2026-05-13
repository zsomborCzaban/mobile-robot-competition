import math
import os
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import yaml

import rclpy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration
from rclpy.node import Node
import tf2_geometry_msgs  # noqa: F401  Registers geometry_msgs transforms with tf2.
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray


@dataclass
class MapCandidate:
    center_x: float
    center_y: float
    diameter: float
    width_x: float
    width_y: float
    occupied_cells: int
    roundness: float
    fill_ratio: float
    score: float


@dataclass
class ConfirmationTrack:
    x: float
    y: float
    width: float
    roundness: float
    score: float
    confirmations: int
    last_seen_ns: int
    last_yaml_write_ns: int = 0


class MapShapeDetector(Node):
    def __init__(self) -> None:
        super().__init__('map_shape_detector')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('lidar_pose_topic', '/barrel_pose')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('occupied_threshold', 65)
        self.declare_parameter('min_blob_cells', 4)
        self.declare_parameter('min_blob_diameter', 0.30)
        self.declare_parameter('max_blob_diameter', 1.20)
        self.declare_parameter('min_roundness', 0.45)
        self.declare_parameter('min_minor_major_ratio', 0.45)
        self.declare_parameter('confirm_distance', 0.65)
        self.declare_parameter('fallback_confirm_distance', 0.85)
        self.declare_parameter('lidar_pose_timeout_sec', 3.0)
        self.declare_parameter('lidar_weight', 0.70)
        self.declare_parameter('min_confirmed_roundness', 0.55)
        self.declare_parameter('min_confirmed_minor_major_ratio', 0.55)
        self.declare_parameter('min_confirmed_score', 0.50)
        self.declare_parameter('stable_confirmations', 2)
        self.declare_parameter('confirmation_track_distance', 0.50)
        self.declare_parameter('confirmation_track_timeout_sec', 8.0)
        self.declare_parameter('write_barrel_yaml', True)
        self.declare_parameter(
            'barrel_yaml_path',
            '~/turtlebot4_ws/barrel_target.yaml',
        )
        self.declare_parameter('yaml_merge_distance', 0.45)
        self.declare_parameter('yaml_write_period_sec', 1.0)

        self.map_topic = self.get_parameter('map_topic').value
        self.lidar_pose_topic = self.get_parameter('lidar_pose_topic').value
        self.target_frame = self.get_parameter('target_frame').value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.map_candidates: List[MapCandidate] = []
        self.latest_lidar_pose: Optional[PoseStamped] = None
        self.latest_lidar_time = None
        self.confirmation_tracks: List[ConfirmationTrack] = []

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            10,
        )
        self.lidar_pose_sub = self.create_subscription(
            PoseStamped,
            self.lidar_pose_topic,
            self.lidar_pose_callback,
            10,
        )

        self.map_candidates_pub = self.create_publisher(
            MarkerArray,
            '/barrel_map_candidate_markers',
            10,
        )
        self.map_marker_pub = self.create_publisher(
            Marker,
            '/barrel_map_marker',
            10,
        )
        self.map_pose_pub = self.create_publisher(
            PoseStamped,
            '/barrel_map_pose',
            10,
        )
        self.confirmed_marker_pub = self.create_publisher(
            Marker,
            '/barrel_confirmed_marker',
            10,
        )
        self.confirmed_pose_pub = self.create_publisher(
            PoseStamped,
            '/barrel_confirmed_pose',
            10,
        )

        self.get_logger().info(
            f'Map shape detector listening on {self.map_topic}, '
            f'fusing with {self.lidar_pose_topic}. Barrel YAML: '
            f'{self.barrel_yaml_path()}'
        )

    def map_callback(self, grid: OccupancyGrid) -> None:
        self.map_candidates = self.detect_map_candidates(grid)
        self.publish_map_candidate_markers(grid.header.stamp)
        self.publish_best_map_candidate(grid.header.stamp)
        self.publish_confirmed_candidate()

    def lidar_pose_callback(self, pose: PoseStamped) -> None:
        pose_in_target_frame = self.transform_pose(pose, self.target_frame)
        if pose_in_target_frame is None:
            return

        self.latest_lidar_pose = pose_in_target_frame
        self.latest_lidar_time = self.get_clock().now()
        self.publish_confirmed_candidate()

    def detect_map_candidates(self, grid: OccupancyGrid) -> List[MapCandidate]:
        width = grid.info.width
        height = grid.info.height
        if width == 0 or height == 0:
            return []

        occupied_threshold = int(self.get_parameter('occupied_threshold').value)
        visited = bytearray(width * height)
        candidates: List[MapCandidate] = []

        for index, value in enumerate(grid.data):
            if visited[index] or value < occupied_threshold:
                continue

            component = self.collect_component(index, grid, visited, occupied_threshold)
            candidate = self.component_to_candidate(component, grid)
            if candidate is not None:
                candidates.append(candidate)

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return candidates

    def collect_component(
        self,
        start_index: int,
        grid: OccupancyGrid,
        visited: bytearray,
        occupied_threshold: int,
    ) -> List[Tuple[int, int]]:
        width = grid.info.width
        height = grid.info.height
        queue = deque([start_index])
        visited[start_index] = 1
        component: List[Tuple[int, int]] = []

        while queue:
            index = queue.popleft()
            x = index % width
            y = index // width
            component.append((x, y))

            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue

                    nx = x + dx
                    ny = y + dy
                    if nx < 0 or nx >= width or ny < 0 or ny >= height:
                        continue

                    neighbor_index = ny * width + nx
                    if visited[neighbor_index]:
                        continue

                    if grid.data[neighbor_index] >= occupied_threshold:
                        visited[neighbor_index] = 1
                        queue.append(neighbor_index)

        return component

    def component_to_candidate(
        self,
        component: List[Tuple[int, int]],
        grid: OccupancyGrid,
    ) -> Optional[MapCandidate]:
        min_cells = int(self.get_parameter('min_blob_cells').value)
        if len(component) < min_cells:
            return None

        resolution = grid.info.resolution
        min_blob_diameter = float(self.get_parameter('min_blob_diameter').value)
        max_blob_diameter = float(self.get_parameter('max_blob_diameter').value)
        min_roundness = float(self.get_parameter('min_roundness').value)
        min_minor_major_ratio = float(
            self.get_parameter('min_minor_major_ratio').value
        )

        xs = [cell[0] for cell in component]
        ys = [cell[1] for cell in component]
        width_x = (max(xs) - min(xs) + 1) * resolution
        width_y = (max(ys) - min(ys) + 1) * resolution
        diameter = max(width_x, width_y)
        minor_diameter = min(width_x, width_y)

        if diameter < min_blob_diameter or diameter > max_blob_diameter:
            return None

        minor_major_ratio = minor_diameter / diameter if diameter > 0.0 else 0.0
        if minor_major_ratio < min_minor_major_ratio:
            return None

        roundness = self.component_roundness(component)
        if roundness < min_roundness:
            return None

        center_cell_x = sum(xs) / len(xs) + 0.5
        center_cell_y = sum(ys) / len(ys) + 0.5
        center_x, center_y = self.cell_to_world(center_cell_x, center_cell_y, grid)

        occupied_area = len(component) * resolution * resolution
        expected_circle_area = math.pi * (diameter * 0.5) ** 2
        fill_ratio = (
            min(occupied_area / expected_circle_area, 1.0)
            if expected_circle_area > 0.0 else 0.0
        )

        size_score = self.size_score(diameter, min_blob_diameter, max_blob_diameter)
        score = 0.60 * roundness + 0.25 * size_score + 0.15 * fill_ratio

        return MapCandidate(
            center_x=center_x,
            center_y=center_y,
            diameter=diameter,
            width_x=width_x,
            width_y=width_y,
            occupied_cells=len(component),
            roundness=roundness,
            fill_ratio=fill_ratio,
            score=score,
        )

    @staticmethod
    def component_roundness(component: List[Tuple[int, int]]) -> float:
        if len(component) < 2:
            return 0.0

        mean_x = sum(cell[0] for cell in component) / len(component)
        mean_y = sum(cell[1] for cell in component) / len(component)
        var_x = sum((cell[0] - mean_x) ** 2 for cell in component) / len(component)
        var_y = sum((cell[1] - mean_y) ** 2 for cell in component) / len(component)
        cov_xy = (
            sum((cell[0] - mean_x) * (cell[1] - mean_y) for cell in component)
            / len(component)
        )

        trace = var_x + var_y
        discriminant = math.sqrt((var_x - var_y) ** 2 + 4.0 * cov_xy * cov_xy)
        major_variance = 0.5 * (trace + discriminant)
        minor_variance = 0.5 * (trace - discriminant)

        if major_variance <= 1e-9:
            return 1.0

        return math.sqrt(max(minor_variance, 0.0) / major_variance)

    @staticmethod
    def size_score(diameter: float, minimum: float, maximum: float) -> float:
        midpoint = 0.5 * (minimum + maximum)
        half_range = 0.5 * (maximum - minimum)
        if half_range <= 0.0:
            return 1.0

        return max(0.0, 1.0 - abs(diameter - midpoint) / half_range)

    @staticmethod
    def cell_to_world(
        cell_x: float,
        cell_y: float,
        grid: OccupancyGrid,
    ) -> Tuple[float, float]:
        resolution = grid.info.resolution
        origin = grid.info.origin
        local_x = cell_x * resolution
        local_y = cell_y * resolution
        yaw = MapShapeDetector.quaternion_to_yaw(origin.orientation)
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)

        world_x = origin.position.x + local_x * cos_yaw - local_y * sin_yaw
        world_y = origin.position.y + local_x * sin_yaw + local_y * cos_yaw
        return world_x, world_y

    @staticmethod
    def quaternion_to_yaw(orientation) -> float:
        siny_cosp = 2.0 * (
            orientation.w * orientation.z + orientation.x * orientation.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            orientation.y * orientation.y + orientation.z * orientation.z
        )
        return math.atan2(siny_cosp, cosy_cosp)

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

    def publish_map_candidate_markers(self, stamp) -> None:
        marker_array = MarkerArray()

        delete_all = Marker()
        delete_all.header.frame_id = self.target_frame
        delete_all.header.stamp = stamp
        delete_all.ns = 'barrel_map_candidates'
        delete_all.action = Marker.DELETEALL
        marker_array.markers.append(delete_all)

        for marker_id, candidate in enumerate(self.map_candidates):
            marker = self.make_cylinder_marker(
                frame_id=self.target_frame,
                stamp=stamp,
                namespace='barrel_map_candidates',
                marker_id=marker_id,
                candidate=candidate,
                height=0.18,
            )
            marker.pose.position.z = 0.09
            marker.color.r = 0.1
            marker.color.g = 0.45
            marker.color.b = 1.0
            marker.color.a = 0.55
            marker.lifetime = Duration(seconds=2.0).to_msg()
            marker_array.markers.append(marker)

        self.map_candidates_pub.publish(marker_array)

    def publish_best_map_candidate(self, stamp) -> None:
        if not self.map_candidates:
            self.publish_delete_marker(self.map_marker_pub, 'barrel_map_selected')
            return

        best = self.map_candidates[0]
        pose = self.make_pose(best.center_x, best.center_y, stamp)
        self.map_pose_pub.publish(pose)

        marker = self.make_cylinder_marker(
            frame_id=self.target_frame,
            stamp=stamp,
            namespace='barrel_map_selected',
            marker_id=0,
            candidate=best,
            height=0.35,
        )
        marker.pose.position.z = 0.18
        marker.color.r = 0.0
        marker.color.g = 0.25
        marker.color.b = 1.0
        marker.color.a = 0.8
        marker.lifetime = Duration(seconds=2.0).to_msg()
        self.map_marker_pub.publish(marker)

    def publish_confirmed_candidate(self) -> None:
        lidar_pose = self.valid_lidar_pose()
        if lidar_pose is None:
            self.get_logger().info(
                'No magenta marker: waiting for a recent /barrel_pose '
                'from the LiDAR detector.',
                throttle_duration_sec=3.0,
            )
            self.publish_delete_marker(self.confirmed_marker_pub, 'barrel_confirmed')
            return

        if not self.map_candidates:
            self.get_logger().info(
                'No magenta marker: no round map candidates found in /map.',
                throttle_duration_sec=3.0,
            )
            self.publish_delete_marker(self.confirmed_marker_pub, 'barrel_confirmed')
            return

        matched_candidate = self.closest_map_candidate(lidar_pose)
        if matched_candidate is None:
            self.publish_delete_marker(self.confirmed_marker_pub, 'barrel_confirmed')
            return

        stamp = self.get_clock().now().to_msg()
        fused_pose = self.fuse_pose(lidar_pose, matched_candidate, stamp)
        stable_track = self.update_confirmation_track(fused_pose, matched_candidate)
        if stable_track is None:
            self.publish_delete_marker(self.confirmed_marker_pub, 'barrel_confirmed')
            return

        stable_pose = self.make_pose(stable_track.x, stable_track.y, stamp)
        self.confirmed_pose_pub.publish(stable_pose)
        self.update_barrel_yaml(stable_track)

        marker = self.make_cylinder_marker(
            frame_id=self.target_frame,
            stamp=stamp,
            namespace='barrel_confirmed',
            marker_id=0,
            candidate=matched_candidate,
            height=0.55,
        )
        marker.pose = stable_pose.pose
        marker.pose.position.z = 0.28
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 1.0
        marker.color.a = 0.9
        marker.lifetime = Duration(seconds=2.0).to_msg()
        self.confirmed_marker_pub.publish(marker)

    def valid_lidar_pose(self) -> Optional[PoseStamped]:
        if self.latest_lidar_pose is None or self.latest_lidar_time is None:
            return None

        timeout_sec = float(self.get_parameter('lidar_pose_timeout_sec').value)
        age = self.get_clock().now() - self.latest_lidar_time
        if age.nanoseconds > Duration(seconds=timeout_sec).nanoseconds:
            return None

        return self.latest_lidar_pose

    def closest_map_candidate(self, lidar_pose: PoseStamped) -> Optional[MapCandidate]:
        confirm_distance = float(self.get_parameter('confirm_distance').value)
        fallback_confirm_distance = float(
            self.get_parameter('fallback_confirm_distance').value
        )
        lidar_x = lidar_pose.pose.position.x
        lidar_y = lidar_pose.pose.position.y

        closest_any = min(
            self.map_candidates,
            key=lambda candidate: math.hypot(
                candidate.center_x - lidar_x,
                candidate.center_y - lidar_y,
            ),
        )
        closest_any_distance = math.hypot(
            closest_any.center_x - lidar_x,
            closest_any.center_y - lidar_y,
        )

        eligible_candidates = [
            candidate
            for candidate in self.map_candidates
            if self.is_confirmable_candidate(candidate)
        ]
        if eligible_candidates:
            closest = min(
                eligible_candidates,
                key=lambda candidate: math.hypot(
                    candidate.center_x - lidar_x,
                    candidate.center_y - lidar_y,
                ),
            )
            distance = math.hypot(closest.center_x - lidar_x, closest.center_y - lidar_y)

            if distance <= confirm_distance:
                return closest

            self.get_logger().info(
                'No magenta marker: closest high-quality map candidate is '
                f'{distance:.2f} m from LiDAR pose, limit is '
                f'{confirm_distance:.2f} m.',
                throttle_duration_sec=3.0,
            )

        if closest_any_distance <= fallback_confirm_distance:
            self.get_logger().info(
                'Using spatial fallback for magenta confirmation: closest map '
                f'candidate is {closest_any_distance:.2f} m from LiDAR pose.',
                throttle_duration_sec=3.0,
            )
            return closest_any

        if not eligible_candidates:
            self.get_logger().info(
                'No magenta marker: map candidates exist, but none pass the '
                'extra roundness/score confirmation gate. Closest candidate is '
                f'{closest_any_distance:.2f} m from LiDAR pose.',
                throttle_duration_sec=3.0,
            )
        else:
            self.get_logger().info(
                'No magenta marker: closest map candidate is '
                f'{closest_any_distance:.2f} m from LiDAR pose, fallback limit '
                f'is {fallback_confirm_distance:.2f} m.',
                throttle_duration_sec=3.0,
            )

        return None

    def is_confirmable_candidate(self, candidate: MapCandidate) -> bool:
        min_roundness = float(self.get_parameter('min_confirmed_roundness').value)
        min_score = float(self.get_parameter('min_confirmed_score').value)
        min_minor_major_ratio = float(
            self.get_parameter('min_confirmed_minor_major_ratio').value
        )

        major = max(candidate.width_x, candidate.width_y)
        minor = min(candidate.width_x, candidate.width_y)
        minor_major_ratio = minor / major if major > 0.0 else 0.0

        return (
            candidate.roundness >= min_roundness
            and candidate.score >= min_score
            and minor_major_ratio >= min_minor_major_ratio
        )

    def update_confirmation_track(
        self,
        pose: PoseStamped,
        candidate: MapCandidate,
    ) -> Optional[ConfirmationTrack]:
        now_ns = self.get_clock().now().nanoseconds
        self.prune_confirmation_tracks(now_ns)

        track_distance = float(
            self.get_parameter('confirmation_track_distance').value
        )
        pose_x = pose.pose.position.x
        pose_y = pose.pose.position.y

        matching_track = None
        if self.confirmation_tracks:
            closest = min(
                self.confirmation_tracks,
                key=lambda track: math.hypot(track.x - pose_x, track.y - pose_y),
            )
            if math.hypot(closest.x - pose_x, closest.y - pose_y) <= track_distance:
                matching_track = closest

        if matching_track is None:
            matching_track = ConfirmationTrack(
                x=pose_x,
                y=pose_y,
                width=candidate.diameter,
                roundness=candidate.roundness,
                score=candidate.score,
                confirmations=1,
                last_seen_ns=now_ns,
            )
            self.confirmation_tracks.append(matching_track)
        else:
            next_count = matching_track.confirmations + 1
            alpha = 1.0 / min(next_count, 8)
            matching_track.x = self.blend(matching_track.x, pose_x, alpha)
            matching_track.y = self.blend(matching_track.y, pose_y, alpha)
            matching_track.width = self.blend(
                matching_track.width,
                candidate.diameter,
                alpha,
            )
            matching_track.roundness = self.blend(
                matching_track.roundness,
                candidate.roundness,
                alpha,
            )
            matching_track.score = self.blend(
                matching_track.score,
                candidate.score,
                alpha,
            )
            matching_track.confirmations = next_count
            matching_track.last_seen_ns = now_ns

        stable_confirmations = int(self.get_parameter('stable_confirmations').value)
        if matching_track.confirmations < max(stable_confirmations, 1):
            self.get_logger().info(
                'No magenta marker yet: candidate awaiting stability '
                f'{matching_track.confirmations}/{stable_confirmations}.',
                throttle_duration_sec=3.0,
            )
            return None

        return matching_track

    def prune_confirmation_tracks(self, now_ns: int) -> None:
        timeout_sec = float(
            self.get_parameter('confirmation_track_timeout_sec').value
        )
        timeout_ns = Duration(seconds=timeout_sec).nanoseconds
        self.confirmation_tracks = [
            track
            for track in self.confirmation_tracks
            if now_ns - track.last_seen_ns <= timeout_ns
        ]

    @staticmethod
    def blend(old_value: float, new_value: float, alpha: float) -> float:
        return (1.0 - alpha) * old_value + alpha * new_value

    def update_barrel_yaml(self, track: ConfirmationTrack) -> None:
        if not bool(self.get_parameter('write_barrel_yaml').value):
            return

        now_ns = self.get_clock().now().nanoseconds
        write_period_sec = float(self.get_parameter('yaml_write_period_sec').value)
        if (
            track.last_yaml_write_ns
            and now_ns - track.last_yaml_write_ns
            < Duration(seconds=write_period_sec).nanoseconds
        ):
            return

        yaml_path = self.barrel_yaml_path()
        try:
            data = self.load_barrel_yaml(yaml_path)
            barrels = data.setdefault('barrels', [])
            self.merge_barrel_entry(barrels, track)
            self.write_barrel_yaml(yaml_path, data)
            track.last_yaml_write_ns = now_ns
        except (OSError, yaml.YAMLError, ValueError) as exc:
            self.get_logger().warn(
                f'Could not update barrel YAML {yaml_path}: {exc}',
                throttle_duration_sec=2.0,
            )

    def barrel_yaml_path(self) -> str:
        return os.path.expanduser(
            str(self.get_parameter('barrel_yaml_path').value)
        )

    @staticmethod
    def load_barrel_yaml(yaml_path: str) -> Dict:
        if not os.path.exists(yaml_path):
            return {'barrels': []}

        with open(yaml_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file) or {}

        if not isinstance(data, dict):
            raise ValueError('YAML root must be a mapping')

        barrels = data.setdefault('barrels', [])
        if not isinstance(barrels, list):
            raise ValueError('YAML "barrels" field must be a list')

        return data

    def merge_barrel_entry(
        self,
        barrels: List[Dict],
        track: ConfirmationTrack,
    ) -> None:
        merge_distance = float(self.get_parameter('yaml_merge_distance').value)
        matching_barrel = self.closest_yaml_barrel(barrels, track, merge_distance)

        if matching_barrel is None:
            barrels.append(
                {
                    'id': self.next_barrel_id(barrels),
                    'map_x': round(track.x, 3),
                    'map_y': round(track.y, 3),
                    'width': round(track.width, 3),
                    'confirmations': int(track.confirmations),
                }
            )
            return

        previous_confirmations = int(matching_barrel.get('confirmations', 1))
        alpha = 1.0 / min(previous_confirmations + 1, 20)
        matching_barrel['map_x'] = round(
            self.blend(float(matching_barrel['map_x']), track.x, alpha),
            3,
        )
        matching_barrel['map_y'] = round(
            self.blend(float(matching_barrel['map_y']), track.y, alpha),
            3,
        )
        matching_barrel['width'] = round(
            self.blend(
                float(matching_barrel.get('width', track.width)),
                track.width,
                alpha,
            ),
            3,
        )
        matching_barrel['confirmations'] = previous_confirmations + 1

    @staticmethod
    def closest_yaml_barrel(
        barrels: List[Dict],
        track: ConfirmationTrack,
        merge_distance: float,
    ) -> Optional[Dict]:
        closest_barrel = None
        closest_distance = math.inf
        for barrel in barrels:
            try:
                distance = math.hypot(
                    float(barrel['map_x']) - track.x,
                    float(barrel['map_y']) - track.y,
                )
            except (KeyError, TypeError, ValueError):
                continue

            if distance < closest_distance:
                closest_barrel = barrel
                closest_distance = distance

        if closest_distance <= merge_distance:
            return closest_barrel

        return None

    @staticmethod
    def next_barrel_id(barrels: List[Dict]) -> str:
        used_ids = {str(barrel.get('id', '')) for barrel in barrels}
        index = 1
        while True:
            candidate_id = f'Barrel_{index:03d}'
            if candidate_id not in used_ids:
                return candidate_id
            index += 1

    @staticmethod
    def write_barrel_yaml(yaml_path: str, data: Dict) -> None:
        directory = os.path.dirname(yaml_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        temp_path = f'{yaml_path}.tmp'
        with open(temp_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(data, file, sort_keys=False)
        os.replace(temp_path, yaml_path)

    def fuse_pose(
        self,
        lidar_pose: PoseStamped,
        candidate: MapCandidate,
        stamp,
    ) -> PoseStamped:
        lidar_weight = float(self.get_parameter('lidar_weight').value)
        lidar_weight = min(max(lidar_weight, 0.0), 1.0)
        map_weight = 1.0 - lidar_weight

        x = lidar_weight * lidar_pose.pose.position.x + map_weight * candidate.center_x
        y = lidar_weight * lidar_pose.pose.position.y + map_weight * candidate.center_y
        return self.make_pose(x, y, stamp)

    def make_pose(self, x: float, y: float, stamp) -> PoseStamped:
        pose = PoseStamped()
        pose.header.frame_id = self.target_frame
        pose.header.stamp = stamp
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.position.z = 0.0
        pose.pose.orientation.w = 1.0
        return pose

    @staticmethod
    def make_cylinder_marker(
        frame_id: str,
        stamp,
        namespace: str,
        marker_id: int,
        candidate: MapCandidate,
        height: float,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose.position.x = candidate.center_x
        marker.pose.position.y = candidate.center_y
        marker.pose.orientation.w = 1.0
        marker.scale.x = max(candidate.diameter, 0.05)
        marker.scale.y = max(candidate.diameter, 0.05)
        marker.scale.z = height
        return marker

    def publish_delete_marker(self, publisher, namespace: str) -> None:
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = namespace
        marker.id = 0
        marker.action = Marker.DELETE
        publisher.publish(marker)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MapShapeDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
