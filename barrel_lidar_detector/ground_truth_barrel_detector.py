import math
import os
import time
from argparse import Namespace
from typing import Dict, List, Tuple

import yaml

import rclpy
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray

from barrel_lidar_detector.offline_barrel_marker import (
    BarrelCandidate,
    ImageMap,
    MapInfo,
    detect_barrels,
)


class GroundTruthBarrelDetector(Node):
    def __init__(self) -> None:
        super().__init__('ground_truth_barrel_detector')

        self.declare_parameter('map_topic', '/map')
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('barrel_yaml_path', '~/turtlebot4_ws/barrel_target.yaml')
        self.declare_parameter('expected_barrel_count', 0)
        self.declare_parameter('detect_period_sec', 2.0)
        self.declare_parameter('min_score', 0.55)
        self.declare_parameter('min_diameter', 0.0)
        self.declare_parameter('max_diameter', 0.0)
        self.declare_parameter('merge_distance', 0.45)
        self.declare_parameter('min_free_ring_ratio', 0.65)
        self.declare_parameter('max_occupied_ring_ratio', 0.20)
        self.declare_parameter('max_unknown_ring_ratio', 0.35)
        self.declare_parameter('min_circle_support_ratio', 0.75)
        self.declare_parameter('max_center_occupied_ratio', 0.85)
        self.declare_parameter('max_square_corner_ratio', 0.36)
        self.declare_parameter('min_hough_component_roundness', 0.75)
        self.declare_parameter('max_hough_component_corner_fill', 0.70)
        self.declare_parameter('min_border_clearance', 0.35)
        self.declare_parameter('max_context_occupied_ratio', 0.04)

        self.target_frame = str(self.get_parameter('target_frame').value)
        self.last_detection_sec = 0.0
        self.last_signature = None

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            str(self.get_parameter('map_topic').value),
            self.map_callback,
            10,
        )
        self.marker_pub = self.create_publisher(
            MarkerArray,
            '/barrel_ground_truth_markers',
            10,
        )
        self.pose_array_pub = self.create_publisher(
            PoseArray,
            '/barrel_ground_truth_poses',
            10,
        )
        self.confirmed_pose_pub = self.create_publisher(
            PoseStamped,
            '/barrel_confirmed_pose',
            10,
        )
        self.status_pub = self.create_publisher(String, 'mission_status', 10)

        self.get_logger().info(
            'Ground-truth barrel detector listening on '
            f'{self.get_parameter("map_topic").value}; YAML: '
            f'{self.barrel_yaml_path()}'
        )

    def map_callback(self, grid: OccupancyGrid) -> None:
        now_sec = time.monotonic()
        detect_period = max(float(self.get_parameter('detect_period_sec').value), 0.1)
        if now_sec - self.last_detection_sec < detect_period:
            return

        signature = self.map_signature(grid)
        if signature == self.last_signature:
            return

        self.last_detection_sec = now_sec
        self.last_signature = signature

        image, info = self.grid_to_image_map(grid)
        candidates = detect_barrels(image, info, self.detector_args())
        expected_count = self.expected_barrel_count()
        if expected_count > 0:
            candidates = candidates[:expected_count]

        self.write_barrel_yaml(candidates)
        self.publish_barrel_topics(candidates, grid)
        self.publish_status(
            f'Ground-truth map detector found {len(candidates)} barrel(s).'
        )

    def detector_args(self) -> Namespace:
        return Namespace(
            count=0,
            max_auto_barrels=0,
            min_diameter=float(self.get_parameter('min_diameter').value),
            max_diameter=float(self.get_parameter('max_diameter').value),
            min_roundness=0.55,
            min_minor_major_ratio=0.50,
            min_circularity=0.15,
            max_corner_fill_ratio=0.90,
            max_bounding_box_fill_ratio=0.90,
            min_score=float(self.get_parameter('min_score').value),
            min_free_ring_ratio=float(
                self.get_parameter('min_free_ring_ratio').value
            ),
            max_occupied_ring_ratio=float(
                self.get_parameter('max_occupied_ring_ratio').value
            ),
            max_unknown_ring_ratio=float(
                self.get_parameter('max_unknown_ring_ratio').value
            ),
            min_circle_support_ratio=float(
                self.get_parameter('min_circle_support_ratio').value
            ),
            max_center_occupied_ratio=float(
                self.get_parameter('max_center_occupied_ratio').value
            ),
            max_radial_cv=0.32,
            max_straight_edge_ratio=0.80,
            max_square_corner_ratio=float(
                self.get_parameter('max_square_corner_ratio').value
            ),
            min_hough_component_roundness=float(
                self.get_parameter('min_hough_component_roundness').value
            ),
            max_hough_component_corner_fill=float(
                self.get_parameter('max_hough_component_corner_fill').value
            ),
            min_border_clearance=float(
                self.get_parameter('min_border_clearance').value
            ),
            max_context_occupied_ratio=float(
                self.get_parameter('max_context_occupied_ratio').value
            ),
            merge_distance=float(self.get_parameter('merge_distance').value),
        )

    def grid_to_image_map(self, grid: OccupancyGrid) -> Tuple[ImageMap, MapInfo]:
        width = int(grid.info.width)
        height = int(grid.info.height)
        pixels: List[int] = []
        for image_y in range(height):
            map_y = height - 1 - image_y
            for x in range(width):
                value = grid.data[map_y * width + x]
                if value < 0:
                    pixels.append(205)
                elif value >= 65:
                    pixels.append(0)
                elif value <= 25:
                    pixels.append(254)
                else:
                    pixels.append(205)

        origin = grid.info.origin
        info = MapInfo(
            image_path='',
            resolution=float(grid.info.resolution),
            origin_x=float(origin.position.x),
            origin_y=float(origin.position.y),
            origin_yaw=self.quaternion_to_yaw(origin.orientation),
            negate=0,
            occupied_thresh=0.65,
            free_thresh=0.25,
        )
        return ImageMap(width=width, height=height, pixels=pixels), info

    @staticmethod
    def quaternion_to_yaw(orientation) -> float:
        siny_cosp = 2.0 * (
            orientation.w * orientation.z + orientation.x * orientation.y
        )
        cosy_cosp = 1.0 - 2.0 * (
            orientation.y * orientation.y + orientation.z * orientation.z
        )
        return math.atan2(siny_cosp, cosy_cosp)

    @staticmethod
    def map_signature(grid: OccupancyGrid) -> Tuple[int, int, float, int]:
        return (
            int(grid.info.width),
            int(grid.info.height),
            float(grid.info.resolution),
            hash(bytes((value + 1) & 0xFF for value in grid.data)),
        )

    def publish_barrel_topics(
        self,
        candidates: List[BarrelCandidate],
        grid: OccupancyGrid,
    ) -> None:
        stamp = self.get_clock().now().to_msg()
        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header.frame_id = self.target_frame
        delete_marker.header.stamp = stamp
        delete_marker.ns = 'barrel_ground_truth'
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        pose_array = PoseArray()
        pose_array.header.frame_id = self.target_frame
        pose_array.header.stamp = stamp

        for index, candidate in enumerate(candidates, start=1):
            pose = self.make_pose(candidate.center_x, candidate.center_y)
            pose_array.poses.append(pose)
            marker_array.markers.append(
                self.make_cylinder_marker(index, candidate, pose, stamp)
            )
            marker_array.markers.append(
                self.make_label_marker(index, candidate, pose, stamp)
            )

        self.marker_pub.publish(marker_array)
        self.pose_array_pub.publish(pose_array)
        if pose_array.poses:
            first_pose = PoseStamped()
            first_pose.header = pose_array.header
            first_pose.pose = pose_array.poses[0]
            self.confirmed_pose_pub.publish(first_pose)

    def make_cylinder_marker(
        self,
        marker_id: int,
        candidate: BarrelCandidate,
        pose: Pose,
        stamp,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = 'barrel_ground_truth'
        marker.id = marker_id
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.pose = self.copy_pose(pose)
        marker.pose.position.z = 0.25
        diameter = max(candidate.diameter, 0.25)
        marker.scale.x = diameter
        marker.scale.y = diameter
        marker.scale.z = 0.50
        marker.color.r = 1.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 0.85
        return marker

    def make_label_marker(
        self,
        marker_id: int,
        candidate: BarrelCandidate,
        pose: Pose,
        stamp,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = self.target_frame
        marker.header.stamp = stamp
        marker.ns = 'barrel_ground_truth_labels'
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose = self.copy_pose(pose)
        marker.pose.position.z = 0.85
        marker.scale.z = 0.22
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.text = f'Barrel_{marker_id:03d}'
        return marker

    @staticmethod
    def make_pose(x: float, y: float) -> Pose:
        pose = Pose()
        pose.position.x = x
        pose.position.y = y
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        return pose

    @staticmethod
    def copy_pose(source: Pose) -> Pose:
        pose = Pose()
        pose.position.x = source.position.x
        pose.position.y = source.position.y
        pose.position.z = source.position.z
        pose.orientation.x = source.orientation.x
        pose.orientation.y = source.orientation.y
        pose.orientation.z = source.orientation.z
        pose.orientation.w = source.orientation.w
        return pose

    def write_barrel_yaml(self, candidates: List[BarrelCandidate]) -> None:
        now_sec = round(time.time(), 3)
        barrels: List[Dict] = []
        for index, candidate in enumerate(candidates, start=1):
            barrels.append(
                {
                    'id': f'Barrel_{index:03d}',
                    'map_x': round(candidate.center_x, 3),
                    'map_y': round(candidate.center_y, 3),
                    'width': round(candidate.diameter, 3),
                    'roundness': round(candidate.roundness, 3),
                    'score': round(candidate.score, 3),
                    'confirmations': 1,
                    'last_seen_unix_sec': now_sec,
                }
            )

        yaml_path = self.barrel_yaml_path()
        directory = os.path.dirname(yaml_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        temp_path = f'{yaml_path}.tmp'
        with open(temp_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump({'barrels': barrels}, file, sort_keys=False)
        os.replace(temp_path, yaml_path)

    def barrel_yaml_path(self) -> str:
        return os.path.expanduser(
            str(self.get_parameter('barrel_yaml_path').value)
        )

    def expected_barrel_count(self) -> int:
        try:
            return max(int(self.get_parameter('expected_barrel_count').value), 0)
        except (TypeError, ValueError):
            return 0

    def publish_status(self, text: str) -> None:
        message = String()
        message.data = text
        self.status_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GroundTruthBarrelDetector()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
