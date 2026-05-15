import argparse
import math
import os
import time
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


DEFAULT_OCCUPIED_THRESHOLD = 50
DEFAULT_FREE_THRESHOLD = 25
DEFAULT_DETECTION_SENSITIVITY = 50.0


DEFAULT_DETECTOR_PARAMS = {
    'min_score': 0.55,
    'min_diameter': 0.0,
    'max_diameter': 0.0,
    'min_radius_pixels': 3,
    'hough_param2': 10.0,
    'merge_distance': 0.45,
    'min_roundness': 0.55,
    'min_minor_major_ratio': 0.50,
    'min_circularity': 0.15,
    'max_corner_fill_ratio': 0.90,
    'max_bounding_box_fill_ratio': 0.90,
    'min_free_ring_ratio': 0.65,
    'max_occupied_ring_ratio': 0.20,
    'max_unknown_ring_ratio': 0.35,
    'min_circle_support_ratio': 0.75,
    'max_center_occupied_ratio': 0.85,
    'max_radial_cv': 0.32,
    'max_straight_edge_ratio': 0.80,
    'max_square_corner_ratio': 0.36,
    'min_hough_component_roundness': 0.75,
    'max_hough_component_corner_fill': 0.70,
    'min_border_clearance': 0.35,
    'max_context_occupied_ratio': 0.04,
    'allow_wall_touching': True,
    'max_wall_touch_context_occupied_ratio': 0.30,
    'max_wall_touch_occupied_ring_ratio': 0.45,
    'max_wall_touch_angle_ratio': 0.45,
    'min_wall_touch_longest_run_ratio': 0.60,
}


@dataclass
class MapInfo:
    image_path: str
    resolution: float
    origin_x: float
    origin_y: float
    origin_yaw: float
    negate: int
    occupied_thresh: float
    free_thresh: float


@dataclass
class ImageMap:
    width: int
    height: int
    pixels: List[int]


@dataclass
class BarrelCandidate:
    center_x: float
    center_y: float
    diameter: float
    width_x: float
    width_y: float
    occupied_cells: int
    roundness: float
    circularity: float
    corner_fill_ratio: float
    bounding_box_fill_ratio: float
    fill_ratio: float
    free_ring_ratio: float
    occupied_ring_ratio: float
    unknown_ring_ratio: float
    circle_support_ratio: float
    center_occupied_ratio: float
    radial_cv: float
    straight_edge_ratio: float
    square_corner_ratio: float
    score: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Mark barrels from a saved ROS map without running the robot. '
            'By default, round occupied blobs are detected from the map image.'
        )
    )
    parser.add_argument(
        'map_yaml',
        nargs='*',
        help='Saved ROS map YAML, for example arena_map.yaml',
    )
    parser.add_argument(
        '--all-maps',
        action='store_true',
        help='Process every saved map YAML in the current directory.',
    )
    parser.add_argument(
        '-o',
        '--output',
        default='barrel_target.yaml',
        help='Output barrel YAML path for one map. Default: barrel_target.yaml',
    )
    parser.add_argument(
        '--output-dir',
        default='barrel_targets',
        help='Output directory when processing multiple maps. Default: barrel_targets',
    )
    parser.add_argument(
        '--barrel',
        action='append',
        default=[],
        metavar='X,Y',
        help='Manually add a barrel center in map-frame meters. Can be repeated.',
    )
    parser.add_argument(
        '--no-auto',
        action='store_true',
        help='Disable automatic round-blob detection and only use --barrel entries.',
    )
    parser.add_argument(
        '--count',
        type=int,
        default=0,
        help='Maximum barrels to keep. 0 means keep every likely barrel.',
    )
    parser.add_argument(
        '--max-auto-barrels',
        type=int,
        default=0,
        help='Maximum automatically detected barrels per map. 0 disables this cap.',
    )
    parser.add_argument(
        '--min-diameter',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_diameter'],
        help='Minimum barrel diameter in meters. 0 lets the map image choose.',
    )
    parser.add_argument(
        '--max-diameter',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_diameter'],
        help='Maximum barrel diameter in meters. 0 lets the map image choose.',
    )
    parser.add_argument(
        '--occupied-threshold',
        type=int,
        default=DEFAULT_OCCUPIED_THRESHOLD,
        help='Occupancy probability threshold, 0-100. Default: 50.',
    )
    parser.add_argument(
        '--free-threshold',
        type=int,
        default=DEFAULT_FREE_THRESHOLD,
        help='Free probability threshold, 0-100. Default: 25.',
    )
    parser.add_argument(
        '--sensitivity',
        dest='detection_sensitivity',
        type=float,
        default=DEFAULT_DETECTION_SENSITIVITY,
        help='Detection sensitivity from 0 to 100. Default: 50.',
    )
    parser.add_argument(
        '--min-radius-pixels',
        type=int,
        default=DEFAULT_DETECTOR_PARAMS['min_radius_pixels'],
        help='Smallest Hough circle radius in map pixels. Default: 3.',
    )
    parser.add_argument(
        '--hough-param2',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['hough_param2'],
        help='Hough accumulator threshold. Lower values find weaker circles.',
    )
    parser.add_argument(
        '--min-roundness',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_roundness'],
    )
    parser.add_argument(
        '--min-minor-major-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_minor_major_ratio'],
    )
    parser.add_argument(
        '--min-circularity',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_circularity'],
    )
    parser.add_argument(
        '--max-corner-fill-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_corner_fill_ratio'],
    )
    parser.add_argument(
        '--max-bounding-box-fill-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_bounding_box_fill_ratio'],
    )
    parser.add_argument(
        '--min-score',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_score'],
        help='Minimum automatic barrel confidence score. Default: 0.55',
    )
    parser.add_argument(
        '--min-free-ring-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_free_ring_ratio'],
        help='Minimum free-space ratio around an automatic barrel. Default: 0.65',
    )
    parser.add_argument(
        '--max-occupied-ring-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_occupied_ring_ratio'],
        help='Maximum occupied-space ratio around an automatic barrel. Default: 0.20',
    )
    parser.add_argument(
        '--max-unknown-ring-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_unknown_ring_ratio'],
        help='Maximum unknown-space ratio around an automatic barrel. Default: 0.35',
    )
    parser.add_argument(
        '--min-circle-support-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_circle_support_ratio'],
        help='Minimum occupied support around the fitted barrel circle.',
    )
    parser.add_argument(
        '--max-center-occupied-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_center_occupied_ratio'],
        help='Reject filled/square blobs with too much occupied center area.',
    )
    parser.add_argument(
        '--max-radial-cv',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_radial_cv'],
        help='Reject blobs whose occupied pixels do not fit a circle closely.',
    )
    parser.add_argument(
        '--max-straight-edge-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_straight_edge_ratio'],
        help='Reject square-like outlines with dominant straight edges.',
    )
    parser.add_argument(
        '--max-square-corner-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_square_corner_ratio'],
        help='Reject Hough circles that are actually square/rectangular corners.',
    )
    parser.add_argument(
        '--min-hough-component-roundness',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_hough_component_roundness'],
        help='Minimum roundness of the occupied component under a Hough circle.',
    )
    parser.add_argument(
        '--max-hough-component-corner-fill',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_hough_component_corner_fill'],
        help='Reject Hough circles over square components with filled corners.',
    )
    parser.add_argument(
        '--min-border-clearance',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_border_clearance'],
        help='Reject circles too close to image edges or unexplored map borders.',
    )
    parser.add_argument(
        '--max-context-occupied-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_context_occupied_ratio'],
        help='Reject circles too close to larger occupied map structure.',
    )
    parser.add_argument(
        '--no-wall-touching',
        dest='allow_wall_touching',
        action='store_false',
        help='Disable special handling for barrels tangent to walls.',
    )
    parser.set_defaults(
        allow_wall_touching=DEFAULT_DETECTOR_PARAMS['allow_wall_touching']
    )
    parser.add_argument(
        '--max-wall-touch-context-occupied-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_wall_touch_context_occupied_ratio'],
    )
    parser.add_argument(
        '--max-wall-touch-occupied-ring-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_wall_touch_occupied_ring_ratio'],
    )
    parser.add_argument(
        '--max-wall-touch-angle-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['max_wall_touch_angle_ratio'],
    )
    parser.add_argument(
        '--min-wall-touch-longest-run-ratio',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['min_wall_touch_longest_run_ratio'],
    )
    parser.add_argument(
        '--merge-distance',
        type=float,
        default=DEFAULT_DETECTOR_PARAMS['merge_distance'],
        help='Merge manual/detected barrels closer than this many meters.',
    )
    parser.add_argument(
        '--annotated-image',
        help='Optional PGM preview path with marked barrel centers.',
    )
    parser.add_argument(
        '--annotated-dir',
        default='marked_maps',
        help='Preview output directory when processing multiple maps.',
    )
    return parser.parse_args()


def load_map_info(map_yaml_path: str) -> MapInfo:
    with open(map_yaml_path, 'r', encoding='utf-8') as file:
        data = yaml.safe_load(file) or {}

    origin = data.get('origin', [0.0, 0.0, 0.0])
    if not isinstance(origin, Sequence) or len(origin) < 3:
        raise ValueError('map YAML origin must contain [x, y, yaw]')

    image_path = str(data['image'])
    if not os.path.isabs(image_path):
        image_path = os.path.join(os.path.dirname(map_yaml_path), image_path)

    return MapInfo(
        image_path=image_path,
        resolution=float(data['resolution']),
        origin_x=float(origin[0]),
        origin_y=float(origin[1]),
        origin_yaw=float(origin[2]),
        negate=int(data.get('negate', 0)),
        occupied_thresh=float(data.get('occupied_thresh', 0.65)),
        free_thresh=float(data.get('free_thresh', 0.25)),
    )


def load_pgm(path: str) -> ImageMap:
    with open(path, 'rb') as file:
        magic = read_token(file)
        if magic not in (b'P2', b'P5'):
            raise ValueError(f'{path} is not a PGM file')

        width = int(read_token(file))
        height = int(read_token(file))
        max_value = int(read_token(file))
        if max_value <= 0 or max_value > 65535:
            raise ValueError(f'{path} has unsupported max value {max_value}')

        if magic == b'P2':
            pixels = [
                scale_pixel(int(read_token(file)), max_value)
                for _ in range(width * height)
            ]
        else:
            if max_value < 256:
                raw = file.read(width * height)
                pixels = [scale_pixel(value, max_value) for value in raw]
            else:
                raw = file.read(width * height * 2)
                pixels = [
                    scale_pixel(int.from_bytes(raw[index:index + 2], 'big'), max_value)
                    for index in range(0, len(raw), 2)
                ]

    if len(pixels) != width * height:
        raise ValueError(f'{path} ended before all pixels were read')

    return ImageMap(width=width, height=height, pixels=pixels)


def read_token(file) -> bytes:
    token = bytearray()
    while True:
        char = file.read(1)
        if not char:
            if token:
                return bytes(token)
            raise ValueError('unexpected end of PGM file')
        if char == b'#':
            file.readline()
            continue
        if char.isspace():
            if token:
                return bytes(token)
            continue
        token.extend(char)


def scale_pixel(value: int, max_value: int) -> int:
    return round(255 * value / max_value)


def occupancy_probability(pixel: int, info: MapInfo) -> float:
    color = pixel if info.negate else 255 - pixel
    return color / 255.0


def detect_barrels(
    image: ImageMap,
    info: MapInfo,
    args: argparse.Namespace,
) -> List[BarrelCandidate]:
    args = apply_detection_sensitivity(args)
    hough_candidates = detect_hough_barrels(image, info, args)
    if hough_candidates:
        return hough_candidates

    visited = bytearray(image.width * image.height)
    candidates: List[BarrelCandidate] = []

    for index, pixel in enumerate(image.pixels):
        if visited[index] or occupancy_probability(pixel, info) < info.occupied_thresh:
            continue

        component = collect_component(index, image, info, visited)
        candidate = component_to_candidate(component, image, info, args)
        if candidate is not None:
            candidates.append(candidate)

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    if args.count > 0:
        candidates = candidates[:args.count]
    return candidates


def apply_detection_sensitivity(args: argparse.Namespace) -> argparse.Namespace:
    sensitivity = clamp(
        float(getattr(args, 'detection_sensitivity', DEFAULT_DETECTION_SENSITIVITY)),
        0.0,
        100.0,
    )
    adjusted = argparse.Namespace(**vars(args))
    if sensitivity > DEFAULT_DETECTION_SENSITIVITY:
        factor = (sensitivity - DEFAULT_DETECTION_SENSITIVITY) / 50.0
        adjusted.min_score = clamp(args.min_score - 0.10 * factor, 0.25, 0.95)
        adjusted.min_free_ring_ratio = clamp(
            args.min_free_ring_ratio - 0.20 * factor,
            0.20,
            0.95,
        )
        adjusted.max_occupied_ring_ratio = clamp(
            args.max_occupied_ring_ratio + 0.15 * factor,
            0.05,
            0.60,
        )
        adjusted.max_unknown_ring_ratio = clamp(
            args.max_unknown_ring_ratio + 0.35 * factor,
            0.05,
            0.95,
        )
        adjusted.min_circle_support_ratio = clamp(
            args.min_circle_support_ratio - 0.20 * factor,
            0.35,
            0.95,
        )
        adjusted.min_hough_component_roundness = clamp(
            args.min_hough_component_roundness - 0.30 * factor,
            0.25,
            0.95,
        )
        adjusted.max_hough_component_corner_fill = clamp(
            args.max_hough_component_corner_fill + 0.10 * factor,
            0.40,
            0.95,
        )
        adjusted.min_border_clearance = clamp(
            args.min_border_clearance - 0.25 * factor,
            0.00,
            1.00,
        )
        adjusted.max_context_occupied_ratio = clamp(
            args.max_context_occupied_ratio + 0.06 * factor,
            0.00,
            0.20,
        )
        adjusted.hough_param2 = clamp(args.hough_param2 - 4.0 * factor, 4.0, 20.0)
        adjusted.min_radius_pixels = max(1, round(args.min_radius_pixels - factor))
    elif sensitivity < DEFAULT_DETECTION_SENSITIVITY:
        factor = (DEFAULT_DETECTION_SENSITIVITY - sensitivity) / 50.0
        adjusted.min_score = clamp(args.min_score + 0.15 * factor, 0.25, 0.95)
        adjusted.min_free_ring_ratio = clamp(
            args.min_free_ring_ratio + 0.15 * factor,
            0.20,
            0.95,
        )
        adjusted.max_occupied_ring_ratio = clamp(
            args.max_occupied_ring_ratio - 0.08 * factor,
            0.05,
            0.60,
        )
        adjusted.max_unknown_ring_ratio = clamp(
            args.max_unknown_ring_ratio - 0.15 * factor,
            0.05,
            0.95,
        )
        adjusted.min_circle_support_ratio = clamp(
            args.min_circle_support_ratio + 0.10 * factor,
            0.35,
            0.95,
        )
        adjusted.min_hough_component_roundness = clamp(
            args.min_hough_component_roundness + 0.15 * factor,
            0.25,
            0.95,
        )
        adjusted.max_context_occupied_ratio = clamp(
            args.max_context_occupied_ratio - 0.02 * factor,
            0.00,
            0.20,
        )
        adjusted.hough_param2 = clamp(args.hough_param2 + 3.0 * factor, 4.0, 20.0)
        adjusted.min_radius_pixels = max(1, round(args.min_radius_pixels + factor))

    return adjusted


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def detect_hough_barrels(
    image: ImageMap,
    info: MapInfo,
    args: argparse.Namespace,
) -> List[BarrelCandidate]:
    grayscale = np.array(image.pixels, dtype=np.uint8).reshape(
        (image.height, image.width)
    )
    occupied_image = 255 - grayscale if not info.negate else grayscale
    min_radius, max_radius = hough_radius_range(image, info, args)
    proposal_images = hough_proposal_images(occupied_image)
    hough_param2_values = [float(getattr(args, 'hough_param2', 10.0))]
    if min_radius <= 2:
        hough_param2_values.append(max(4.0, hough_param2_values[0] - 2.0))

    candidates = []
    for proposal_image in proposal_images:
        for hough_param2 in hough_param2_values:
            circles = cv2.HoughCircles(
                proposal_image,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=max(8, int(round(args.merge_distance / info.resolution))),
                param1=80,
                param2=hough_param2,
                minRadius=min_radius,
                maxRadius=max_radius,
            )
            if circles is None:
                continue

            for image_x, image_y, radius in np.round(circles[0, :]).astype(int):
                candidate = hough_circle_to_candidate(
                    int(image_x),
                    int(image_y),
                    int(radius),
                    image,
                    info,
                    args,
                )
                if candidate is not None:
                    candidates.append(candidate)

    candidates = merge_close_candidates(candidates, args.merge_distance)
    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    if args.count > 0:
        candidates = candidates[:args.count]
    return candidates


def hough_radius_range(
    image: ImageMap,
    info: MapInfo,
    args: argparse.Namespace,
) -> Tuple[int, int]:
    if args.min_diameter > 0.0:
        min_radius = int(math.floor(args.min_diameter * 0.5 / info.resolution))
    else:
        min_radius = int(getattr(args, 'min_radius_pixels', 3))

    if args.max_diameter > 0.0:
        max_radius = int(math.ceil(args.max_diameter * 0.5 / info.resolution))
    else:
        max_radius = max(8, min(image.width, image.height) // 12)

    min_radius = max(1, min_radius)
    max_radius = max(min_radius + 1, max_radius)
    return min_radius, max_radius


def hough_proposal_images(occupied_image: np.ndarray) -> List[np.ndarray]:
    blurred = cv2.medianBlur(occupied_image, 5)
    _, occupied_mask = cv2.threshold(occupied_image, 160, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(occupied_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    filled = fill_enclosed_regions(closed)
    filled_edges = cv2.Canny(filled, 50, 150)
    return [
        blurred,
        cv2.medianBlur(closed, 5),
        cv2.GaussianBlur(filled_edges, (5, 5), 0),
    ]


def fill_enclosed_regions(mask: np.ndarray) -> np.ndarray:
    flood = mask.copy()
    height, width = flood.shape
    flood_mask = np.zeros((height + 2, width + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    enclosed = cv2.bitwise_not(flood)
    return cv2.bitwise_or(mask, enclosed)


def hough_circle_to_candidate(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
    args: argparse.Namespace,
) -> Optional[BarrelCandidate]:
    map_cell_y = image.height - 1 - image_y
    diameter = 2.0 * radius_cells * info.resolution
    center_x, center_y = cell_to_world(image_x + 0.5, map_cell_y + 0.5, info)
    border_clearance_cells = radius_cells + max(
        1.0,
        args.min_border_clearance / info.resolution,
    )
    if (
        image_x < border_clearance_cells
        or image_y < border_clearance_cells
        or image.width - 1 - image_x < border_clearance_cells
        or image.height - 1 - image_y < border_clearance_cells
    ):
        return None

    circle_support_ratio, center_occupied_ratio = hough_circle_template_ratios(
        image_x,
        image_y,
        radius_cells,
        image,
        info,
    )
    if circle_support_ratio < args.min_circle_support_ratio:
        return None
    if center_occupied_ratio > args.max_center_occupied_ratio:
        return None

    component_metrics = hough_component_metrics(
        image_x,
        image_y,
        radius_cells,
        image,
        info,
        local_limit=False,
    )
    component_ok = hough_component_passes(component_metrics, args)

    free_ring_ratio, occupied_ring_ratio, unknown_ring_ratio = hough_free_ring_ratios(
        image_x,
        image_y,
        radius_cells,
        image,
        info,
    )
    if unknown_ring_ratio > args.max_unknown_ring_ratio:
        return None

    square_corner_ratio = hough_square_corner_ratio(
        image_x,
        image_y,
        radius_cells,
        image,
        info,
    )
    if square_corner_ratio > args.max_square_corner_ratio:
        return None

    context_metrics = hough_context_occupied_metrics(
        image_x,
        image_y,
        radius_cells,
        image,
        info,
    )
    (
        context_occupied_ratio,
        context_angle_ratio,
        context_longest_run_ratio,
    ) = context_metrics
    wall_touching = hough_allows_wall_touching(
        args,
        free_ring_ratio,
        occupied_ring_ratio,
        context_occupied_ratio,
        context_angle_ratio,
        context_longest_run_ratio,
    )
    if wall_touching and not component_ok:
        component_metrics = hough_component_metrics(
            image_x,
            image_y,
            radius_cells,
            image,
            info,
            local_limit=True,
        )
        component_ok = hough_component_passes(component_metrics, args)

    if not component_ok or component_metrics is None:
        return None

    component_cells, component_roundness, component_corner_fill = component_metrics
    if not wall_touching and free_ring_ratio < args.min_free_ring_ratio:
        return None
    if not wall_touching and occupied_ring_ratio > args.max_occupied_ring_ratio:
        return None
    if not wall_touching and context_occupied_ratio > args.max_context_occupied_ratio:
        return None

    side_ratio = hough_square_side_ratio(image_x, image_y, radius_cells, image, info)
    size_score = score_size(diameter, args.min_diameter, args.max_diameter)
    score = (
        0.30 * circle_support_ratio
        + 0.18 * component_roundness
        + 0.12 * (1.0 - component_corner_fill)
        + 0.10 * (1.0 - center_occupied_ratio)
        + 0.15 * free_ring_ratio
        + 0.15 * (1.0 - square_corner_ratio)
        + 0.10 * size_score
        - 0.05 * side_ratio
        - 0.05 * occupied_ring_ratio
        - 0.05 * unknown_ring_ratio
        - 0.08 * context_occupied_ratio
    )
    if score < args.min_score:
        return None

    return BarrelCandidate(
        center_x=center_x,
        center_y=center_y,
        diameter=diameter,
        width_x=diameter,
        width_y=diameter,
        occupied_cells=component_cells,
        roundness=component_roundness,
        circularity=circle_support_ratio,
        corner_fill_ratio=component_corner_fill,
        bounding_box_fill_ratio=side_ratio,
        fill_ratio=1.0 - center_occupied_ratio,
        free_ring_ratio=free_ring_ratio,
        occupied_ring_ratio=occupied_ring_ratio,
        unknown_ring_ratio=unknown_ring_ratio,
        circle_support_ratio=circle_support_ratio,
        center_occupied_ratio=center_occupied_ratio,
        radial_cv=0.0,
        straight_edge_ratio=side_ratio,
        square_corner_ratio=square_corner_ratio,
        score=score,
    )


def hough_circle_template_ratios(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
) -> Tuple[float, float]:
    circle_bins = 32
    supported_bins = 0
    for bin_index in range(circle_bins):
        angle = 2.0 * math.pi * bin_index / circle_bins
        if hough_circle_bin_has_occupied_cell(
            image_x,
            image_y,
            radius_cells,
            angle,
            image,
            info,
        ):
            supported_bins += 1

    center_radius = max(1.0, radius_cells * 0.45)
    occupied_center_cells = 0
    total_center_cells = 0
    for y in range(
        int(round(image_y - center_radius)),
        int(round(image_y + center_radius)) + 1,
    ):
        for x in range(
            int(round(image_x - center_radius)),
            int(round(image_x + center_radius)) + 1,
        ):
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue
            if math.hypot(x - image_x, y - image_y) > center_radius:
                continue

            total_center_cells += 1
            if image_cell_is_occupied(x, y, image, info):
                occupied_center_cells += 1

    center_occupied_ratio = (
        occupied_center_cells / total_center_cells
        if total_center_cells
        else 1.0
    )
    return supported_bins / circle_bins, center_occupied_ratio


def hough_component_metrics(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
    local_limit: bool = False,
) -> Optional[Tuple[int, float, float]]:
    start = nearest_occupied_cell_on_circle(image_x, image_y, radius_cells, image, info)
    if start is None:
        return None

    queue = deque([start])
    visited = {start}
    component: List[Tuple[int, int]] = []
    local_radius = radius_cells + max(2.0, 0.15 / info.resolution)

    while queue:
        x, y = queue.popleft()
        component.append((x, y))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                next_x = x + dx
                next_y = y + dy
                next_cell = (next_x, next_y)
                if next_cell in visited:
                    continue
                if (
                    local_limit
                    and math.hypot(next_x - image_x, next_y - image_y) > local_radius
                ):
                    continue
                if not image_cell_is_occupied(next_x, next_y, image, info):
                    continue

                visited.add(next_cell)
                queue.append(next_cell)

    return (
        len(component),
        component_roundness(component),
        component_corner_fill_ratio(component),
    )


def hough_component_passes(
    component_metrics: Optional[Tuple[int, float, float]],
    args: argparse.Namespace,
) -> bool:
    if component_metrics is None:
        return False
    _, component_roundness, component_corner_fill = component_metrics
    if component_roundness < args.min_hough_component_roundness:
        return False
    if component_corner_fill > args.max_hough_component_corner_fill:
        return False
    return True


def nearest_occupied_cell_on_circle(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
) -> Optional[Tuple[int, int]]:
    best_cell = None
    best_distance = math.inf
    search_radius = radius_cells + 2
    for y in range(image_y - search_radius, image_y + search_radius + 1):
        for x in range(image_x - search_radius, image_x + search_radius + 1):
            if not image_cell_is_occupied(x, y, image, info):
                continue

            distance = abs(math.hypot(x - image_x, y - image_y) - radius_cells)
            if distance < best_distance:
                best_cell = (x, y)
                best_distance = distance

    return best_cell


def hough_circle_bin_has_occupied_cell(
    image_x: int,
    image_y: int,
    radius_cells: int,
    angle: float,
    image: ImageMap,
    info: MapInfo,
) -> bool:
    for radius_offset in (-1.8, -0.6, 0.6, 1.8):
        radius = radius_cells + radius_offset
        if radius <= 0.0:
            continue
        for angle_offset in (-0.08, 0.0, 0.08):
            x = int(round(image_x + radius * math.cos(angle + angle_offset)))
            y = int(round(image_y + radius * math.sin(angle + angle_offset)))
            if image_cell_is_occupied(x, y, image, info):
                return True
    return False


def hough_free_ring_ratios(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
) -> Tuple[float, float, float]:
    inner_radius = radius_cells + max(1.0, 0.08 / info.resolution)
    outer_radius = radius_cells + max(2.0, 0.35 / info.resolution)
    outer_cells = int(math.ceil(outer_radius))
    free_cells = 0
    occupied_cells = 0
    unknown_cells = 0

    for y in range(image_y - outer_cells, image_y + outer_cells + 1):
        for x in range(image_x - outer_cells, image_x + outer_cells + 1):
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue
            distance = math.hypot(x - image_x, y - image_y)
            if distance < inner_radius or distance > outer_radius:
                continue

            occupancy = image_cell_occupancy(x, y, image, info)
            if occupancy >= info.occupied_thresh:
                occupied_cells += 1
            elif occupancy <= info.free_thresh:
                free_cells += 1
            else:
                unknown_cells += 1

    total = free_cells + occupied_cells + unknown_cells
    if total == 0:
        return 0.0, 1.0, 0.0
    return free_cells / total, occupied_cells / total, unknown_cells / total


def hough_context_occupied_ratio(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
) -> float:
    inner_radius = radius_cells + max(2.0, 0.35 / info.resolution)
    outer_radius = radius_cells + max(inner_radius + 1.0, 1.0 / info.resolution)
    outer_cells = int(math.ceil(outer_radius))
    occupied_cells = 0
    total_cells = 0

    for y in range(image_y - outer_cells, image_y + outer_cells + 1):
        for x in range(image_x - outer_cells, image_x + outer_cells + 1):
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue

            distance = math.hypot(x - image_x, y - image_y)
            if distance < inner_radius or distance > outer_radius:
                continue

            total_cells += 1
            if image_cell_is_occupied(x, y, image, info):
                occupied_cells += 1

    return occupied_cells / total_cells if total_cells else 1.0


def hough_context_occupied_metrics(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
) -> Tuple[float, float, float]:
    inner_radius = radius_cells + max(2.0, 0.35 / info.resolution)
    outer_radius = radius_cells + max(inner_radius + 1.0, 1.0 / info.resolution)
    outer_cells = int(math.ceil(outer_radius))
    angle_bins = 36
    occupied_bins = [False] * angle_bins
    occupied_cells = 0
    total_cells = 0

    for y in range(image_y - outer_cells, image_y + outer_cells + 1):
        for x in range(image_x - outer_cells, image_x + outer_cells + 1):
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue

            dx = x - image_x
            dy = y - image_y
            distance = math.hypot(dx, dy)
            if distance < inner_radius or distance > outer_radius:
                continue

            total_cells += 1
            if image_cell_is_occupied(x, y, image, info):
                occupied_cells += 1
                angle = math.atan2(dy, dx)
                bin_index = int(((angle + math.pi) / (2.0 * math.pi)) * angle_bins)
                occupied_bins[min(angle_bins - 1, max(0, bin_index))] = True

    if total_cells == 0:
        return 1.0, 1.0, 1.0

    occupied_bin_count = sum(1 for occupied in occupied_bins if occupied)
    angle_ratio = occupied_bin_count / angle_bins
    longest_run = longest_circular_true_run(occupied_bins)
    longest_run_ratio = (
        longest_run / occupied_bin_count
        if occupied_bin_count
        else 0.0
    )
    return occupied_cells / total_cells, angle_ratio, longest_run_ratio


def longest_circular_true_run(values: Sequence[bool]) -> int:
    if not values:
        return 0
    if all(values):
        return len(values)

    doubled = list(values) + list(values)
    best = 0
    current = 0
    for value in doubled:
        if value:
            current += 1
            best = max(best, min(current, len(values)))
        else:
            current = 0
    return best


def hough_allows_wall_touching(
    args: argparse.Namespace,
    free_ring_ratio: float,
    occupied_ring_ratio: float,
    context_occupied_ratio: float,
    context_angle_ratio: float,
    context_longest_run_ratio: float,
) -> bool:
    if not bool(getattr(args, 'allow_wall_touching', True)):
        return False
    ring_shows_contact = (
        free_ring_ratio < args.min_free_ring_ratio
        or occupied_ring_ratio > args.max_occupied_ring_ratio
    )
    if not ring_shows_contact:
        return False
    if occupied_ring_ratio > args.max_wall_touch_occupied_ring_ratio:
        return False
    if context_occupied_ratio > args.max_wall_touch_context_occupied_ratio:
        return False
    if context_angle_ratio > args.max_wall_touch_angle_ratio:
        return False
    if context_longest_run_ratio < args.min_wall_touch_longest_run_ratio:
        return False
    return context_occupied_ratio > args.max_context_occupied_ratio


def hough_square_corner_ratio(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
) -> float:
    corner_ratios = []
    for corner_x in (image_x - radius_cells, image_x + radius_cells):
        for corner_y in (image_y - radius_cells, image_y + radius_cells):
            occupied_cells = 0
            total_cells = 0
            for y in range(corner_y - 1, corner_y + 2):
                for x in range(corner_x - 1, corner_x + 2):
                    if x < 0 or x >= image.width or y < 0 or y >= image.height:
                        continue
                    total_cells += 1
                    if image_cell_is_occupied(x, y, image, info):
                        occupied_cells += 1
            if total_cells:
                corner_ratios.append(occupied_cells / total_cells)

    return sum(corner_ratios) / len(corner_ratios) if corner_ratios else 1.0


def hough_square_side_ratio(
    image_x: int,
    image_y: int,
    radius_cells: int,
    image: ImageMap,
    info: MapInfo,
) -> float:
    occupied_cells = 0
    total_cells = 0
    for y in range(image_y - radius_cells, image_y + radius_cells + 1):
        for x in (image_x - radius_cells, image_x + radius_cells):
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue
            total_cells += 1
            if image_cell_is_occupied(x, y, image, info):
                occupied_cells += 1

    for x in range(image_x - radius_cells, image_x + radius_cells + 1):
        for y in (image_y - radius_cells, image_y + radius_cells):
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue
            total_cells += 1
            if image_cell_is_occupied(x, y, image, info):
                occupied_cells += 1

    return occupied_cells / total_cells if total_cells else 1.0


def image_cell_occupancy(x: int, y: int, image: ImageMap, info: MapInfo) -> float:
    if x < 0 or x >= image.width or y < 0 or y >= image.height:
        return 0.0
    return occupancy_probability(image.pixels[y * image.width + x], info)


def image_cell_is_occupied(x: int, y: int, image: ImageMap, info: MapInfo) -> bool:
    return image_cell_occupancy(x, y, image, info) >= info.occupied_thresh


def collect_component(
    start_index: int,
    image: ImageMap,
    info: MapInfo,
    visited: bytearray,
) -> List[Tuple[int, int]]:
    queue = deque([start_index])
    visited[start_index] = 1
    component: List[Tuple[int, int]] = []

    while queue:
        index = queue.popleft()
        x = index % image.width
        y = image.height - 1 - (index // image.width)
        component.append((x, y))

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy
                if nx < 0 or nx >= image.width or ny < 0 or ny >= image.height:
                    continue

                neighbor_index = (image.height - 1 - ny) * image.width + nx
                if visited[neighbor_index]:
                    continue

                occupancy = occupancy_probability(image.pixels[neighbor_index], info)
                if occupancy >= info.occupied_thresh:
                    visited[neighbor_index] = 1
                    queue.append(neighbor_index)

    return component


def component_to_candidate(
    component: List[Tuple[int, int]],
    image: ImageMap,
    info: MapInfo,
    args: argparse.Namespace,
) -> Optional[BarrelCandidate]:
    if len(component) < 4:
        return None

    xs = [cell[0] for cell in component]
    ys = [cell[1] for cell in component]
    width_x = (max(xs) - min(xs) + 1) * info.resolution
    width_y = (max(ys) - min(ys) + 1) * info.resolution
    diameter = max(width_x, width_y)
    minor_diameter = min(width_x, width_y)

    if diameter < args.min_diameter or diameter > args.max_diameter:
        return None

    minor_major_ratio = minor_diameter / diameter if diameter > 0.0 else 0.0
    if minor_major_ratio < args.min_minor_major_ratio:
        return None

    roundness = component_roundness(component)
    if roundness < args.min_roundness:
        return None

    circularity = component_circularity(component)
    if circularity < args.min_circularity:
        return None

    corner_fill_ratio = component_corner_fill_ratio(component)
    if corner_fill_ratio > args.max_corner_fill_ratio:
        return None

    bounding_box_fill_ratio = len(component) / (
        (max(xs) - min(xs) + 1) * (max(ys) - min(ys) + 1)
    )
    if bounding_box_fill_ratio > args.max_bounding_box_fill_ratio:
        return None

    center_cell_x = sum(xs) / len(xs) + 0.5
    center_cell_y = sum(ys) / len(ys) + 0.5
    center_x, center_y = cell_to_world(center_cell_x, center_cell_y, info)
    free_ring_ratio, occupied_ring_ratio, unknown_ring_ratio = free_space_ring_ratios(
        component,
        center_cell_x,
        center_cell_y,
        diameter,
        image,
        info,
    )
    if free_ring_ratio < args.min_free_ring_ratio:
        return None
    if occupied_ring_ratio > args.max_occupied_ring_ratio:
        return None
    if unknown_ring_ratio > args.max_unknown_ring_ratio:
        return None
    context_occupied_ratio = hough_context_occupied_ratio(
        int(round(center_cell_x)),
        image.height - 1 - int(round(center_cell_y)),
        int(round(diameter * 0.5 / info.resolution)),
        image,
        info,
    )
    if context_occupied_ratio > args.max_context_occupied_ratio:
        return None
    circle_support_ratio, center_occupied_ratio = circle_template_ratios(
        center_cell_x,
        center_cell_y,
        diameter,
        image,
        info,
    )
    if circle_support_ratio < args.min_circle_support_ratio:
        return None
    if center_occupied_ratio > args.max_center_occupied_ratio:
        return None
    radial_cv, straight_edge_ratio = component_circle_quality(component)
    if radial_cv > args.max_radial_cv:
        return None
    if straight_edge_ratio > args.max_straight_edge_ratio:
        return None

    occupied_area = len(component) * info.resolution * info.resolution
    expected_circle_area = math.pi * (diameter * 0.5) ** 2
    fill_ratio = min(occupied_area / expected_circle_area, 1.0)
    size_score = score_size(diameter, args.min_diameter, args.max_diameter)
    score = (
        0.35 * roundness
        + 0.20 * circularity
        + 0.15 * (1.0 - corner_fill_ratio)
        + 0.10 * (1.0 - abs(bounding_box_fill_ratio - math.pi / 4.0))
        + 0.15 * size_score
        + 0.05 * fill_ratio
        + 0.08 * free_ring_ratio
        - 0.05 * occupied_ring_ratio
        - 0.05 * unknown_ring_ratio
        + 0.18 * circle_support_ratio
        + 0.08 * (1.0 - center_occupied_ratio)
        + 0.08 * (1.0 - radial_cv)
        - 0.05 * straight_edge_ratio
    )
    if score < args.min_score:
        return None

    return BarrelCandidate(
        center_x=center_x,
        center_y=center_y,
        diameter=diameter,
        width_x=width_x,
        width_y=width_y,
        occupied_cells=len(component),
        roundness=roundness,
        circularity=circularity,
        corner_fill_ratio=corner_fill_ratio,
        bounding_box_fill_ratio=bounding_box_fill_ratio,
        fill_ratio=fill_ratio,
        free_ring_ratio=free_ring_ratio,
        occupied_ring_ratio=occupied_ring_ratio,
        unknown_ring_ratio=unknown_ring_ratio,
        circle_support_ratio=circle_support_ratio,
        center_occupied_ratio=center_occupied_ratio,
        radial_cv=radial_cv,
        straight_edge_ratio=straight_edge_ratio,
        square_corner_ratio=corner_fill_ratio,
        score=score,
    )


def component_circle_quality(component: List[Tuple[int, int]]) -> Tuple[float, float]:
    mean_x = sum(cell[0] for cell in component) / len(component)
    mean_y = sum(cell[1] for cell in component) / len(component)
    radii = [math.hypot(x - mean_x, y - mean_y) for x, y in component]
    mean_radius = sum(radii) / len(radii)
    if mean_radius <= 1e-6:
        radial_cv = 1.0
    else:
        radial_std = math.sqrt(
            sum((radius - mean_radius) ** 2 for radius in radii) / len(radii)
        )
        radial_cv = radial_std / mean_radius

    cells = set(component)
    boundary_cells = [
        (x, y)
        for x, y in component
        if any(
            neighbor not in cells
            for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1))
        )
    ]
    if not boundary_cells:
        return radial_cv, 1.0

    straight_cells = 0
    for x, y in boundary_cells:
        horizontal = (x - 1, y) in cells and (x + 1, y) in cells
        vertical = (x, y - 1) in cells and (x, y + 1) in cells
        if horizontal or vertical:
            straight_cells += 1

    return radial_cv, straight_cells / len(boundary_cells)


def circle_template_ratios(
    center_cell_x: float,
    center_cell_y: float,
    diameter: float,
    image: ImageMap,
    info: MapInfo,
) -> Tuple[float, float]:
    radius_cells = max(2.0, diameter * 0.5 / info.resolution)
    circle_bins = 32
    supported_bins = 0

    for bin_index in range(circle_bins):
        angle = 2.0 * math.pi * bin_index / circle_bins
        if circle_bin_has_occupied_cell(
            center_cell_x,
            center_cell_y,
            radius_cells,
            angle,
            image,
            info,
        ):
            supported_bins += 1

    center_radius = max(1.0, radius_cells * 0.45)
    center_x = int(round(center_cell_x))
    center_y = int(round(center_cell_y))
    center_cells = int(math.ceil(center_radius))
    occupied_center_cells = 0
    total_center_cells = 0

    for y in range(center_y - center_cells, center_y + center_cells + 1):
        for x in range(center_x - center_cells, center_x + center_cells + 1):
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue
            distance = math.hypot(
                x + 0.5 - center_cell_x,
                y + 0.5 - center_cell_y,
            )
            if distance > center_radius:
                continue

            total_center_cells += 1
            image_index = (image.height - 1 - y) * image.width + x
            occupancy = occupancy_probability(image.pixels[image_index], info)
            if occupancy >= info.occupied_thresh:
                occupied_center_cells += 1

    center_occupied_ratio = (
        occupied_center_cells / total_center_cells
        if total_center_cells
        else 1.0
    )
    return supported_bins / circle_bins, center_occupied_ratio


def circle_bin_has_occupied_cell(
    center_cell_x: float,
    center_cell_y: float,
    radius_cells: float,
    angle: float,
    image: ImageMap,
    info: MapInfo,
) -> bool:
    for radius_offset in (-1.5, 0.0, 1.5):
        radius = radius_cells + radius_offset
        if radius <= 0.0:
            continue
        for angle_offset in (-0.06, 0.0, 0.06):
            x = int(round(center_cell_x + radius * math.cos(angle + angle_offset)))
            y = int(round(center_cell_y + radius * math.sin(angle + angle_offset)))
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue

            image_index = (image.height - 1 - y) * image.width + x
            occupancy = occupancy_probability(image.pixels[image_index], info)
            if occupancy >= info.occupied_thresh:
                return True

    return False


def free_space_ring_ratios(
    component: List[Tuple[int, int]],
    center_cell_x: float,
    center_cell_y: float,
    diameter: float,
    image: ImageMap,
    info: MapInfo,
) -> Tuple[float, float, float]:
    component_cells = set(component)
    radius_cells = max(1.0, diameter * 0.5 / info.resolution)
    inner_radius = radius_cells + max(1.0, 0.08 / info.resolution)
    outer_radius = radius_cells + max(2.0, 0.35 / info.resolution)
    center_x = int(round(center_cell_x))
    center_y = int(round(center_cell_y))
    outer_cells = int(math.ceil(outer_radius))

    free_cells = 0
    occupied_cells = 0
    unknown_cells = 0

    for y in range(center_y - outer_cells, center_y + outer_cells + 1):
        for x in range(center_x - outer_cells, center_x + outer_cells + 1):
            if x < 0 or x >= image.width or y < 0 or y >= image.height:
                continue
            if (x, y) in component_cells:
                continue

            distance = math.hypot(x + 0.5 - center_cell_x, y + 0.5 - center_cell_y)
            if distance < inner_radius or distance > outer_radius:
                continue

            image_index = (image.height - 1 - y) * image.width + x
            occupancy = occupancy_probability(image.pixels[image_index], info)
            if occupancy >= info.occupied_thresh:
                occupied_cells += 1
            elif occupancy <= info.free_thresh:
                free_cells += 1
            else:
                unknown_cells += 1

    total = free_cells + occupied_cells + unknown_cells
    if total == 0:
        return 0.0, 1.0, 0.0

    return (
        free_cells / total,
        occupied_cells / total,
        unknown_cells / total,
    )


def component_roundness(component: List[Tuple[int, int]]) -> float:
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


def component_circularity(component: List[Tuple[int, int]]) -> float:
    cells = set(component)
    exposed_edges = 0
    for x, y in cells:
        for neighbor in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if neighbor not in cells:
                exposed_edges += 1
    if exposed_edges == 0:
        return 0.0
    return min(4.0 * math.pi * len(cells) / (exposed_edges * exposed_edges), 1.0)


def component_corner_fill_ratio(component: List[Tuple[int, int]]) -> float:
    cells = set(component)
    xs = [cell[0] for cell in component]
    ys = [cell[1] for cell in component]
    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)
    corner_width = max(1, math.ceil((max_x - min_x + 1) * 0.25))
    corner_height = max(1, math.ceil((max_y - min_y + 1) * 0.25))
    corner_cells = set()

    x_ranges = (
        range(min_x, min_x + corner_width),
        range(max_x - corner_width + 1, max_x + 1),
    )
    y_ranges = (
        range(min_y, min_y + corner_height),
        range(max_y - corner_height + 1, max_y + 1),
    )
    for x_range in x_ranges:
        for y_range in y_ranges:
            for x in x_range:
                for y in y_range:
                    corner_cells.add((x, y))

    return sum(1 for cell in corner_cells if cell in cells) / len(corner_cells)


def score_size(diameter: float, minimum: float, maximum: float) -> float:
    if minimum <= 0.0 and maximum <= 0.0:
        return 1.0
    if minimum <= 0.0:
        return 1.0 if diameter <= maximum else max(0.0, maximum / diameter)
    if maximum <= 0.0:
        return 1.0 if diameter >= minimum else max(0.0, diameter / minimum)

    midpoint = 0.5 * (minimum + maximum)
    half_range = 0.5 * (maximum - minimum)
    if half_range <= 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(diameter - midpoint) / half_range)


def cell_to_world(cell_x: float, cell_y: float, info: MapInfo) -> Tuple[float, float]:
    local_x = cell_x * info.resolution
    local_y = cell_y * info.resolution
    cos_yaw = math.cos(info.origin_yaw)
    sin_yaw = math.sin(info.origin_yaw)
    return (
        info.origin_x + local_x * cos_yaw - local_y * sin_yaw,
        info.origin_y + local_x * sin_yaw + local_y * cos_yaw,
    )


def world_to_cell(x: float, y: float, info: MapInfo) -> Tuple[int, int]:
    dx = x - info.origin_x
    dy = y - info.origin_y
    cos_yaw = math.cos(info.origin_yaw)
    sin_yaw = math.sin(info.origin_yaw)
    local_x = dx * cos_yaw + dy * sin_yaw
    local_y = -dx * sin_yaw + dy * cos_yaw
    return round(local_x / info.resolution), round(local_y / info.resolution)


def parse_manual_barrels(values: Iterable[str]) -> List[BarrelCandidate]:
    barrels = []
    for value in values:
        pieces = value.split(',')
        if len(pieces) != 2:
            raise ValueError(f'--barrel must be X,Y, got {value!r}')
        x = float(pieces[0])
        y = float(pieces[1])
        barrels.append(
            BarrelCandidate(
                center_x=x,
                center_y=y,
                diameter=0.45,
                width_x=0.45,
                width_y=0.45,
                occupied_cells=0,
                roundness=1.0,
                circularity=1.0,
                corner_fill_ratio=0.0,
                bounding_box_fill_ratio=0.0,
                fill_ratio=1.0,
                free_ring_ratio=1.0,
                occupied_ring_ratio=0.0,
                unknown_ring_ratio=0.0,
                circle_support_ratio=1.0,
                center_occupied_ratio=0.0,
                radial_cv=0.0,
                straight_edge_ratio=0.0,
                square_corner_ratio=0.0,
                score=1.0,
            )
        )
    return barrels


def merge_close_candidates(
    candidates: List[BarrelCandidate],
    merge_distance: float,
) -> List[BarrelCandidate]:
    merged: List[BarrelCandidate] = []
    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate = any(
            math.hypot(
                candidate.center_x - existing.center_x,
                candidate.center_y - existing.center_y,
            )
            <= merge_distance
            for existing in merged
        )
        if not duplicate:
            merged.append(candidate)
    return merged


def write_barrel_yaml(path: str, candidates: List[BarrelCandidate]) -> None:
    now_sec = round(time.time(), 3)
    barrels = []
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

    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as file:
        yaml.safe_dump({'barrels': barrels}, file, sort_keys=False)


def write_annotated_pgm(
    path: str,
    image: ImageMap,
    info: MapInfo,
    candidates: List[BarrelCandidate],
) -> None:
    color_output = path.lower().endswith('.ppm')
    if color_output:
        color_pixels = [(pixel, pixel, pixel) for pixel in image.pixels]
    else:
        pixels = image.pixels[:]

    for candidate in candidates:
        center_x, center_y = world_to_cell(candidate.center_x, candidate.center_y, info)
        candidate_radius = 0.5 * candidate.diameter / info.resolution
        visibility_radius = 0.55 / info.resolution
        radius_cells = max(4, round(candidate_radius + visibility_radius))
        thickness_cells = max(2, round(radius_cells * 0.32))
        for y in range(center_y - radius_cells, center_y + radius_cells + 1):
            for x in range(center_x - radius_cells, center_x + radius_cells + 1):
                if x < 0 or x >= image.width or y < 0 or y >= image.height:
                    continue
                distance = math.hypot(x - center_x, y - center_y)
                if (
                    distance <= radius_cells
                    and distance >= radius_cells - thickness_cells
                ):
                    row = image.height - 1 - y
                    pixel_index = row * image.width + x
                    if color_output:
                        color_pixels[pixel_index] = (255, 0, 0)
                    else:
                        pixels[pixel_index] = 0

    with open(path, 'wb') as file:
        if color_output:
            file.write(f'P6\n{image.width} {image.height}\n255\n'.encode('ascii'))
            for red, green, blue in color_pixels:
                file.write(bytes((red, green, blue)))
        else:
            file.write(f'P5\n{image.width} {image.height}\n255\n'.encode('ascii'))
            file.write(bytes(pixels))


def discover_map_yamls(directory: str) -> List[str]:
    map_paths = []
    for name in sorted(os.listdir(directory)):
        if not name.endswith(('.yaml', '.yml')):
            continue

        path = os.path.join(directory, name)
        try:
            with open(path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file) or {}
        except (OSError, yaml.YAMLError):
            continue

        if not isinstance(data, dict):
            continue

        image = data.get('image')
        if image is None or 'resolution' not in data:
            continue

        image_path = str(image)
        if not os.path.isabs(image_path):
            image_path = os.path.join(directory, image_path)
        if os.path.exists(image_path):
            map_paths.append(path)

    return map_paths


def output_paths_for_map(
    map_yaml: str,
    args: argparse.Namespace,
    multiple_maps: bool,
) -> Tuple[str, Optional[str]]:
    if not multiple_maps:
        return args.output, args.annotated_image

    stem = os.path.splitext(os.path.basename(map_yaml))[0]
    output_path = os.path.join(args.output_dir, f'{stem}_barrel_target.yaml')
    annotated_path = os.path.join(args.annotated_dir, f'{stem}_barrels.ppm')
    return output_path, annotated_path


def process_map(
    map_yaml: str,
    output_path: str,
    annotated_path: Optional[str],
    args: argparse.Namespace,
) -> List[BarrelCandidate]:
    info = load_map_info(map_yaml)
    info.occupied_thresh = int(args.occupied_threshold) / 100.0
    info.free_thresh = int(args.free_threshold) / 100.0
    image = None
    needs_image = not args.no_auto or bool(annotated_path)
    if needs_image:
        image = load_pgm(info.image_path)

    candidates: List[BarrelCandidate] = []
    if not args.no_auto:
        if image is None:
            raise RuntimeError('map image is required for automatic detection')
        automatic_candidates = detect_barrels(image, info, args)
        if args.max_auto_barrels > 0:
            automatic_candidates = automatic_candidates[:args.max_auto_barrels]
        candidates.extend(automatic_candidates)
    candidates.extend(parse_manual_barrels(args.barrel))
    candidates = merge_close_candidates(candidates, args.merge_distance)
    if args.count > 0:
        candidates = candidates[:args.count]

    write_barrel_yaml(output_path, candidates)
    if annotated_path:
        if image is None:
            raise RuntimeError('map image is required for annotated output')
        directory = os.path.dirname(annotated_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        write_annotated_pgm(annotated_path, image, info, candidates)

    return candidates


def main() -> None:
    args = parse_args()
    if args.all_maps:
        map_yamls = discover_map_yamls(os.getcwd())
    else:
        map_yamls = args.map_yaml

    if not map_yamls:
        raise SystemExit('No map YAML files provided or discovered.')

    multiple_maps = len(map_yamls) > 1 or args.all_maps
    if multiple_maps:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.annotated_dir, exist_ok=True)

    for map_yaml in map_yamls:
        output_path, annotated_path = output_paths_for_map(
            map_yaml,
            args,
            multiple_maps,
        )
        candidates = process_map(map_yaml, output_path, annotated_path, args)

        print(f'{map_yaml}: wrote {len(candidates)} barrel(s) to {output_path}')
        if annotated_path:
            print(f'  preview: {annotated_path}')
        for candidate in candidates:
            print(
                f'  x={candidate.center_x:.3f}, y={candidate.center_y:.3f}, '
                f'width={candidate.diameter:.3f}, score={candidate.score:.3f}'
            )


if __name__ == '__main__':
    main()
