"""OAK-D Lite visual validator using DepthAI YOLOv8 (COCO bottle as barrel proxy)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool

try:
    import depthai as dai
except ImportError:  # pragma: no cover - optional at lint time
    dai = None  # type: ignore


# COCO class index for "bottle" (proxy object until a custom barrel blob is trained).
DEFAULT_TARGET_CLASS_ID = 39


class BarrelCameraValidator(Node):
    """Runs YOLOv8 on-device and publishes whether the proxy target is seen."""

    def __init__(self) -> None:
        super().__init__('barrel_camera_validator')

        # Swap this path (or override via ROS param) when switching to a custom barrel .blob.
        # Default is relative to the process working directory (e.g. repo root when launched from there).
        default_blob = os.environ.get(
            'BARREL_YOLO_BLOB_PATH',
            './models/yolov8n_coco_640x352.blob',
        )

        self.declare_parameter('model_blob_path', default_blob)
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('target_class_id', DEFAULT_TARGET_CLASS_ID)
        self.declare_parameter('iou_threshold', 0.5)
        self.declare_parameter('num_classes', 80)
        self.declare_parameter('camera_fps', 30.0)
        self.declare_parameter('preview_width', 640)
        self.declare_parameter('preview_height', 352)
        self.declare_parameter('confirm_topic', '/camera_barrel_confirmed')
        self.declare_parameter('inference_timer_period_sec', 0.05)

        self._confidence_threshold = float(self.get_parameter('confidence_threshold').value)
        self._target_class_id = int(self.get_parameter('target_class_id').value)
        self._blob_path = str(self.get_parameter('model_blob_path').value).strip()

        confirm_topic = str(self.get_parameter('confirm_topic').value)
        self._pub = self.create_publisher(Bool, confirm_topic, 10)

        self._device: Optional[dai.Device] = None
        self._q_rgb = None
        self._q_det = None

        if dai is None:
            self.get_logger().error(
                'depthai is not installed. Install with: pip install depthai'
            )
        elif not self._blob_path or not Path(self._blob_path).expanduser().is_file():
            self.get_logger().error(
                f'model_blob_path is missing or not a file: {self._blob_path!r}. '
                'Set ROS param model_blob_path or env BARREL_YOLO_BLOB_PATH to your .blob file.'
            )
        else:
            self._blob_path = str(Path(self._blob_path).expanduser().resolve())
            try:
                self._start_pipeline()
            except Exception as exc:  # pragma: no cover - hardware dependent
                self.get_logger().error(f'Failed to start DepthAI pipeline: {exc}')

        period = float(self.get_parameter('inference_timer_period_sec').value)
        self._timer = self.create_timer(period, self._timer_callback)

        if self._device is None:
            self._pub.publish(Bool(data=False))

        self.get_logger().info(
            f'Barrel camera validator publishing Bool on {confirm_topic} '
            f'(target_class_id={self._target_class_id}, confidence>{self._confidence_threshold}). '
            f'YOLO blob: {self._blob_path or "(none)"}'
        )

    def _start_pipeline(self) -> None:
        if dai is None:
            return

        pipeline = dai.Pipeline()

        cam_rgb = pipeline.create(dai.node.ColorCamera)
        detection_network = pipeline.create(dai.node.YoloDetectionNetwork)
        xout_rgb = pipeline.create(dai.node.XLinkOut)
        nn_out = pipeline.create(dai.node.XLinkOut)

        xout_rgb.setStreamName('rgb')
        nn_out.setStreamName('nn')

        preview_w = int(self.get_parameter('preview_width').value)
        preview_h = int(self.get_parameter('preview_height').value)
        cam_rgb.setPreviewSize(preview_w, preview_h)
        cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam_rgb.setInterleaved(False)
        cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam_rgb.setFps(float(self.get_parameter('camera_fps').value))

        # Keep NN threshold slightly below the logical threshold so we can apply strict > in software.
        nn_floor = max(0.01, min(0.49, self._confidence_threshold - 1e-3))
        detection_network.setConfidenceThreshold(nn_floor)
        detection_network.setNumClasses(int(self.get_parameter('num_classes').value))
        detection_network.setCoordinateSize(4)
        detection_network.setIouThreshold(float(self.get_parameter('iou_threshold').value))
        detection_network.setBlobPath(self._blob_path)
        detection_network.setNumInferenceThreads(2)
        detection_network.input.setBlocking(False)

        cam_rgb.preview.link(detection_network.input)
        detection_network.passthrough.link(xout_rgb.input)
        detection_network.out.link(nn_out.input)

        self._device = dai.Device(pipeline)
        self._q_rgb = self._device.getOutputQueue(name='rgb', maxSize=4, blocking=False)
        self._q_det = self._device.getOutputQueue(name='nn', maxSize=4, blocking=False)
        self.get_logger().info('DepthAI device connected; YOLOv8 inference running.')

    def _timer_callback(self) -> None:
        if self._device is None or self._q_det is None:
            return

        if self._q_rgb is not None:
            self._q_rgb.tryGet()

        in_det = self._q_det.tryGet()
        if in_det is None:
            return

        detections = in_det.detections
        confirmed = any(
            det.label == self._target_class_id and det.confidence > self._confidence_threshold
            for det in detections
        )
        self._pub.publish(Bool(data=confirmed))

    def destroy_node(self) -> bool:
        if self._device is not None:
            try:
                self._device.close()
            except Exception:
                pass
            self._device = None
        return super().destroy_node()


def main(args: Optional[list] = None) -> None:
    rclpy.init(args=args)
    node = BarrelCameraValidator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
