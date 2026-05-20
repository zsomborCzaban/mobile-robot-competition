import os
import signal
import subprocess
import threading
import time
import tkinter as tk
from typing import Dict, List

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformException, TransformListener


class MissionUIButton(Node):
    def __init__(self) -> None:
        super().__init__('mission_ui_button')
        self.cli_calc = self.create_client(Trigger, 'calculate_target')
        self.cli_nav = self.create_client(Trigger, 'start_navigation')
        self.cli_pause = self.create_client(Trigger, 'pause_navigation')
        self.cli_res = self.create_client(Trigger, 'resume_navigation')
        self.cli_stop = self.create_client(Trigger, 'stop_navigation')
        self.mission_param_cli = self.create_client(
            SetParameters,
            '/barrel_mission_controller/set_parameters',
        )
        self.detector_param_cli = self.create_client(
            SetParameters,
            '/ground_truth_barrel_detector/set_parameters',
        )
        
        self.processes: Dict[str, subprocess.Popen] = {}
        self.status_label = None
        self.readiness_labels: Dict[str, tk.Label] = {}
        self.last_map_time = 0.0
        self.last_scan_time = 0.0
        self.last_amcl_pose_time = 0.0
        self.last_tf_time = 0.0
        self.last_map_odom_tf_time = 0.0
        self.route_ready = False
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        static_map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        live_map_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            static_map_qos,
        )
        self.create_subscription(
            OccupancyGrid,
            '/map',
            self.map_callback,
            live_map_qos,
        )
        self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            '/amcl_pose',
            self.amcl_pose_callback,
            10,
        )
        self.create_subscription(
            TFMessage,
            '/tf',
            self.tf_callback,
            50,
        )
        self.create_subscription(
            String,
            'mission_status',
            self.mission_status_callback,
            10,
        )
        self.create_timer(1.0, self.update_readiness)

    def set_status_label(self, status_label) -> None:
        self.status_label = status_label

    def set_readiness_labels(self, labels: Dict[str, tk.Label]) -> None:
        self.readiness_labels = labels

    def map_callback(self, _message: OccupancyGrid) -> None:
        self.last_map_time = time.monotonic()

    def scan_callback(self, _message: LaserScan) -> None:
        self.last_scan_time = time.monotonic()

    def amcl_pose_callback(self, _message: PoseWithCovarianceStamped) -> None:
        self.last_amcl_pose_time = time.monotonic()

    def tf_callback(self, message: TFMessage) -> None:
        now = time.monotonic()
        self.last_tf_time = now
        for transform in message.transforms:
            parent_frame = transform.header.frame_id.lstrip('/')
            child_frame = transform.child_frame_id.lstrip('/')
            if parent_frame == 'map' and child_frame == 'odom':
                self.last_map_odom_tf_time = now

    def mission_status_callback(self, message: String) -> None:
        if self.status_label is None:
            return

        text = message.data.lower()
        if 'calculated shortest route' in text:
            self.route_ready = True
        elif 'mission stopped' in text or 'calculate the target' in text:
            self.route_ready = False

        self.set_status(self.status_label, message.data, 'blue')

    def update_readiness(self) -> None:
        now = time.monotonic()
        map_age = now - self.last_map_time if self.last_map_time else None
        scan_age = now - self.last_scan_time if self.last_scan_time else None
        amcl_pose_age = (
            now - self.last_amcl_pose_time if self.last_amcl_pose_time else None
        )
        tf_age = now - self.last_tf_time if self.last_tf_time else None
        map_odom_tf_age = (
            now - self.last_map_odom_tf_time if self.last_map_odom_tf_time else None
        )
        map_ready = self.last_map_time > 0.0
        scan_ready = scan_age is not None and scan_age <= 3.0
        amcl_pose_ready = amcl_pose_age is not None and amcl_pose_age <= 5.0
        tf_ready = tf_age is not None and tf_age <= 2.0
        map_odom_tf_ready = map_odom_tf_age is not None and map_odom_tf_age <= 5.0
        localized, tf_detail = self.localization_status()
        mission_ready = self.cli_calc.service_is_ready()
        detector_ready = self.detector_param_cli.service_is_ready()
        particle_publishers = self.count_publishers('/particle_cloud')
        amcl_pose_publishers = self.count_publishers('/amcl_pose')

        self.set_readiness(
            'map',
            map_ready,
            f'/map received{self.age_suffix(map_age)}' if map_ready else 'waiting for /map',
        )
        self.set_readiness(
            'scan',
            scan_ready,
            f'/scan live{self.age_suffix(scan_age)}' if scan_ready else 'waiting for live /scan',
        )
        self.set_readiness(
            'amcl_pose',
            amcl_pose_ready,
            (
                f'/amcl_pose live{self.age_suffix(amcl_pose_age)}'
                if amcl_pose_ready
                else f'waiting for /amcl_pose; publishers={amcl_pose_publishers}'
            ),
        )
        self.set_readiness(
            'particles',
            particle_publishers > 0,
            f'/particle_cloud publishers={particle_publishers}',
        )
        self.set_readiness(
            'tf',
            tf_ready,
            f'/tf live{self.age_suffix(tf_age)}' if tf_ready else 'waiting for /tf',
        )
        self.set_readiness(
            'map_odom',
            map_odom_tf_ready,
            (
                f'map->odom on /tf{self.age_suffix(map_odom_tf_age)}'
                if map_odom_tf_ready
                else 'waiting for AMCL map->odom TF'
            ),
        )
        self.set_readiness('localization', localized, tf_detail)
        self.set_readiness(
            'mission',
            mission_ready,
            'mission controller service online' if mission_ready else 'start mission controller',
        )
        self.set_readiness(
            'detector',
            detector_ready,
            'map detector online' if detector_ready else 'start map barrel detection',
        )
        self.set_readiness(
            'route',
            self.route_ready,
            'route calculated' if self.route_ready else 'calculate target path',
        )

    @staticmethod
    def age_suffix(age) -> str:
        if age is None:
            return ''
        return f' age={age:.1f}s'

    def localization_status(self):
        for base_frame in ('base_footprint', 'base_link'):
            try:
                transform = self.tf_buffer.lookup_transform(
                    'map',
                    base_frame,
                    rclpy.time.Time(),
                )
            except TransformException:
                continue
            translation = transform.transform.translation
            return (
                True,
                f'map->{base_frame} ({translation.x:.2f}, {translation.y:.2f})',
            )
        return False, 'set 2D Pose Estimate; waiting for map->base TF'

    def set_readiness(self, key: str, ready: bool, detail: str) -> None:
        label = self.readiness_labels.get(key)
        if label is None:
            return
        color = '#1b7f35' if ready else '#9a6700'
        prefix = 'READY' if ready else 'WAIT'
        label.after(0, lambda: label.config(text=f'{prefix}: {detail}', fg=color))

    @staticmethod
    def set_status(status_label, text: str, color: str) -> None:
        status_label.after(0, lambda: status_label.config(text=text, fg=color))

    def start_mission_controller(self, status_label, barrel_count_text: str) -> None:
        expected_count = self.parse_expected_barrel_count(barrel_count_text, status_label)
        if expected_count is None:
            return

        if self.cli_calc.wait_for_service(timeout_sec=0.2):
            self.set_status(status_label, 'Mission controller already running.', 'green')
            self.set_mission_expected_count(status_label, expected_count)
            return

        self.start_process(
            'mission_controller',
            [
                'ros2',
                'run',
                'barrel_lidar_detector',
                'mission_controller',
                '--ros-args',
                '-p',
                f'expected_barrel_count:={expected_count}',
            ],
            status_label,
        )

    def set_mission_expected_count(self, status_label, expected_count: int) -> None:
        if not self.mission_param_cli.wait_for_service(timeout_sec=0.1):
            return

        request = SetParameters.Request()
        parameter = Parameter()
        parameter.name = 'expected_barrel_count'
        parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_INTEGER,
            integer_value=int(expected_count),
        )
        request.parameters = [parameter]
        future = self.mission_param_cli.call_async(request)
        future.add_done_callback(
            lambda done: self.mission_expected_count_response_callback(
                done,
                status_label,
                expected_count,
            )
        )

    def mission_expected_count_response_callback(
        self,
        future,
        status_label,
        expected_count: int,
    ) -> None:
        try:
            response = future.result()
            successful = bool(response.results and response.results[0].successful)
        except Exception as exc:
            self.set_status(status_label, f'Mission target update failed: {exc}', 'red')
            return

        if successful:
            label = str(expected_count) if expected_count > 0 else 'unlimited'
            self.set_status(status_label, f'Mission expected barrels: {label}.', 'blue')
        else:
            reason = response.results[0].reason if response.results else 'unknown'
            self.set_status(status_label, f'Mission target rejected: {reason}', 'red')

    def start_detectors(
        self,
        status_label,
        barrel_count_text: str,
        sensitivity_value: int,
    ) -> None:
        expected_count = self.parse_expected_barrel_count(barrel_count_text, status_label)
        if expected_count is None:
            return
        sensitivity = self.clamp_sensitivity(sensitivity_value)

        if self.detector_param_cli.wait_for_service(timeout_sec=0.2):
            self.set_status(
                status_label,
                'Map barrel detector already running.',
                'green',
            )
            self.set_detector_parameters(status_label, expected_count, sensitivity)
            return

        self.start_process(
            'ground_truth_barrel_detector',
            [
                'ros2',
                'run',
                'barrel_lidar_detector',
                'ground_truth_barrel_detector',
                '--ros-args',
                '-p',
                f'expected_barrel_count:={expected_count}',
                '-p',
                f'detection_sensitivity:={sensitivity}',
            ],
            status_label,
        )

    def set_detector_parameters(
        self,
        status_label,
        expected_count: int,
        sensitivity: int,
    ) -> None:
        request = SetParameters.Request()

        count_parameter = Parameter()
        count_parameter.name = 'expected_barrel_count'
        count_parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_INTEGER,
            integer_value=int(expected_count),
        )

        sensitivity_parameter = Parameter()
        sensitivity_parameter.name = 'detection_sensitivity'
        sensitivity_parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE,
            double_value=float(sensitivity),
        )

        request.parameters = [count_parameter, sensitivity_parameter]
        future = self.detector_param_cli.call_async(request)
        future.add_done_callback(
            lambda done: self.detector_parameters_response_callback(
                done,
                status_label,
                expected_count,
                sensitivity,
            )
        )

    def detector_parameters_response_callback(
        self,
        future,
        status_label,
        expected_count: int,
        sensitivity: int,
    ) -> None:
        try:
            response = future.result()
            successful = all(result.successful for result in response.results)
        except Exception as exc:
            self.set_status(
                status_label,
                f'Detector parameter update failed: {exc}',
                'red',
            )
            return

        if successful:
            self.set_status(
                status_label,
                (
                    'Detector running. '
                    f'Expected barrels: {expected_count}, sensitivity: {sensitivity}.'
                ),
                'green',
            )
        else:
            reason = response.results[0].reason if response.results else 'unknown'
            self.set_status(status_label, f'Detector parameter rejected: {reason}', 'red')

    def set_detection_sensitivity(self, status_label, sensitivity_value: int) -> None:
        sensitivity = self.clamp_sensitivity(sensitivity_value)
        if not self.detector_param_cli.wait_for_service(timeout_sec=0.05):
            self.set_status(
                status_label,
                f'Detection sensitivity {sensitivity}; applies when detector starts.',
                'gray',
            )
            return

        request = SetParameters.Request()
        parameter = Parameter()
        parameter.name = 'detection_sensitivity'
        parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_DOUBLE,
            double_value=float(sensitivity),
        )
        request.parameters = [parameter]
        future = self.detector_param_cli.call_async(request)
        future.add_done_callback(
            lambda done: self.sensitivity_response_callback(
                done,
                status_label,
                sensitivity,
            )
        )

    def sensitivity_response_callback(
        self,
        future,
        status_label,
        sensitivity: int,
    ) -> None:
        try:
            response = future.result()
            successful = bool(response.results and response.results[0].successful)
        except Exception as exc:
            self.set_status(status_label, f'Sensitivity update failed: {exc}', 'red')
            return

        if successful:
            self.set_status(
                status_label,
                f'Detection sensitivity set to {sensitivity}.',
                'blue',
            )
        else:
            reason = response.results[0].reason if response.results else 'unknown'
            self.set_status(status_label, f'Sensitivity rejected: {reason}', 'red')

    @staticmethod
    def clamp_sensitivity(value) -> int:
        try:
            sensitivity = int(float(value))
        except (TypeError, ValueError):
            return 50
        return max(0, min(100, sensitivity))

    def calculate_target_path(
        self,
        status_label,
        barrel_count_text: str,
        sensitivity_value: int,
    ) -> None:
        expected_count = self.parse_expected_barrel_count(barrel_count_text, status_label)
        if expected_count is None:
            return

        sensitivity = self.clamp_sensitivity(sensitivity_value)
        if self.detector_param_cli.wait_for_service(timeout_sec=0.05):
            self.set_detector_parameters(status_label, expected_count, sensitivity)

        if not self.mission_param_cli.wait_for_service(timeout_sec=0.1):
            self.trigger_action(self.cli_calc, 'Calculating Target', status_label)
            return

        request = SetParameters.Request()
        parameter = Parameter()
        parameter.name = 'expected_barrel_count'
        parameter.value = ParameterValue(
            type=ParameterType.PARAMETER_INTEGER,
            integer_value=int(expected_count),
        )
        request.parameters = [parameter]
        future = self.mission_param_cli.call_async(request)
        future.add_done_callback(
            lambda done: self.calculate_after_mission_parameter(
                done,
                status_label,
            )
        )

    def calculate_after_mission_parameter(self, future, status_label) -> None:
        try:
            response = future.result()
            successful = bool(response.results and response.results[0].successful)
        except Exception as exc:
            self.set_status(status_label, f'Mission target update failed: {exc}', 'red')
            return

        if not successful:
            reason = response.results[0].reason if response.results else 'unknown'
            self.set_status(status_label, f'Mission target rejected: {reason}', 'red')
            return

        self.trigger_action(self.cli_calc, 'Calculating Target', status_label)

    def parse_expected_barrel_count(self, barrel_count_text: str, status_label):
        value = barrel_count_text.strip()
        if not value:
            return 0

        try:
            count = int(value)
        except ValueError:
            self.set_status(
                status_label,
                'Expected barrels must be a whole number.',
                'red',
            )
            return None

        if count < 0:
            self.set_status(status_label, 'Expected barrels cannot be negative.', 'red')
            return None

        return count

    def start_process(self, name: str, command: List[str], status_label) -> None:
        process = self.processes.get(name)
        if process is not None and process.poll() is None:
            self.set_status(status_label, f'{name} already running.', 'green')
            return

        try:
            self.processes[name] = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            self.set_status(status_label, f'Failed to start {name}: {exc}', 'red')
            return

        self.set_status(status_label, f'Started {name}.', 'green')
        status_label.after(1000, lambda: self.report_process_exit(name, status_label))

    def report_process_exit(self, name: str, status_label) -> None:
        process = self.processes.get(name)
        if process is None or process.poll() is None:
            return

        self.processes.pop(name, None)
        self.set_status(
            status_label,
            f'{name} exited immediately with code {process.returncode}.',
            'red',
        )

    def trigger_action(self, client, action_name, status_label) -> None:
        if not client.wait_for_service(timeout_sec=0.5):
            self.set_status(status_label, 'Error: Controller offline.', 'red')
            return

        self.set_status(status_label, f'{action_name}...', 'blue')
        req = Trigger.Request()
        future = client.call_async(req)
        future.add_done_callback(lambda f: self.service_response_callback(f, status_label))

    def service_response_callback(self, future, status_label) -> None:
        try:
            response = future.result()
            color = 'green' if response.success else 'red'
            self.set_status(status_label, response.message, color)
        except Exception as exc:
            self.set_status(status_label, f'Service call failed: {exc}', 'red')

    def restart_ros_daemon(self, status_label) -> None:
        self.set_status(status_label, 'Restarting ROS 2 daemon...', 'orange')
        threading.Thread(
            target=self._restart_ros_daemon_worker,
            args=(status_label,),
            daemon=True,
        ).start()

    def _restart_ros_daemon_worker(self, status_label) -> None:
        try:
            stop_result = subprocess.run(
                ['ros2', 'daemon', 'stop'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10.0,
                check=False,
            )
            start_result = subprocess.run(
                ['ros2', 'daemon', 'start'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10.0,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            self.set_status(status_label, f'ROS 2 daemon restart failed: {exc}', 'red')
            return

        if start_result.returncode == 0:
            self.set_status(status_label, 'ROS 2 daemon restarted.', 'green')
            return

        details = (start_result.stderr or start_result.stdout or '').strip()
        if not details:
            details = f'stop={stop_result.returncode}, start={start_result.returncode}'
        self.set_status(status_label, f'ROS 2 daemon restart failed: {details}', 'red')

    def stop_all(self, status_label) -> None:
        self.set_status(status_label, 'Stopping mission and package nodes...', 'orange')
        cleanup_done = {'value': False}

        def cleanup(message: str) -> None:
            if cleanup_done['value']:
                return
            cleanup_done['value'] = True
            stopped = self.stop_named_processes(
                (
                    'mission_controller',
                    'ground_truth_barrel_detector',
                    'lidar_cluster_detector',
                    'map_shape_detector',
                ),
                include_stale=True,
            )
            if stopped:
                message = f'{message} Stopped: {", ".join(stopped)}.'
            self.set_status(status_label, message, 'orange')

        if self.cli_stop.wait_for_service(timeout_sec=0.3):
            req = Trigger.Request()
            future = self.cli_stop.call_async(req)

            def on_stop_done(done_future) -> None:
                try:
                    response = done_future.result()
                    message = response.message
                except Exception as exc:
                    message = f'Stop service failed: {exc}'

                status_label.after(0, lambda: cleanup(message))

            future.add_done_callback(on_stop_done)
            status_label.after(
                2000,
                lambda: cleanup('Stop service timed out; forcing package nodes down.'),
            )
        else:
            cleanup('Stop service offline; forcing package nodes down.')

    def stop_named_processes(self, names, include_stale: bool = False) -> List[str]:
        stopped: List[str] = []
        for name in names:
            process = self.processes.pop(name, None)
            if self.terminate_process_group(process):
                stopped.append(name)

        if include_stale:
            for name in names:
                if self.kill_stale_process(name) and name not in stopped:
                    stopped.append(name)

        return stopped

    @staticmethod
    def terminate_process_group(process) -> bool:
        if process is None or process.poll() is not None:
            return False

        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return False
        except OSError:
            process.terminate()

        try:
            process.wait(timeout=2.0)
            return True
        except subprocess.TimeoutExpired:
            pass

        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        except OSError:
            process.kill()

        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            return False

        return True

    @staticmethod
    def kill_stale_process(name: str) -> bool:
        patterns = (
            f'barrel_lidar_detector.*{name}',
            f'/barrel_lidar_detector/{name}',
            f'/{name}( |$)',
        )
        killed = False
        for pattern in patterns:
            result = subprocess.run(
                ['pkill', '-TERM', '-f', pattern],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            killed = killed or result.returncode == 0
        if killed:
            time.sleep(0.2)
            for pattern in patterns:
                subprocess.run(
                    ['pkill', '-KILL', '-f', pattern],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    check=False,
                )
        return killed

    def stop_child_processes(self) -> None:
        self.stop_named_processes(
            (
                'mission_controller',
                'ground_truth_barrel_detector',
                'lidar_cluster_detector',
                'map_shape_detector',
            ),
            include_stale=True,
        )


def run_ros_spin(node: MissionUIButton) -> None:
    rclpy.spin(node)


def add_legend_row(parent, color: str, title: str, detail: str) -> None:
    row = tk.Frame(parent)
    row.pack(fill='x', pady=1)

    swatch = tk.Canvas(row, width=18, height=14, highlightthickness=0)
    swatch.create_rectangle(2, 2, 16, 12, fill=color, outline='black')
    swatch.pack(side='left', padx=(0, 6))

    label = tk.Label(
        row,
        text=f'{title}: {detail}',
        anchor='w',
        justify='left',
        font=('Arial', 9),
        wraplength=285,
    )
    label.pack(side='left', fill='x', expand=True)


def create_marker_legend(root) -> None:
    legend = tk.LabelFrame(
        root,
        text='RViz marker legend',
        font=('Arial', 10, 'bold'),
        padx=8,
        pady=5,
    )
    legend.pack(fill='x', padx=10, pady=(4, 8))

    add_legend_row(legend, '#ff0000', 'Red', 'ground-truth barrels from the map')
    add_legend_row(legend, '#ffffff', 'White', 'barrel ID labels')


def create_readiness_panel(root):
    panel = tk.LabelFrame(
        root,
        text='Readiness',
        font=('Arial', 10, 'bold'),
        padx=8,
        pady=5,
    )
    panel.pack(fill='x', padx=10, pady=(0, 8))

    labels = {}
    rows = (
        ('map', 'Map'),
        ('scan', 'Laser scan'),
        ('amcl_pose', 'AMCL pose'),
        ('particles', 'AMCL particles'),
        ('tf', 'TF stream'),
        ('map_odom', 'AMCL TF'),
        ('localization', 'Localization'),
        ('mission', 'Mission controller'),
        ('detector', 'Barrel detector'),
        ('route', 'Route'),
    )
    for key, title in rows:
        row = tk.Frame(panel)
        row.pack(fill='x', pady=1)
        tk.Label(
            row,
            text=f'{title}:',
            width=17,
            anchor='w',
            font=('Arial', 9, 'bold'),
        ).pack(side='left')
        label = tk.Label(
            row,
            text='WAIT: checking...',
            anchor='w',
            justify='left',
            font=('Arial', 9),
            wraplength=230,
            fg='#9a6700',
        )
        label.pack(side='left', fill='x', expand=True)
        labels[key] = label

    return labels


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionUIButton()

    ros_thread = threading.Thread(target=run_ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    root = tk.Tk()
    root.title('TurtleBot Barrel Mission Pro')
    root.geometry('440x780')
    root.attributes('-topmost', True)

    status_lbl = tk.Label(root, text='Waiting for input...', font=('Arial', 10), wraplength=330)
    status_lbl.pack(fill='x', padx=10, pady=(8, 4))
    node.set_status_label(status_lbl)
    create_marker_legend(root)
    node.set_readiness_labels(create_readiness_panel(root))

    target_frame = tk.LabelFrame(
        root,
        text='Mission target',
        font=('Arial', 10, 'bold'),
        padx=8,
        pady=5,
    )
    target_frame.pack(fill='x', padx=10, pady=(0, 8))

    tk.Label(
        target_frame,
        text='Expected barrels',
        anchor='w',
        font=('Arial', 10),
    ).pack(side='left')

    expected_barrels_var = tk.StringVar(value='0')
    tk.Spinbox(
        target_frame,
        from_=0,
        to=20,
        width=5,
        textvariable=expected_barrels_var,
        font=('Arial', 10),
    ).pack(side='right')

    sensitivity_frame = tk.LabelFrame(
        root,
        text='Barrel detection sensitivity',
        font=('Arial', 10, 'bold'),
        padx=8,
        pady=5,
    )
    sensitivity_frame.pack(fill='x', padx=10, pady=(0, 8))

    sensitivity_var = tk.IntVar(value=50)
    sensitivity_value_lbl = tk.Label(
        sensitivity_frame,
        text='50',
        width=4,
        font=('Arial', 10, 'bold'),
    )
    sensitivity_value_lbl.pack(side='right', padx=(6, 0))

    pending_sensitivity_update = {'job': None}

    def on_sensitivity_change(value) -> None:
        sensitivity = MissionUIButton.clamp_sensitivity(value)
        sensitivity_value_lbl.config(text=str(sensitivity))
        pending_job = pending_sensitivity_update['job']
        if pending_job is not None:
            root.after_cancel(pending_job)
        pending_sensitivity_update['job'] = root.after(
            300,
            lambda: node.set_detection_sensitivity(status_lbl, sensitivity),
        )

    tk.Scale(
        sensitivity_frame,
        from_=0,
        to=100,
        orient='horizontal',
        variable=sensitivity_var,
        command=on_sensitivity_change,
        showvalue=False,
        resolution=1,
        length=285,
    ).pack(side='left', fill='x', expand=True)

    btn_controller = tk.Button(root, text='1. Start Mission Controller', font=('Arial', 11, 'bold'), bg='lightgray', fg='black',
                               command=lambda: node.start_mission_controller(status_lbl, expected_barrels_var.get()))
    btn_controller.pack(fill='x', padx=10, pady=2, ipady=8)

    btn_calc = tk.Button(root, text='2. Calculate Target Path', font=('Arial', 11, 'bold'), bg='orange', fg='black',
                         command=lambda: node.calculate_target_path(
                             status_lbl,
                             expected_barrels_var.get(),
                             sensitivity_var.get(),
                         ))
    btn_calc.pack(fill='x', padx=10, pady=(12, 2), ipady=8)

    btn_nav = tk.Button(root, text='3. START NAVIGATION', font=('Arial', 12, 'bold'), bg='green', fg='white',
                        command=lambda: node.trigger_action(node.cli_nav, "Starting Navigation", status_lbl))
    btn_nav.pack(fill='x', padx=10, pady=2, ipady=8)

    control_frame = tk.Frame(root)
    control_frame.pack(fill='x', padx=10, pady=5)
    
    btn_pause = tk.Button(control_frame, text="⏸ PAUSE", font=("Arial", 10, "bold"), bg="gold", fg="black", width=12,
                          command=lambda: node.trigger_action(node.cli_pause, "Pausing", status_lbl))
    btn_pause.pack(side='left', expand=True)
              
    btn_resume = tk.Button(control_frame, text="▶ RESUME", font=("Arial", 10, "bold"), bg="lightgreen", fg="black", width=12,
                           command=lambda: node.trigger_action(node.cli_res, "Resuming", status_lbl))
    btn_resume.pack(side='right', expand=True)

    tk.Button(root, text='Restart ROS 2 Daemon', font=('Arial', 10, 'bold'), bg='steelblue', fg='white',
              command=lambda: node.restart_ros_daemon(status_lbl)).pack(fill='x', padx=10, pady=2, ipady=6)

    tk.Button(root, text="⏹ STOP & RESET", font=("Arial", 11, "bold"), bg="red", fg="white",
              command=lambda: node.stop_all(status_lbl)).pack(fill='x', padx=10, pady=2, ipady=6)

    def on_close() -> None:
        node.stop_child_processes()
        node.destroy_node()
        rclpy.shutdown()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
