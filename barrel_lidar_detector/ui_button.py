import subprocess
import threading
import tkinter as tk
from typing import Dict, List

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger


class MissionUIButton(Node):
    def __init__(self) -> None:
        super().__init__('mission_ui_button')
        self.cli_calc = self.create_client(Trigger, 'calculate_target')
        self.cli_nav = self.create_client(Trigger, 'start_navigation')
        self.processes: Dict[str, subprocess.Popen] = {}

    def start_mission_controller(self, status_label) -> None:
        if self.cli_calc.wait_for_service(timeout_sec=0.2):
            status_label.config(text='Mission controller already running.', fg='green')
            return

        self.start_process(
            'mission_controller',
            ['ros2', 'run', 'barrel_lidar_detector', 'mission_controller'],
            status_label,
        )

    def start_detectors(self, status_label) -> None:
        self.start_process(
            'lidar_cluster_detector',
            ['ros2', 'run', 'barrel_lidar_detector', 'lidar_cluster_detector'],
            status_label,
        )
        self.start_process(
            'map_shape_detector',
            ['ros2', 'run', 'barrel_lidar_detector', 'map_shape_detector'],
            status_label,
        )

    def stop_detectors(self, status_label) -> None:
        stopped: List[str] = []
        for name in ('lidar_cluster_detector', 'map_shape_detector'):
            process = self.processes.get(name)
            if process is None or process.poll() is not None:
                continue

            process.terminate()
            stopped.append(name)

        if stopped:
            status_label.config(
                text='Stopped: ' + ', '.join(stopped),
                fg='orange',
            )
        else:
            status_label.config(text='No detector process started by this UI.', fg='gray')

    def start_process(self, name: str, command: List[str], status_label) -> None:
        process = self.processes.get(name)
        if process is not None and process.poll() is None:
            status_label.config(text=f'{name} already running.', fg='green')
            return

        try:
            self.processes[name] = subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as exc:
            status_label.config(text=f'Failed to start {name}: {exc}', fg='red')
            return

        status_label.config(text=f'Started {name}.', fg='green')

    def trigger_calc(self, status_label) -> None:
        if not self.cli_calc.wait_for_service(timeout_sec=1.0):
            status_label.config(text='Error: mission controller not running.', fg='red')
            return

        status_label.config(text='Calculating target...', fg='blue')
        req = Trigger.Request()
        future = self.cli_calc.call_async(req)
        future.add_done_callback(lambda f: self.service_response_callback(f, status_label))

    def trigger_nav(self, status_label) -> None:
        if not self.cli_nav.wait_for_service(timeout_sec=1.0):
            status_label.config(text='Error: mission controller not running.', fg='red')
            return

        status_label.config(text='Starting navigation...', fg='blue')
        req = Trigger.Request()
        future = self.cli_nav.call_async(req)
        future.add_done_callback(lambda f: self.service_response_callback(f, status_label))

    @staticmethod
    def service_response_callback(future, status_label) -> None:
        try:
            response = future.result()
        except Exception as exc:
            status_label.config(text=f'Service call failed: {exc}', fg='red')
            return

        color = 'green' if response.success else 'red'
        status_label.config(text=response.message, fg=color)

    def stop_child_processes(self) -> None:
        for process in self.processes.values():
            if process.poll() is None:
                process.terminate()


def run_ros_spin(node: MissionUIButton) -> None:
    rclpy.spin(node)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionUIButton()

    ros_thread = threading.Thread(target=run_ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    root = tk.Tk()
    root.title('TurtleBot Barrel Mission')
    root.geometry('360x360')
    root.attributes('-topmost', True)

    status_lbl = tk.Label(root, text='Waiting for input...', font=('Arial', 10), wraplength=330)

    btn_controller = tk.Button(
        root,
        text='1. Start Mission Controller',
        font=('Arial', 11, 'bold'),
        bg='lightgray',
        fg='black',
        command=lambda: node.start_mission_controller(status_lbl),
    )
    btn_controller.pack(expand=True, fill='both', padx=10, pady=5)

    btn_detection = tk.Button(
        root,
        text='2. Start LiDAR + Map Detection',
        font=('Arial', 11, 'bold'),
        bg='deepskyblue',
        fg='black',
        command=lambda: node.start_detectors(status_lbl),
    )
    btn_detection.pack(expand=True, fill='both', padx=10, pady=5)

    btn_stop_detection = tk.Button(
        root,
        text='Stop Detection',
        font=('Arial', 11, 'bold'),
        bg='gray',
        fg='white',
        command=lambda: node.stop_detectors(status_lbl),
    )
    btn_stop_detection.pack(expand=True, fill='both', padx=10, pady=5)

    btn_calc = tk.Button(
        root,
        text='3. Calculate Target',
        font=('Arial', 11, 'bold'),
        bg='orange',
        fg='black',
        command=lambda: node.trigger_calc(status_lbl),
    )
    btn_calc.pack(expand=True, fill='both', padx=10, pady=5)

    btn_nav = tk.Button(
        root,
        text='4. START NAVIGATION',
        font=('Arial', 11, 'bold'),
        bg='green',
        fg='white',
        command=lambda: node.trigger_nav(status_lbl),
    )
    btn_nav.pack(expand=True, fill='both', padx=10, pady=5)

    status_lbl.pack(pady=10)

    def on_close() -> None:
        node.stop_child_processes()
        node.destroy_node()
        rclpy.shutdown()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_close)
    root.mainloop()


if __name__ == '__main__':
    main()
