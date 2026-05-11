import subprocess
import threading
import tkinter as tk
from typing import Dict, List

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from std_srvs.srv import Trigger


class MissionUIButton(Node):
    def __init__(self) -> None:
        super().__init__('mission_ui_button')
        # All 5 Service Clients
        self.cli_calc = self.create_client(Trigger, 'calculate_target')
        self.cli_nav = self.create_client(Trigger, 'start_navigation')
        self.cli_pause = self.create_client(Trigger, 'pause_navigation')
        self.cli_res = self.create_client(Trigger, 'resume_navigation')
        self.cli_stop = self.create_client(Trigger, 'stop_navigation')

        # Safety switch: mission_controller subscribes and toggles strict LiDAR vs camera policy.
        self.pub_strict_camera_validation = self.create_publisher(
            Bool,
            '/strict_camera_validation',
            10,
        )

        self.processes: Dict[str, subprocess.Popen] = {}

    def publish_strict_camera_validation(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self.pub_strict_camera_validation.publish(msg)

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
            status_label.config(text='Stopped: ' + ', '.join(stopped), fg='orange')
        else:
            status_label.config(text='No detector process started by this UI.', fg='gray')

    def start_process(self, name: str, command: List[str], status_label) -> None:
        process = self.processes.get(name)
        if process is not None and process.poll() is None:
            status_label.config(text=f'{name} already running.', fg='green')
            return

        try:
            self.processes[name] = subprocess.Popen(
                command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True
            )
        except OSError as exc:
            status_label.config(text=f'Failed to start {name}: {exc}', fg='red')
            return

        status_label.config(text=f'Started {name}.', fg='green')

    def trigger_action(self, client, action_name, status_label) -> None:
        if not client.wait_for_service(timeout_sec=0.5):
            status_label.config(text=f'Error: Controller offline.', fg='red')
            return

        status_label.config(text=f'{action_name}...', fg='blue')
        req = Trigger.Request()
        future = client.call_async(req)
        future.add_done_callback(lambda f: self.service_response_callback(f, status_label))

    @staticmethod
    def service_response_callback(future, status_label) -> None:
        try:
            response = future.result()
            color = 'green' if response.success else 'red'
            status_label.config(text=response.message, fg=color)
        except Exception as exc:
            status_label.config(text=f'Service call failed: {exc}', fg='red')

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
    root.title('TurtleBot Barrel Mission Pro')
    # Increased height to fit new buttons
    root.geometry('360x620')
    root.attributes('-topmost', True)

    status_lbl = tk.Label(root, text='Waiting for input...', font=('Arial', 10), wraplength=330)

    strict_var = tk.BooleanVar(value=False)

    def on_strict_toggle() -> None:
        enabled = bool(strict_var.get())
        node.publish_strict_camera_validation(enabled)
        status_lbl.config(
            text=(
                'Strict camera validation is ON (mission pauses if camera disagrees with LiDAR).'
                if enabled
                else 'Strict camera validation is OFF (camera mismatch is log-only).'
            ),
            fg='#E65100',
        )

    strict_frame = tk.LabelFrame(
        root,
        text=' Safety Switch ',
        font=('Arial', 10, 'bold'),
        fg='#E65100',
        bg='#FFF3E0',
        padx=8,
        pady=6,
    )
    strict_frame.pack(fill='x', padx=10, pady=(4, 8))

    strict_check = tk.Checkbutton(
        strict_frame,
        text='AI Camera Validation', # 
        variable=strict_var,
        command=on_strict_toggle,
        font=('Arial', 11, 'bold'),
        fg='black',
        bg='#FFF3E0',
        activebackground='#FFE0B2',
        activeforeground='black',
        highlightthickness=0,
        selectcolor='#FF9800',
        indicatoron=True,
    )
    strict_check.pack(anchor='w')

    # --- Setup & Detection Frame ---
    btn_controller = tk.Button(root, text='1. Start Mission Controller', font=('Arial', 11, 'bold'), bg='lightgray', fg='black',
                               command=lambda: node.start_mission_controller(status_lbl))
    btn_controller.pack(expand=True, fill='both', padx=10, pady=2)

    btn_detection = tk.Button(root, text='2. Start LiDAR + Map Detection', font=('Arial', 11, 'bold'), bg='deepskyblue', fg='black',
                              command=lambda: node.start_detectors(status_lbl))
    btn_detection.pack(expand=True, fill='both', padx=10, pady=2)

    btn_stop_detection = tk.Button(root, text='Stop Detection', font=('Arial', 10), bg='gray', fg='white',
                                   command=lambda: node.stop_detectors(status_lbl))
    btn_stop_detection.pack(expand=True, fill='both', padx=10, pady=2)

    # --- Mission Execution Frame ---
    btn_calc = tk.Button(root, text='3. Calculate Target Path', font=('Arial', 11, 'bold'), bg='orange', fg='black',
                         command=lambda: node.trigger_action(node.cli_calc, "Calculating Target", status_lbl))
    btn_calc.pack(expand=True, fill='both', padx=10, pady=(15, 2))

    btn_nav = tk.Button(root, text='4. START NAVIGATION', font=('Arial', 12, 'bold'), bg='green', fg='white',
                        command=lambda: node.trigger_action(node.cli_nav, "Starting Navigation", status_lbl))
    btn_nav.pack(expand=True, fill='both', padx=10, pady=2)

    # --- State Machine Controls Frame ---
    control_frame = tk.Frame(root)
    control_frame.pack(fill='x', padx=10, pady=5)
    
    tk.Button(control_frame, text="⏸ PAUSE", font=("Arial", 10, "bold"), bg="gold", fg="black", width=12,
              command=lambda: node.trigger_action(node.cli_pause, "Pausing", status_lbl)).pack(side='left', expand=True)
              
    tk.Button(control_frame, text="▶ RESUME", font=("Arial", 10, "bold"), bg="lightgreen", fg="black", width=12,
              command=lambda: node.trigger_action(node.cli_res, "Resuming", status_lbl)).pack(side='right', expand=True)

    tk.Button(root, text="⏹ STOP & RESET", font=("Arial", 11, "bold"), bg="red", fg="white",
              command=lambda: node.trigger_action(node.cli_stop, "Stopping Mission", status_lbl)).pack(fill='x', padx=10, pady=2)

    status_lbl.pack(pady=10)

    def on_close() -> None:
        node.stop_child_processes()
        node.destroy_node()
        rclpy.shutdown()
        root.destroy()

    root.protocol('WM_DELETE_WINDOW', on_close)

    def sync_strict_after_spin() -> None:
        node.publish_strict_camera_validation(strict_var.get())

    root.after(400, sync_strict_after_spin)
    root.mainloop()


if __name__ == '__main__':
    main()