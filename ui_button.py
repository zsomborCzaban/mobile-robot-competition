import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
import tkinter as tk
import threading

class MissionUIButton(Node):
    def __init__(self):
        super().__init__('mission_ui_button')
        # Two clients for our two new services
        self.cli_calc = self.create_client(Trigger, 'calculate_target')
        self.cli_nav = self.create_client(Trigger, 'start_navigation')

    def trigger_calc(self, status_label):
        if not self.cli_calc.wait_for_service(timeout_sec=1.0):
            status_label.config(text="Error: Controller not running!", fg="red")
            return
            
        status_label.config(text="Calculating...", fg="blue")
        req = Trigger.Request()
        future = self.cli_calc.call_async(req)
        future.add_done_callback(lambda f: self.calc_response_callback(f, status_label))

    def trigger_nav(self, status_label):
        if not self.cli_nav.wait_for_service(timeout_sec=1.0):
            status_label.config(text="Error: Controller not running!", fg="red")
            return

        status_label.config(text="Starting Nav...", fg="blue")
        req = Trigger.Request()
        future = self.cli_nav.call_async(req)
        future.add_done_callback(lambda f: self.nav_response_callback(f, status_label))

    def calc_response_callback(self, future, status_label):
        response = future.result()
        color = "green" if response.success else "red"
        # Update UI text with the message from the Python controller
        status_label.config(text=response.message, fg=color)

    def nav_response_callback(self, future, status_label):
        response = future.result()
        color = "green" if response.success else "red"
        status_label.config(text=response.message, fg=color)

def run_ros_spin(node):
    rclpy.spin(node)

def main():
    rclpy.init()
    node = MissionUIButton()

    ros_thread = threading.Thread(target=run_ros_spin, args=(node,), daemon=True)
    ros_thread.start()

    # Setup the Tkinter GUI
    root = tk.Tk()
    root.title("TurtleBot Controller")
    root.geometry("300x200")
    root.attributes('-topmost', True) 

    # --- UI Layout ---
    btn_calc = tk.Button(
        root, text="1. Calculate Target", font=("Arial", 12, "bold"), 
        bg="orange", fg="black",
        command=lambda: node.trigger_calc(status_lbl)
    )
    btn_calc.pack(expand=True, fill='both', padx=10, pady=5)

    btn_nav = tk.Button(
        root, text="2. START NAVIGATION", font=("Arial", 12, "bold"), 
        bg="green", fg="white",
        command=lambda: node.trigger_nav(status_lbl)
    )
    btn_nav.pack(expand=True, fill='both', padx=10, pady=5)

    status_lbl = tk.Label(root, text="Waiting for input...", font=("Arial", 10), wraplength=280)
    status_lbl.pack(pady=10)

    root.mainloop()
    rclpy.shutdown()

if __name__ == '__main__':
    main()