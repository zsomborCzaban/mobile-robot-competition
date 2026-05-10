import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger
from nav2_simple_commander.robot_navigator import BasicNavigator
from geometry_msgs.msg import PoseStamped
import yaml
import os

class BarrelMissionController(Node):
    def __init__(self):
        super().__init__('barrel_mission_controller')
        
        # 1. Create TWO Service Servers
        self.srv_calc = self.create_service(Trigger, 'calculate_target', self.calc_callback)
        self.srv_nav = self.create_service(Trigger, 'start_navigation', self.nav_callback)
        
        # 2. Initialize Nav2 Commander
        self.navigator = BasicNavigator()
        
        # 3. State variable to hold the goal
        self.target_pose = None
        
        self.get_logger().info("Mission Controller ready. Waiting for calculation trigger...")

    def calc_callback(self, request, response):
        """Reads the YAML and calculates the target pose, but DOES NOT move."""
        self.get_logger().info("Calculating target...")
        
        # Absolute path to your workspace (Adjust if your path is different!)
        yaml_path = os.path.expanduser('~/turtlebot4_ws/barrel_target.yaml')
        
        try:
            with open(yaml_path, 'r') as file:
                data = yaml.safe_load(file)
                barrel_x = data['barrel']['map_x']
                barrel_y = data['barrel']['map_y']
        except Exception as e:
            response.success = False
            response.message = f"Failed to read yaml: {e}"
            self.get_logger().error(response.message)
            return response

        # Calculate Offset (0.6 meters in front of the barrel)
        goal_x = barrel_x - 0.6
        goal_y = barrel_y

        # Store the goal in the class variable
        self.target_pose = PoseStamped()
        self.target_pose.header.frame_id = 'map'
        self.target_pose.header.stamp = self.navigator.get_clock().now().to_msg()
        self.target_pose.pose.position.x = float(goal_x)
        self.target_pose.pose.position.y = float(goal_y)
        self.target_pose.pose.orientation.w = 1.0

        response.success = True
        response.message = f"Calculated goal: X={goal_x:.2f}, Y={goal_y:.2f}"
        self.get_logger().info(response.message)
        return response

    def nav_callback(self, request, response):
        """Sends the stored target pose to Nav2."""
        if self.target_pose is None:
            response.success = False
            response.message = "Cannot start! You must calculate the target first."
            self.get_logger().warn(response.message)
            return response

        self.get_logger().info("Starting navigation to calculated target!")
        self.navigator.waitUntilNav2Active()
        
        # goToPose is non-blocking, it sends the command and continues
        self.navigator.goToPose(self.target_pose)
        
        response.success = True
        response.message = "Navigation command sent to Nav2!"
        return response

def main(args=None):
    rclpy.init(args=args)
    node = BarrelMissionController()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()