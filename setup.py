from glob import glob

from setuptools import find_packages, setup

package_name = 'barrel_lidar_detector'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml', 'barrel_target.example.yaml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'PyYAML', 'numpy', 'opencv-python'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='Map-based ground-truth barrel detector for TurtleBot 4.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'auto_explorer = barrel_lidar_detector.auto_explorer:main',
            'ground_truth_barrel_detector = barrel_lidar_detector.ground_truth_barrel_detector:main',
            'map_debug_monitor = barrel_lidar_detector.map_debug_monitor:main',
            'mission_controller = barrel_lidar_detector.mission_controller:main',
            'offline_barrel_marker = barrel_lidar_detector.offline_barrel_marker:main',
            'create3_music = barrel_lidar_detector.create3_music:main',
            'ui_remote = barrel_lidar_detector.ui_button:main',
        ],
    },
)
