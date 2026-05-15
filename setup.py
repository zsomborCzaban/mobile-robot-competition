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
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='LiDAR and map-shape based barrel candidate detector for TurtleBot 4.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'auto_explorer = barrel_lidar_detector.auto_explorer:main',
            'lidar_cluster_detector = barrel_lidar_detector.lidar_cluster_detector:main',
            'map_shape_detector = barrel_lidar_detector.map_shape_detector:main',
            'mission_controller = barrel_lidar_detector.mission_controller:main',
            'ui_remote = barrel_lidar_detector.ui_button:main',
        ],
    },
)
