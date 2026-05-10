from setuptools import find_packages, setup

package_name = 'barrel_challenge'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='ubuntu@todo.todo',
    description='TurtleBot 4 Barrel Discovery and Navigation Framework',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # Format: 'command_name = package_name.script_name:main_function'
            'mission_controller = barrel_challenge.barrel_mission:main',
            'ui_remote = barrel_challenge.ui_button:main'
        ],
    },
)