from setuptools import find_packages, setup

package_name = 'scene_trigger_pkg'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='athul-s-nair',
    maintainer_email='athul.sukesh@fau.de',
    description='ROS2 node for automatic interaction triggering in HRI using RGB vision and YOLO',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'scene_trigger_node = scene_trigger_pkg.scene_trigger_node:main',
            'camera_publisher = scene_trigger_pkg.camera_publisher:main',
            'dashboard_node = scene_trigger_pkg.dashboard_node:main',
            'data_collector = scene_trigger_pkg.data_collector:main',
            'generate_graphs = scene_trigger_pkg.generate_graphs:main',
    ],
    },
)
