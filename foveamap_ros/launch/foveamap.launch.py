"""ROS 2 launch script for FoveaMap (Phase 8)."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    """Generate launch description for FoveaMap ROS 2 node."""
    return LaunchDescription([
        DeclareLaunchArgument(
            "input_topic",
            default_value="/lidar/points",
            description="Input sensor_msgs/PointCloud2 topic name",
        ),
        DeclareLaunchArgument(
            "grid_topic",
            default_value="/foveamap/grid",
            description="Output occupancy/elevation grid topic name",
        ),
        DeclareLaunchArgument(
            "base_frame",
            default_value="base_link",
            description="Vehicle base coordinate frame ID",
        ),
        DeclareLaunchArgument(
            "world_frame",
            default_value="odom",
            description="Odometry or map coordinate frame ID",
        ),
        DeclareLaunchArgument(
            "device",
            default_value="auto",
            description="Computing device: auto, cuda, cpu",
        ),
        Node(
            package="foveamap_ros",
            executable="foveamap-ros",
            name="foveamap_node",
            output="screen",
            parameters=[{
                "io.input_topic": LaunchConfiguration("input_topic"),
                "io.grid_topic": LaunchConfiguration("grid_topic"),
                "io.base_frame": LaunchConfiguration("base_frame"),
                "io.world_frame": LaunchConfiguration("world_frame"),
                "runtime.device": LaunchConfiguration("device"),
            }],
        ),
    ])
