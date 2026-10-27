"""Launch único del equipo (reglamento §9.1).

Este es el archivo que la organización lanza en la corrida:

    ros2 launch equipo_jar competencia.launch.py

Tiene que levantar TODO el software del equipo y quedarse esperando a que el
juez publique el mapa en /map. No mueve al robot ni publica reportes antes.

`victimas` es la cantidad de víctimas a buscar (N). La organización la informa en el lugar, antes de
cada corrida, y se pasa así:  ros2 launch equipo_jar competencia.launch.py victimas:=N

`use_sim_time` es false por defecto porque el robot real no publica /clock
(§7.3). En el simulador se pasa en true (lo hacen tools/correr_simulador.sh y
el CI).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml


def generate_launch_description():
    share = get_package_share_directory('equipo_jar')
    params = os.path.join(share, 'params', 'params.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    victimas = LaunchConfiguration('victimas')

    nav2_params = RewrittenYaml(
        source_file=os.path.join(share, 'params', 'nav2_params.yaml'),
        root_key='',
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='cantidad de víctimas a buscar (N)'),

        Node(
            package='equipo_jar',
            executable='nodo_equipo',
            name='nodo_equipo',
            output='screen',
            parameters=[params, {
                'use_sim_time': use_sim_time,
                'cantidad_victimas': ParameterValue(victimas, value_type=int),
            }],
        ),

        Node(
            package='nav2_amcl',
            executable='amcl',
            name='amcl',
            output='screen',
            parameters=[nav2_params],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_localizacion',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time,
                         'autostart': True,
                         'bond_timeout': 0.0,
                         'node_names': ['amcl']}],
        ),

        Node(
            package='nav2_controller',
            executable='controller_server',
            name='controller_server',
            output='screen',
            parameters=[nav2_params],
        ),
        Node(
            package='nav2_planner',
            executable='planner_server',
            name='planner_server',
            output='screen',
            parameters=[nav2_params],
        ),
        Node(
            package='nav2_behaviors',
            executable='behavior_server',
            name='behavior_server',
            output='screen',
            parameters=[nav2_params],
        ),
        Node(
            package='nav2_bt_navigator',
            executable='bt_navigator',
            name='bt_navigator',
            output='screen',
            parameters=[nav2_params],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navegacion',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time,
                         'autostart': True,
                         'bond_timeout': 0.0,
                         'node_names': ['controller_server', 'planner_server',
                                        'behavior_server', 'bt_navigator']}],
        ),
    ])