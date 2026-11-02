# launch unico: levanta todo y espera /map. victimas:=N
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from nav2_common.launch import RewrittenYaml

from launch.conditions import IfCondition
from launch.substitutions import PythonExpression

def generate_launch_description():
    share = get_package_share_directory('equipo_jar')
    params = os.path.join(share, 'params', 'params.yaml')
    use_sim_time = LaunchConfiguration('use_sim_time')
    victimas = LaunchConfiguration('victimas')

    localizador = LaunchConfiguration('localizador')
    es_propio = IfCondition(PythonExpression(["'", localizador, "' == 'propio'"]))
    es_amcl = IfCondition(PythonExpression(["'", localizador, "' == 'amcl'"]))

    nav2_params = RewrittenYaml(
        source_file=os.path.join(share, 'params', 'nav2_params.yaml'),
        root_key='',
        param_rewrites={'use_sim_time': use_sim_time},
        convert_types=True)

    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='false'),
        DeclareLaunchArgument('victimas', default_value='5',
                              description='cantidad de víctimas a buscar (N)'),
        DeclareLaunchArgument('localizador', default_value='propio',
                              description='propio | amcl'),

        Node(
            package='equipo_jar',
            executable='gestor_victimas',
            name='gestor_victimas',
            output='screen',
            parameters=[params, {
                'use_sim_time': use_sim_time,
                'cantidad_victimas': ParameterValue(victimas, value_type=int),
            }],
        ),

        Node(
            package='equipo_jar',
            executable='percepcion',
            name='percepcion',
            output='screen',
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),

        Node(
            package='nav2_amcl',
            executable='amcl',
            condition=es_amcl,
            name='amcl',
            output='screen',
            parameters=[nav2_params],
        ),
        Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            condition=es_amcl,
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
        TimerAction(period=5.0, actions=[Node(
            package='nav2_lifecycle_manager',
            executable='lifecycle_manager',
            name='lifecycle_manager_navegacion',
            output='screen',
            parameters=[{'use_sim_time': use_sim_time,
                         'autostart': True,
                         'bond_timeout': 0.0,
                         'node_names': ['controller_server', 'planner_server',
                                        'behavior_server', 'bt_navigator']}],
        )]),
        Node(
            package='equipo_jar', executable='localizador', name='localizador',
            output='screen', condition=es_propio,
            parameters=[params, {'use_sim_time': use_sim_time}],
        ),
        Node(
            package='equipo_jar', executable='exploracion', name='exploracion',
            output='screen',
            parameters=[params, {
                'use_sim_time': use_sim_time,
                'cantidad_victimas': ParameterValue(victimas, value_type=int),
            }],
        ),
    ])