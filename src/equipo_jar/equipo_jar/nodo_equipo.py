# nodo viejo del cerebro con bt, no se lanza (ahora manda exploracion)
import rclpy
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Empty

from equipo_jar.reportes import armar_reporte

from equipo_jar.reglas_cerebro import (
    tiempo_restante, calcular_reserva, hay_que_volver, toca_loguear)

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)


class NodoEquipo(Node):
    def __init__(self):
        super().__init__('nodo_equipo')
        self.declare_parameter('frecuencia_control_hz', 10.0)
        self.frecuencia_control_hz = self.get_parameter('frecuencia_control_hz').value

        self.declare_parameter('velocidad_lineal_max', 0.30)
        self.velocidad_lineal_max = self.get_parameter('velocidad_lineal_max').value

        self.declare_parameter('velocidad_angular_max', 1.00)
        self.velocidad_angular_max = self.get_parameter('velocidad_angular_max').value

        self.declare_parameter('velocidad_regreso_ms', 0.25)
        self.velocidad_regreso_ms = self.get_parameter('velocidad_regreso_ms').value

        self.declare_parameter('factor_cambio', 1.5)
        self.factor_cambio = self.get_parameter('factor_cambio').value

        self.declare_parameter('margen_regreso_s', 20.0)
        self.margen_regreso_s = self.get_parameter('margen_regreso_s').value

        self.declare_parameter('duracion_corrida_s', 600.0)
        self.duracion_corrida_s = self.get_parameter('duracion_corrida_s').value

        self.declare_parameter('periodo_logs_s', 10.0)
        self.periodo_logs_s = self.get_parameter('periodo_logs_s').value

        self.t_inicio_s = None
        self.t_ultimo_log_s = None
        self.termino = False

        self.declare_parameter('cantidad_victimas', 5)
        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value

        self.mapa = None

        self.sub_mapa = self.create_subscription(
            OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)

        periodo = 1.0 / self.get_parameter('frecuencia_control_hz').value
        self.create_timer(periodo, self.ciclo_de_control)
        self.get_logger().info(
            f'Hay que encontrar {self.cantidad_victimas} víctimas. Esperando el mapa en /map...')

    def al_recibir_mapa(self, mapa):
        if self.mapa is None:
            self.get_logger().info(
                f'Mapa recibido: {mapa.info.width}x{mapa.info.height} celdas, '
                f'resolución {mapa.info.resolution:.3f} m')
        self.mapa = mapa

        if self.t_inicio_s is None:
            self.t_inicio_s = (self.get_clock().now()).nanoseconds / 1e9
            self.get_logger().info(f"Cronometro iniciado en t={self.t_inicio_s}")

    def ciclo_de_control(self):
        if self.mapa is None:
            return


        if self.termino:
                return
        
        ahora = self.get_clock().now().nanoseconds / 1e9
        
        tiempo_restante_s = tiempo_restante(self.t_inicio_s, ahora, self.duracion_corrida_s)
        
        reserva_s = calcular_reserva(
            0.0, self.velocidad_regreso_ms, self.factor_cambio, self.margen_regreso_s)
         
        volver = hay_que_volver(
            tiempo_restante_s, 
            reserva_s,
            reportes_hechos=0,
            n_victimas=self.cantidad_victimas, 
            quedan_viewpoints=True,
            )
        
        if volver:
            self.terminar()
            self.termino = True
            self.get_logger().info(
                f"Vuelvo y termino: restante: {tiempo_restante_s:.1f} s, "
                f"reserva: {reserva_s:.1f} s")
            return
            
        if toca_loguear(self.t_ultimo_log_s, ahora, self.periodo_logs_s):
            self.t_ultimo_log_s = ahora
            self.get_logger().info(
                f"tiempo restante: {tiempo_restante_s:.1f} s, "
                f"reserva: {reserva_s:.1f} s")
        

    def reportar_victima(self, x, y):
        self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))

    def terminar(self):
        self.pub_done.publish(Empty())


def main(args=None):
    rclpy.init(args=args)
    nodo = NodoEquipo()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
