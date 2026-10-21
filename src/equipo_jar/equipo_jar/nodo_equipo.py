"""Nodo esqueleto del equipo.

No hace nada útil todavía: sirve para que el template compile, arranque, espere
el mapa y pase el CI. Reemplazalo (o agregá nodos) con tu estrategia.

Lo que ya resuelve:
- Se suscribe a /map con el QoS que usa el juez (reliable + transient local).
- No publica nada en /cmd_vel ni en /victimas hasta recibir el mapa (§5).
- Deja armado el canal de /victimas, /done y /cmd_vel.

Lo que tenés que hacer vos, en este orden:
1. Localizarte en el mapa. Arrancás cerca del origen (0, 0), pero no exactamente ahí (§1).
2. Explorar y esquivar obstáculos.
3. Detectar cajas rojas y reportarlas con `armar_reporte`.
4. Opcional: volver a la largada y publicar en /done.
"""
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

        self.t_inicio_s = None #Se le asigna valor al recibir el mapa
        self.t_ultimo_log_s = None #lo actualiza ciclo_de_control
        self.termino = False #lo escribe ciclo_de_control, lo pone en True despues de llamar a terminar()

        # Cantidad de víctimas a buscar (N), que llega como argumento del launch (victimas:=N).
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
            return  # esperar el mapa: no mover el robot ni reportar antes

        # TODO: acá va tu estrategia. Por ahora el robot se queda quieto.
        self.pub_cmd_vel.publish(Twist())

        # Si ya publicamos /done con self.terminar() no hacemos nada
        if self.termino:
                return
        
        # Hora actual en segundos
        ahora = self.get_clock().now().nanoseconds / 1e9
        
        # Calcular tiempo restante en s
        tiempo_restante_s = tiempo_restante(self.t_inicio_s, ahora, self.duracion_corrida_s)
        
        # TODO: distancia real a la largada (parte 3)
        reserva_s = calcular_reserva(
            0.0, self.velocidad_regreso_ms, self.factor_cambio, self.margen_regreso_s)
         
        volver = hay_que_volver(
            tiempo_restante_s, 
            reserva_s,
            reportes_hechos=0,   #TODO: lo va a dar el gestor de victimas 
            n_victimas=self.cantidad_victimas, 
            quedan_viewpoints=True, # TODO: lo va a dar la exploracion
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
        """Publica un reporte. Cada llamada gasta un reporte (§7.2)."""
        self.pub_victimas.publish(armar_reporte(x, y, self.get_clock().now().to_msg()))

    def terminar(self):
        """Publica en /done: termina la corrida (§5)."""
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
