'''
1) Llega un aviso (x, y)
    buscar_candidata -> Hay una cerca?
        si -> agregar el aviso
        no -> anotar una candidata nueva en el pizarron
    esa candidata ahora esta condirmada, no fue reportada, no es duplicada,
    quedan reportes y ya pasaron los segundos de espera inicial?
        si -> publicar su posicion /victimas y marcarla reportada
'''

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from geometry_msgs.msg import PointStamped
from nav_msgs.msg import OccupancyGrid

from equipo_jar.reglas_gestor import(
    posicion, buscar_candidata, esta_confirmada, es_duplicada
)

QOS_MAPA = QoSProfile(
    depth=1, reliability=ReliabilityPolicy.RELIABLE, 
    durability=DurabilityPolicy.TRANSIENT_LOCAL)

class NodoGestor(Node):
    def __init__(self):
        super().__init__('gestor_victimas')

        self.declare_parameter('cantidad_victimas', 5)
        self.declare_parameter('radio_asociacion', 0.4)
        self.declare_parameter('min_avisos', 5)
        self.declare_parameter('disp_max', 0.10)
        self.declare_parameter('radio_duplicado', 0.6)
        self.declare_parameter('espera_inicial_s', 5.0)

        self.cantidad_victimas = self.get_parameter('cantidad_victimas').value
        self.radio_asociacion = self.get_parameter('radio_asociacion').value
        self.min_avisos = self.get_parameter('min_avisos').value
        self.disp_max = self.get_parameter('disp_max').value
        self.radio_duplicado = self.get_parameter('radio_duplicado').value
        self.espera_inicial_s = self.get_parameter('espera_inicial_s').value

        self.pizarron = []
        self.reportes_hechos = 0
        self.t_mapa = None
        self.mapa_recibido = False

        self.create_subscription(OccupancyGrid, '/map', self.al_recibir_mapa, QOS_MAPA)
        self.create_subscription(
            PointStamped, '/percepcion/observaciones',
            self.al_recibir_observacion, 10)
        self.pub_victimas = self.create_publisher(PointStamped, '/victimas', 10)

    def al_recibir_mapa(self, msg):
        self.mapa_recibido = True
        if self.t_mapa is None:
            self.t_mapa = self.get_clock().now()
            self.get_logger().info(
                f"Hora llegada de mapa: {self.t_mapa}"
            )
    
    def al_recibir_observacion(self, msg):
        if not self.mapa_recibido:
            return
        if self.t_mapa is None:
            ahora = self.get_clock().now()
            if ahora.nanoseconds == 0:
                return
            self.t_mapa = ahora
            self.get_logger().info(f"Arranca la corrida: t={ahora.nanoseconds * 1e-9:.1f} s")
            return
        x, y = msg.point.x, msg.point.y

        i = buscar_candidata(self.pizarron, x, y, self.radio_asociacion)

        if i is None:
            self.pizarron.append({'xs': [x], 'ys': [y], 'reportada': False})
            i = len(self.pizarron) -1
        else:
            self.pizarron[i]['xs'].append(x)
            self.pizarron[i]['ys'].append(y)
            
        candidata = self.pizarron[i]
        if self.hay_que_reportar(candidata):
            self.reportar(candidata)

    def hay_que_reportar(self, candidata: dict) -> bool:
        px, py = posicion(candidata)
        quedan_reportes = self.reportes_hechos < self.cantidad_victimas
        es_dupli = es_duplicada(self.pizarron, px, py, self.radio_duplicado)
        segundos = (self.get_clock().now() - self.t_mapa).nanoseconds * 1e-9
        paso_espera = segundos >= self.espera_inicial_s
        confirmada = esta_confirmada(candidata, self.min_avisos, self.disp_max)
        return not candidata['reportada'] and quedan_reportes and not es_dupli and paso_espera and confirmada

    def reportar(self, candidata: dict):
        px, py = posicion(candidata)
        reporte = PointStamped()
        reporte.header.frame_id = 'map'
        reporte.header.stamp = self.get_clock().now().to_msg()
        reporte.point.x = float(px)
        reporte.point.y = float(py)
        self.pub_victimas.publish(reporte)

        candidata['reportada'] = True
        self.reportes_hechos += 1
        self.get_logger().info(
            f"Victima {self.reportes_hechos}/{self.cantidad_victimas} reportada en: "
            f"({px:.3f}, {py:.3f}) con {len(candidata['xs'])} avisos"
        )
def main(args=None):
    rclpy.init(args=args)
    nodo = NodoGestor()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()