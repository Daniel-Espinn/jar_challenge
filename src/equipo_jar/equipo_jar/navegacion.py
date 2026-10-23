"""Nodo de navegación del equipo (esqueleto).

Responsabilidad: mover el robot por el laberinto sin chocar y, al final,
volver a la largada y publicar /done. NO calcula la pose (la da el localizador
vía TF map -> odom) y NO detecta víctimas (eso es percepción).

Estados:
    ESPERANDO   sin /map o sin TF map->base válida: quieto.
    ASENTANDO   con TF válida: espera unos segundos y guarda la pose de largada.
    EXPLORANDO  recorre el laberinto (hoy: reactivo; TODO: cobertura + A*).
    REGRESANDO  vuelve a la pose de largada guardada.
    TERMINADO   publicó /done (o no hay nada más que hacer): quieto.

Interfaces (todas oficiales, §7.1):
    sub  /map        OccupancyGrid  (reliable + transient local)
    sub  /scan       LaserScan      (best effort -> qos_profile_sensor_data)
    sub  /victimas   PointStamped   (solo para contar cuántos reportes hubo)
    pub  /cmd_vel    Twist          (canal exclusivo de velocidad)
    pub  /done       Empty
    TF   map -> base_footprint      (la publica el localizador)
"""
import math

import numpy as np
import rclpy
import tf2_ros
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, QoSProfile, ReliabilityPolicy,
                       qos_profile_sensor_data)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from tf2_ros import TransformException

QOS_MAPA = QoSProfile(
    depth=1,
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
)

ESPERANDO = 'esperando'
ASENTANDO = 'asentando'
EXPLORANDO = 'explorando'
REGRESANDO = 'regresando'
TERMINADO = 'terminado'


class Navegacion(Node):
    def __init__(self):
        super().__init__('navegacion')
        # Todos estos valores se pisan desde params/params.yaml (sección `navegacion`).
        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('frecuencia_control_hz', 10.0)
        self.declare_parameter('velocidad_maxima', 0.25)
        self.declare_parameter('suavizado', 0.3)
        self.declare_parameter('radio_repulsion', 0.40)
        self.declare_parameter('ganancia_repulsiva', 0.15)
        self.declare_parameter('distancia_minima_valida', 0.25)
        self.declare_parameter('tiempo_asentamiento_s', 3.0)
        self.declare_parameter('tiempo_regreso_s', 480.0)    # 8:00 desde el /map
        self.declare_parameter('tolerancia_llegada', 0.15)   # < 0.25 m del bono
        self.declare_parameter('cantidad_victimas', 5)       # lo setea el launch

        self.map_frame = self.get_parameter('map_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.n_victimas = self.get_parameter('cantidad_victimas').value

        self.estado = ESPERANDO
        self.mapa = None
        self.scan = None
        self.t_mapa = None          # instante en que llegó /map (inicio del cronómetro)
        self.t_tf_ok = None         # primer instante con TF válida
        self.pose_largada = None    # (x, y) en map
        self.reportes = 0
        self.cmd_anterior = np.zeros(3)  # vx, vy, wz

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(OccupancyGrid, '/map', self.recibir_mapa, QOS_MAPA)
        self.create_subscription(LaserScan, '/scan', self.recibir_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(PointStamped, '/victimas', self.contar_reporte, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)

        periodo = 1.0 / self.get_parameter('frecuencia_control_hz').value
        self.create_timer(periodo, self.ciclo)
        self.get_logger().info('Navegación lista. Esperando /map...')

    # ---------------------------------------------------------------- callbacks
    def recibir_mapa(self, msg):
        if self.mapa is None:
            self.t_mapa = self.get_clock().now()
            self.get_logger().info(
                f'Mapa recibido ({msg.info.width}x{msg.info.height}, '
                f'{msg.info.resolution:.3f} m/celda). Arranca el cronómetro.')
        self.mapa = msg

    def recibir_scan(self, msg):
        self.scan = msg  # solo se guarda; la lógica corre en el timer

    def contar_reporte(self, _msg):
        self.reportes += 1

    # ------------------------------------------------------------------ helpers
    def segundos_desde_mapa(self):
        if self.t_mapa is None:
            return 0.0
        return (self.get_clock().now() - self.t_mapa).nanoseconds * 1e-9

    def pose_en_mapa(self):
        """(x, y, yaw) de base_footprint en map, o None si TF todavía no existe."""
        try:
            t = self.tf_buffer.lookup_transform(
                self.map_frame, self.base_frame, rclpy.time.Time())
        except TransformException:
            return None
        q = t.transform.rotation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                         1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        return t.transform.translation.x, t.transform.translation.y, yaw

    def cambiar_estado(self, nuevo):
        if nuevo != self.estado:
            self.get_logger().info(f'Estado: {self.estado} -> {nuevo}')
            self.estado = nuevo

    def rayos_validos(self):
        """(distancias, ángulos en el frame del lidar) sin inf/NaN ni rebotes propios."""
        s = self.scan
        r = np.asarray(s.ranges, dtype=float)
        a = s.angle_min + np.arange(len(r)) * s.angle_increment
        ok = np.isfinite(r) & (r > self.get_parameter('distancia_minima_valida').value)
        return r[ok], a[ok]

    # ------------------------------------------------------ capas de movimiento
    def vector_repulsivo(self):
        """Empuja al robot lejos de lo que el /scan ve cerca (incluye señuelos).

        Se asume el lidar alineado con base_footprint; si no, hay que rotar por
        la TF del frame del lidar.
        """
        if self.scan is None:
            return np.zeros(2)
        r, a = self.rayos_validos()
        radio = self.get_parameter('radio_repulsion').value
        cerca = r < radio
        if not np.any(cerca):
            return np.zeros(2)
        r, a = r[cerca], a[cerca]
        peso = (1.0 / r - 1.0 / radio)
        vx = -np.sum(peso * np.cos(a))
        vy = -np.sum(peso * np.sin(a))
        return self.get_parameter('ganancia_repulsiva').value * np.array([vx, vy]) / len(r)

    def direccion_mas_libre(self):
        """Vector unitario (en el robot) hacia el sector con más espacio libre.

        Es un explorador reactivo mínimo, para poder probar el robot en
        movimiento. TODO: reemplazarlo por cobertura + A* con costmap inflado.
        """
        if self.scan is None:
            return np.zeros(2)
        r, a = self.rayos_validos()
        if len(r) == 0:
            return np.zeros(2)
        r = np.minimum(r, 3.0)
        # Suavizar: promediar ventanas de rayos para no apuntar a una rendija.
        k = max(1, len(r) // 36)
        suave = np.convolve(r, np.ones(k) / k, mode='same')
        mejor = a[int(np.argmax(suave))]
        return np.array([math.cos(mejor), math.sin(mejor)])

    def vector_hacia(self, destino, pose):
        """Vector unitario hacia (x, y) de map, expresado en el frame del robot.

        Mecanum: se manda directo a linear.x / linear.y sin girar el chasis.
        """
        x, y, yaw = pose
        dx, dy = destino[0] - x, destino[1] - y
        d = math.hypot(dx, dy)
        if d < 1e-6:
            return np.zeros(2), 0.0
        # rotar de map a robot
        rx = math.cos(-yaw) * dx - math.sin(-yaw) * dy
        ry = math.sin(-yaw) * dx + math.cos(-yaw) * dy
        return np.array([rx, ry]) / d, d

    def publicar_velocidad(self, v_xy, wz=0.0):
        vmax = self.get_parameter('velocidad_maxima').value
        n = np.linalg.norm(v_xy)
        if n > 1.0:
            v_xy = v_xy / n
        nuevo = np.array([v_xy[0] * vmax, v_xy[1] * vmax, wz])
        s = self.get_parameter('suavizado').value
        cmd = s * nuevo + (1.0 - s) * self.cmd_anterior
        self.cmd_anterior = cmd
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.angular.z = float(cmd[0]), float(cmd[1]), float(cmd[2])
        self.pub_cmd.publish(msg)

    def parar(self):
        self.cmd_anterior = np.zeros(3)
        self.pub_cmd.publish(Twist())

    # -------------------------------------------------------------- ciclo principal
    def ciclo(self):
        # 1) Nada se mueve antes del mapa (§5).
        if self.mapa is None:
            return

        pose = self.pose_en_mapa()
        if pose is None:
            self.t_tf_ok = None
            self.cambiar_estado(ESPERANDO)
            return

        self.transicionar(pose)

        if self.estado in (ESPERANDO, ASENTANDO, TERMINADO):
            self.parar()
        elif self.estado == EXPLORANDO:
            # TODO: objetivo = celda no cubierta más cercana; camino = A*; seguir camino.
            v = self.direccion_mas_libre() * 0.6 + self.vector_repulsivo()
            self.publicar_velocidad(v)
        elif self.estado == REGRESANDO:
            # TODO: A* hacia la largada en vez de línea recta.
            u, _ = self.vector_hacia(self.pose_largada, pose)
            self.publicar_velocidad(u * 0.6 + self.vector_repulsivo())

    def transicionar(self, pose):
        """Máquina de estados de misión: separada de la acción de cada estado."""
        ahora = self.get_clock().now()

        if self.estado == ESPERANDO:
            self.t_tf_ok = ahora
            self.cambiar_estado(ASENTANDO)

        elif self.estado == ASENTANDO:
            espera = self.get_parameter('tiempo_asentamiento_s').value
            if (ahora - self.t_tf_ok).nanoseconds * 1e-9 >= espera:
                self.pose_largada = (pose[0], pose[1])
                self.get_logger().info(
                    f'Pose de largada guardada: ({pose[0]:.2f}, {pose[1]:.2f})')
                self.cambiar_estado(EXPLORANDO)

        elif self.estado == EXPLORANDO:
            todas = self.reportes >= self.n_victimas
            se_acaba = self.segundos_desde_mapa() >= self.get_parameter('tiempo_regreso_s').value
            if todas or se_acaba:
                self.cambiar_estado(REGRESANDO)

        elif self.estado == REGRESANDO:
            _, d = self.vector_hacia(self.pose_largada, pose)
            if d <= self.get_parameter('tolerancia_llegada').value:
                # El bono exige >= 1 víctima puntuada; el reglamento no deja
                # saber desde acá cuál puntuó, así que se confía en los reportes.
                self.parar()
                self.pub_done.publish(Empty())
                self.get_logger().info('Llegué a la largada: /done publicado.')
                self.cambiar_estado(TERMINADO)


def main(args=None):
    rclpy.init(args=args)
    nodo = Navegacion()
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