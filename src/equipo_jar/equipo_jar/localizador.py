"""Localizador por filtro de partículas (MCL) del equipo.

Publica la transformada map -> odom corrigiendo la odometría con el /scan
contra el /map del juez. Junta en un solo nodo lo que en los workshops eran
campo_verosimilitud.py y localizador.py: el campo de verosimilitud se calcula
acá adentro apenas llega /map, sin topic intermedio.

Interfaces (todas oficiales, §7.1):
    sub  /map     OccupancyGrid  (reliable + transient local)
    sub  /scan    LaserScan      (best effort -> qos_profile_sensor_data)
    sub  /odom    Odometry
    TF   lee lidar -> base_footprint, publica map -> odom

Diferencias con los workshops (a propósito):
    - Sin /ground_truth/odom (no existe en el robot real).
    - Sin /likelihood_map ni /initialpose (topics que no son del reglamento).
    - Sin caminos de debug. La nube de partículas solo se publica si el
      parámetro `publicar_nube` es true (apagado por defecto).
    - escala_avance / escala_giro: corrigen el error de escala de la
      odometría de las ruedas (workshop 10, paso 3) dentro del modelo de
      movimiento, sin tocar /odom ni odom -> base_footprint.
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import Pose, PoseArray, Quaternion, TransformStamped
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time
from scipy import ndimage
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformBroadcaster, TransformListener

UMBRAL_OCUPADO = 65


def normalizar_angulo(angulo: float) -> float:
    return math.atan2(math.sin(angulo), math.cos(angulo))


def yaw_a_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.z = math.sin(yaw * 0.5)
    return q


def invertir_pose(pose):
    x, y, theta = pose
    c, s = math.cos(-theta), math.sin(-theta)
    return (-(c * x - s * y), -(s * x + c * y), -theta)


def componer_poses(pose_a, pose_b):
    xa, ya, ta = pose_a
    xb, yb, tb = pose_b
    c, s = math.cos(ta), math.sin(ta)
    return (xa + c * xb - s * yb, ya + s * xb + c * yb, normalizar_angulo(ta + tb))


class Localizador(Node):

    def __init__(self):
        super().__init__('localizador')

        self.declare_parameter('map_frame', 'map')
        self.declare_parameter('odom_frame', 'odom')
        self.declare_parameter('base_frame', 'base_footprint')
        self.declare_parameter('num_particulas', 500)
        self.declare_parameter('pose_inicial_x', 0.0)
        self.declare_parameter('pose_inicial_y', 0.0)
        self.declare_parameter('pose_inicial_theta', 0.0)
        # La largada es "cerca del origen, sin ser exacta" (§1): la nube
        # arranca ancha para cubrir ese error.

        # Medir el error
        self.declare_parameter('ruido_minimo_xy', 0.004)     # m por paso de odom
        self.declare_parameter('ruido_minimo_theta', 0.004)  # rad por paso de odom
        self.declare_parameter('actualizar_min_d', 0.03)     # m
        self.declare_parameter('actualizar_min_a', 0.05)     # rad

        self.declare_parameter('dispersion_inicial_xy', 0.35)
        self.declare_parameter('dispersion_inicial_theta', 0.5)
        # Varianzas (no desvíos) del modelo de movimiento mecanum.
        for i in range(1, 7):
            self.declare_parameter(f'gamma{i}', 0.0025)
        self.declare_parameter('submuestreo_scan', 15)
        self.declare_parameter('sigma_sensor', 0.2)
        # Escalas medidas en el workshop 10 (paso 3). 1.0 = sin corrección.
        self.declare_parameter('escala_avance', 1.0)
        self.declare_parameter('escala_giro', 1.0)
        self.declare_parameter('frecuencia_tf_hz', 20.0)
        self.declare_parameter('publicar_nube', False)

        self.map_frame = self.get_parameter('map_frame').value
        self.odom_frame = self.get_parameter('odom_frame').value
        self.base_frame = self.get_parameter('base_frame').value
        self.num_particulas = self.get_parameter('num_particulas').value
        self.gamma1, self.gamma2, self.gamma3, self.gamma4, self.gamma5, self.gamma6 = (
            self.get_parameter(f'gamma{i}').value for i in range(1, 7))
        self.submuestreo_scan = self.get_parameter('submuestreo_scan').value
        self.sigma_sensor = self.get_parameter('sigma_sensor').value
        self.escala_avance = self.get_parameter('escala_avance').value
        self.escala_giro = self.get_parameter('escala_giro').value

        self.ruido_min_xy = self.get_parameter('ruido_minimo_xy').value
        self.ruido_min_theta = self.get_parameter('ruido_minimo_theta').value
        self.actualizar_min_d = self.get_parameter('actualizar_min_d').value
        self.actualizar_min_a = self.get_parameter('actualizar_min_a').value
        self.odom_ult_correccion = None

        n = self.num_particulas
        dxy = self.get_parameter('dispersion_inicial_xy').value
        dth = self.get_parameter('dispersion_inicial_theta').value
        x0 = np.random.normal(self.get_parameter('pose_inicial_x').value, dxy, n)
        y0 = np.random.normal(self.get_parameter('pose_inicial_y').value, dxy, n)
        t0 = np.random.normal(self.get_parameter('pose_inicial_theta').value, dth, n)
        self.particulas = np.stack([x0, y0, t0, np.full(n, 1.0 / n)], axis=1)

        self.campo = None
        self.info_mapa = None
        self.ultimo_odom = None
        self.pose_map_odom = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        qos_mapa = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.create_subscription(OccupancyGrid, '/map', self.recibir_mapa, qos_mapa)
        self.create_subscription(LaserScan, '/scan', self.recibir_scan,
                                 qos_profile_sensor_data)
        self.create_subscription(Odometry, '/odom', self.recibir_odom, 10)

        self.pub_particulas = None
        if self.get_parameter('publicar_nube').value:
            self.pub_particulas = self.create_publisher(PoseArray, '/particlecloud', 10)

        self.create_timer(1.0 / self.get_parameter('frecuencia_tf_hz').value,
                          self.publicar_tf_periodico)
        self.get_logger().info(
            f'Localizador listo ({n} partículas, escala_avance={self.escala_avance:.3f}, '
            f'escala_giro={self.escala_giro:.3f}). Esperando /map...')

    # ------------------------------------------------------------ callbacks
    def recibir_mapa(self, msg: OccupancyGrid):
        alto, ancho = msg.info.height, msg.info.width
        grid = np.array(msg.data, dtype=np.int16).reshape((alto, ancho))
        self.campo = self.campo_de_probabilidad(grid, msg.info.resolution)
        self.info_mapa = msg.info
        self.get_logger().info(
            f'Mapa recibido ({ancho}x{alto}, {msg.info.resolution:.3f} m/celda): '
            f'campo de verosimilitud listo.')

    def recibir_odom(self, msg: Odometry):
        try:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            theta = self.yaw_de_quaternion(msg.pose.pose.orientation)

            if self.ultimo_odom is None:
                # Primera odometría: publicar ya map -> odom con la pose inicial,
                # así Nav2 tiene TF antes del primer scan.
                self.ultimo_odom = (x, y, theta)
                self.odom_ult_correccion = (x, y, theta) 
                self.actualizar_map_odom(self.estimar_pose(self.particulas))
                return

            x0, y0, theta0 = self.ultimo_odom
            dx, dy = x - x0, y - y0
            dtheta = normalizar_angulo(theta - theta0)
            self.particulas = self.mover_particulas(self.particulas, dx, dy, dtheta, theta0)
            self.ultimo_odom = (x, y, theta)
            self.publicar_nube()
        except Exception as e:  # un fallo acá no puede tirar el nodo (§11.4)
            self.get_logger().error(f'recibir_odom: {e}', throttle_duration_sec=2.0)

    def recibir_scan(self, msg: LaserScan):
        if self.campo is None or self.ultimo_odom is None:
            return
        try:
            ox, oy, ot = self.odom_ult_correccion
            ux, uy, ut = self.ultimo_odom
            if (math.hypot(ux - ox, uy - oy) < self.actualizar_min_d
                    and abs(normalizar_angulo(ut - ot)) < self.actualizar_min_a):
                return   # robot quieto: no corregir ni remuestrear (evita colapsar la nube)
            puntos = self.scan_a_puntos(msg)
            if puntos is None or len(puntos) == 0:
                return
            pesos = self.pesar_particulas(self.particulas, puntos)
            self.particulas = self.remuestrear(self.particulas, pesos)
            self.actualizar_map_odom(self.estimar_pose(self.particulas))
            self.publicar_nube()
            self.odom_ult_correccion = self.ultimo_odom
        except Exception as e:
            self.get_logger().error(f'recibir_scan: {e}', throttle_duration_sec=2.0)

    # --------------------------------------------------------------- helpers
    def yaw_de_quaternion(self, q: Quaternion) -> float:
        return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def scan_a_puntos(self, msg: LaserScan):
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, msg.header.frame_id, Time())
        except Exception:
            return None

        rangos = np.array(msg.ranges)
        angulos = msg.angle_min + np.arange(len(rangos)) * msg.angle_increment
        rangos = rangos[:: self.submuestreo_scan]
        angulos = angulos[:: self.submuestreo_scan]
        validos = np.isfinite(rangos) & (rangos >= msg.range_min) & (rangos <= msg.range_max)
        rangos, angulos = rangos[validos], angulos[validos]

        xs_laser = rangos * np.cos(angulos)
        ys_laser = rangos * np.sin(angulos)
        tx, ty = tf.transform.translation.x, tf.transform.translation.y
        tyaw = self.yaw_de_quaternion(tf.transform.rotation)
        c, s = math.cos(tyaw), math.sin(tyaw)
        return np.stack([tx + c * xs_laser - s * ys_laser,
                         ty + s * xs_laser + c * ys_laser], axis=1)

    def estimar_pose(self, particulas):
        x = float(np.mean(particulas[:, 0]))
        y = float(np.mean(particulas[:, 1]))
        theta = float(math.atan2(np.mean(np.sin(particulas[:, 2])),
                                 np.mean(np.cos(particulas[:, 2]))))
        return (x, y, theta)

    def actualizar_map_odom(self, pose_estimada):
        """map->odom = (map->base) compuesta con la inversa de (odom->base)."""
        self.pose_map_odom = componer_poses(pose_estimada, invertir_pose(self.ultimo_odom))

    def publicar_tf_periodico(self):
        """Re-publica map->odom a frecuencia fija con la hora actual, para que
        la TF no envejezca entre scans (Nav2 la consulta todo el tiempo)."""
        if self.pose_map_odom is None:
            return
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = self.map_frame
        t.child_frame_id = self.odom_frame
        t.transform.translation.x = self.pose_map_odom[0]
        t.transform.translation.y = self.pose_map_odom[1]
        t.transform.rotation = yaw_a_quaternion(self.pose_map_odom[2])
        self.tf_broadcaster.sendTransform(t)

    def publicar_nube(self):
        if self.pub_particulas is None:
            return
        msg = PoseArray()
        msg.header.frame_id = self.map_frame
        msg.header.stamp = self.get_clock().now().to_msg()
        for x, y, theta, _ in self.particulas:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)
            pose.orientation = yaw_a_quaternion(float(theta))
            msg.poses.append(pose)
        self.pub_particulas.publish(msg)

    # ----------------------------------------------- campo de verosimilitud
    def campo_de_probabilidad(self, grid: np.ndarray, resolucion: float) -> np.ndarray:
        mascara = grid >= UMBRAL_OCUPADO
        distancia_px = ndimage.distance_transform_edt(~mascara)
        distancia_m = distancia_px * resolucion
        probabilidad = np.exp(-(distancia_m ** 2) / (2 * self.sigma_sensor ** 2))
        return probabilidad * 100

    # ---------------------------------------------------- filtro de partículas
    def mover_particulas(self, particulas, dx, dy, dtheta, theta0):
        d_trans_x = dx * math.cos(theta0) + dy * math.sin(theta0)
        d_trans_y = -dx * math.sin(theta0) + dy * math.cos(theta0)
        d_rot = dtheta

        # Corrección del error de escala de la odometría (workshop 10, paso 3),
        # antes de calcular el ruido.
        d_trans_x *= self.escala_avance
        d_trans_y *= self.escala_avance
        d_rot *= self.escala_giro

        std_x = math.sqrt(self.gamma1 * d_trans_x ** 2 + self.gamma2 * d_trans_y ** 2 + self.gamma3 * d_rot ** 2 + self.ruido_min_xy ** 2)
        std_y = math.sqrt(self.gamma2 * d_trans_x ** 2 + self.gamma1 * d_trans_y ** 2 + self.gamma3 * d_rot ** 2 + self.ruido_min_xy ** 2)
        std_rot = math.sqrt(self.gamma4 * d_trans_x ** 2 + self.gamma5 * d_trans_y ** 2 + self.gamma6 * d_rot ** 2 + self.ruido_min_theta ** 2)

        n = len(particulas)
        x_hat = d_trans_x - np.random.normal(0.0, std_x, n)
        y_hat = d_trans_y - np.random.normal(0.0, std_y, n)
        rot_hat = d_rot - np.random.normal(0.0, std_rot, n)

        theta = particulas[:, 2]
        nuevas = particulas.copy()
        nuevas[:, 0] = particulas[:, 0] + x_hat * np.cos(theta) - y_hat * np.sin(theta)
        nuevas[:, 1] = particulas[:, 1] + x_hat * np.sin(theta) + y_hat * np.cos(theta)
        nuevas[:, 2] = np.arctan2(np.sin(theta + rot_hat), np.cos(theta + rot_hat))
        return nuevas

    def pesar_particulas(self, particulas, puntos_scan):
        n = len(particulas)
        res = self.info_mapa.resolution
        ox = self.info_mapa.origin.position.x
        oy = self.info_mapa.origin.position.y
        alto, ancho = self.campo.shape

        px = particulas[:, 0][:, None]
        py = particulas[:, 1][:, None]
        c = np.cos(particulas[:, 2])[:, None]
        s = np.sin(particulas[:, 2])[:, None]
        sx = puntos_scan[:, 0][None, :]
        sy = puntos_scan[:, 1][None, :]
        mx = px + c * sx - s * sy
        my = py + s * sx + c * sy

        columna = np.floor((mx - ox) / res).astype(int)
        fila = np.floor((my - oy) / res).astype(int)
        dentro = (fila >= 0) & (fila < alto) & (columna >= 0) & (columna < ancho)

        prob = self.campo[np.clip(fila, 0, alto - 1), np.clip(columna, 0, ancho - 1)]
        prob = np.where(dentro, prob, 1.0)
        prob = np.maximum(prob, 1.0)

        log_pesos = np.sum(np.log(prob), axis=1)
        log_pesos -= log_pesos.max()
        pesos = np.exp(log_pesos)

        suma = pesos.sum()
        if suma <= 0 or not np.isfinite(suma):
            return np.full(n, 1.0 / n)
        return pesos / suma

    def remuestrear(self, particulas, pesos):
        n = len(particulas)
        offset = np.random.uniform()
        pasos = (np.arange(n) + offset) / n
        acumulada = np.cumsum(pesos)
        acumulada[-1] = 1.0
        indices = np.searchsorted(acumulada, pasos)
        nuevas = particulas[indices].copy()
        nuevas[:, 3] = 1.0 / n
        return nuevas


def main(args=None):
    rclpy.init(args=args)
    nodo = Localizador()
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