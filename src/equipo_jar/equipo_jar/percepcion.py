from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.time import Time
from rclpy.duration import Duration

from sensor_msgs.msg import Image, CameraInfo, LaserScan
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

import tf2_ros

from equipo_jar.reglas_percepcion import (
    mascara_rojo, rayos_sobre_rojo, puntos_en_mapa, 
    agrupar_en_rachas, estimar_mitad_caja, corregir_al_centro,
    cuaternion_a_matriz, tramo_mas_cercano, columnas_de_rayos, toca_borde,
    algo_muy_cerca_en_imagen
)
from equipo_jar.reglas_cerebro import cuaternion_a_yaw

class NodePerception(Node):
    def __init__(self):
        super().__init__('percepcion')
        self.declare_parameter('hue_rojo_bajo_1', 0.0)
        self.declare_parameter('hue_rojo_alto_1', 10.0)
        self.declare_parameter('hue_rojo_bajo_2', 170.0)
        self.declare_parameter('hue_rojo_alto_2', 180.0)
        self.declare_parameter('saturacion_min', 120.0)
        self.declare_parameter('valor_min', 80.0)
        self.declare_parameter('ancho_min', 0.10)
        self.declare_parameter('ancho_max', 0.40)
        self.declare_parameter('distancia_minima_valida', 0.25)
        self.declare_parameter('distancia_maxima_valida', 1.2)
        self.declare_parameter('rayos_minimos', 3)
        self.declare_parameter('salto_rango_max', 0.08)
        self.declare_parameter('margen_borde_px', 4)
        self.declare_parameter('desfase_max_imagen_s', 0.10)

        self.hue_rojo_bajo_1 = self.get_parameter('hue_rojo_bajo_1').value
        self.hue_rojo_alto_1 = self.get_parameter('hue_rojo_alto_1').value
        self.hue_rojo_bajo_2 = self.get_parameter('hue_rojo_bajo_2').value
        self.hue_rojo_alto_2 = self.get_parameter('hue_rojo_alto_2').value
        
        self.saturacion_min = self.get_parameter('saturacion_min').value
        self.valor_min = self.get_parameter('valor_min').value
        self.ancho_min = self.get_parameter('ancho_min').value
        self.ancho_max = self.get_parameter('ancho_max').value
        self.rayos_minimos = self.get_parameter('rayos_minimos').value
        
        self.distancia_minima_valida = self.get_parameter('distancia_minima_valida').value
        self.distancia_maxima_valida = self.get_parameter('distancia_maxima_valida').value
        self.salto_rango_max = self.get_parameter('salto_rango_max').value
        self.margen_borde_px = self.get_parameter('margen_borde_px').value
        self.desfase_max_imagen_s = self.get_parameter('desfase_max_imagen_s').value

        self.puente = CvBridge()
        self.ultima_imagen = None
        self.imagenes = deque(maxlen=10)
        self.intrinsecos = None
        self.frame_camara = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(Image, '/cam_1/color/image_raw', self.al_recibir_imagen, qos_profile_sensor_data)
        self.create_subscription(
            CameraInfo, '/cam_1/color/camera_info', 
            self.al_recibir_info, qos_profile_sensor_data
            )
        self.create_subscription(
            LaserScan, '/scan', self.al_recibir_scan, qos_profile_sensor_data
            )

        self.pub_observaciones = self.create_publisher(PointStamped, '/percepcion/observaciones', 10)

    def al_recibir_imagen(self, msg: Image):
        self.ultima_imagen = self.puente.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.frame_camara = msg.header.frame_id
        self.imagenes.append((Time.from_msg(msg.header.stamp).nanoseconds, self.ultima_imagen))

    def al_recibir_info(self, msg: CameraInfo):
        if self.intrinsecos is None:
            fx, fy, cx, cy = msg.k[0], msg.k[4], msg.k[2], msg.k[5]
            self.get_logger().info(
                f"Intrinsecos: fx={fx:.1f} fy={fy:.1f} cx={cx:.1f} cy={cy:.1f}"
            )
            self.intrinsecos = (fx, fy, cx, cy)
        
    def al_recibir_scan(self, msg: LaserScan):
        if self.ultima_imagen is None or self.intrinsecos is None or self.frame_camara is None:
            return
        rangos = np.array(msg.ranges)
        n = len(rangos)
        angulos = msg.angle_min + np.arange(n) * msg.angle_increment
        t_scan = Time.from_msg(msg.header.stamp).nanoseconds
        t_img, imagen = min(self.imagenes, key=lambda par: abs(par[0] - t_scan))
        if abs(t_img - t_scan) * 1e-9 > self.desfase_max_imagen_s:
            return
        try:
            hora = Time.from_msg(msg.header.stamp)
            espera = Duration(seconds=0.1)
            tf_mapa = self.tf_buffer.lookup_transform(
                'map', msg.header.frame_id, hora, timeout=espera
            )
            tf_camara = self.tf_buffer.lookup_transform(
                self.frame_camara, msg.header.frame_id, hora, timeout=espera
            )
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f'Sin TF todavia: {e}', throttle_duration_sec=2.0)
            return
        except tf2_ros.TransformException as e:
            return
        
        tr = tf_mapa.transform.translation
        rot = tf_mapa.transform.rotation

        tr_c = tf_camara.transform.translation
        rot_c = tf_camara.transform.rotation

        R = cuaternion_a_matriz(rot_c.x, rot_c.y, rot_c.z, rot_c.w)
        t = np.array([tr_c.x, tr_c.y, tr_c.z])

        yaw = cuaternion_a_yaw(rot.z, rot.w)

        tx, ty = tr.x, tr.y

        fx, fy, cx, cy = self.intrinsecos

        mascara = mascara_rojo(imagen, self.hue_rojo_bajo_1, self.hue_rojo_alto_1, self.hue_rojo_bajo_2, self.hue_rojo_alto_2,
                  self.saturacion_min, self.valor_min)

        r_sobre_rojo = rayos_sobre_rojo(rangos, angulos, R, t, fx, fy, cx, cy, mascara)
        
        xs_map, ys_map, validos = puntos_en_mapa(rangos, angulos, tx, ty, yaw, max(msg.range_min, self.distancia_minima_valida),
                                                 min(msg.range_max, self.distancia_maxima_valida))

        columnas = columnas_de_rayos(rangos, angulos, R, t, fx, cx)
        ancho_imagen = mascara.shape[1]
        rango_min = max(msg.range_min, self.distancia_minima_valida)
        if algo_muy_cerca_en_imagen(rangos, columnas, ancho_imagen, rango_min):
            return

        rachas = agrupar_en_rachas(r_sobre_rojo & validos)

        for r in rachas:
            if len(r) < self.rayos_minimos:
                continue
            r = tramo_mas_cercano(r, rangos, self.salto_rango_max, self.rayos_minimos)
            if len(r) < self.rayos_minimos:
                continue
            if toca_borde(r, columnas, ancho_imagen, self.margen_borde_px):
                continue
            x_cara = xs_map[r].mean()
            y_cara = ys_map[r].mean()

            ancho = np.hypot(xs_map[r[-1]] - xs_map[r[0]], ys_map[r[-1]] - ys_map[r[0]])

            mitad = estimar_mitad_caja(ancho, self.ancho_min, self.ancho_max)

            x_centro, y_centro = corregir_al_centro(x_cara, y_cara, tx, ty, mitad)
            self.publicar_observacion(x_centro, y_centro, msg.header.stamp)

            


    def publicar_observacion(self, x_centro, y_centro, stamp):
        obs = PointStamped()
        obs.header.frame_id = 'map'
        obs.header.stamp = stamp
        obs.point.x = float(x_centro)
        obs.point.y = float(y_centro)
        self.pub_observaciones.publish(obs)

def main(args=None):
    rclpy.init(args=args)
    nodo = NodePerception()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()