"""Exploración del equipo: cobertura del mapa conocido, con goals a Nav2.

El juez publica el mapa del laberinto (/map); lo desconocido son las cajas.
Este nodo decide A DÓNDE ir (qué zona todavía no se vio con la cámara) y deja
que Nav2 planifique y esquive (acción NavigateToPose). No publica /cmd_vel
salvo para girar despacio en el lugar y apuntar la cámara a lo no visto.

Diferencias con el workshop 08:
  - La cobertura se marca solo dentro del campo de visión de la CÁMARA y hasta
    un alcance corto: las cajas se detectan con la cámara (frontal), no con el
    lidar de 360°.
  - Sin 2D Pose Estimate ni topics de debug: arranca solo cuando hay /map, TF
    map->base y Nav2 listo.
  - El objetivo se elige por distancia de camino (BFS sobre el mapa con margen
    a las paredes), no en línea recta.
  - Goals solo sobre celdas libres con margen a las paredes; los fallidos se
    descartan para no quedar en un bucle de recoveries.
  - Regreso a la largada y /done (parámetros volver_a_la_largada y publicar_done).

Estados: esperando -> asentando -> explorando <-> orientando -> regresando -> terminado
"""
import math
from collections import deque
from functools import partial

import numpy as np
import rclpy
import scipy.ndimage as nd
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

ESPERANDO = 'esperando'
ASENTANDO = 'asentando'
EXPLORANDO = 'explorando'
ORIENTANDO = 'orientando'
REGRESANDO = 'regresando'
TERMINADO = 'terminado'


def normalizar_angulo(a: float) -> float:
    return math.atan2(math.sin(a), math.cos(a))


def yaw_de_quaternion(q: Quaternion) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_a_quaternion(yaw: float) -> Quaternion:
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.z = math.sin(yaw * 0.5)
    return q


def trazar_rayo(fila_desde, col_desde, fila_hasta, col_hasta):
    """Bresenham entre dos celdas: lista de (fila, col), origen y destino incluidos."""
    f0, c0, f1, c1 = int(fila_desde), int(col_desde), int(fila_hasta), int(col_hasta)
    dc, df = abs(c1 - c0), abs(f1 - f0)
    sc = 1 if c0 < c1 else -1
    sf = 1 if f0 < f1 else -1
    error = dc - df
    f, c = f0, c0
    celdas = []
    while True:
        celdas.append((f, c))
        if (f, c) == (f1, c1):
            break
        e2 = 2 * error
        if e2 > -df:
            error -= df
            c += sc
        if e2 < dc:
            error += dc
            f += sf
    return celdas


class Exploracion(Node):

    def __init__(self):
        super().__init__('exploracion')
        P = self.declare_parameter
        P('map_frame', 'map')
        P('base_frame', 'base_footprint')
        P('factor_escala', 4)                 # celdas finas por lado de una celda de cobertura
        P('distancia_minima_valida', 0.25)    # descarta rebotes del lidar contra el propio robot
        P('rango_cobertura', 2.0)             # m: hasta dónde cuenta "visto" con la cámara
        P('fov_camara_grados', 70.0)          # campo de visión horizontal que se marca como visto
        P('submuestreo_cobertura', 3)
        P('radio_robot', 0.18)
        P('margen_seguridad', 0.05)
        P('margen_objetivo', 0.30)            # m mínimos a una pared para ser destino
        P('tiempo_asentamiento_s', 4.0)
        P('tiempo_max_objetivo_s', 45.0)
        P('max_fallos_celda', 2)
        P('giro_orientar', 0.3)               # rad/s: lento, percepción ignora giros rápidos
        P('tolerancia_orientar_grados', 8.0)
        P('tiempo_max_orientar_s', 8.0)
        P('radio_orientar', 2.0)
        P('tiempo_sin_avance_s', 15.0)        # s sin avanzar 0.15 m con un goal activo -> abandonarlo
        P('radio_ya_visto', 0.35)             # m: alrededor del robot se da por visto (la cámara no ve tan cerca)
        P('distancia_min_destino', 0.5)       # m: no elegir destinos más cerca que esto (Nav2 los da por cumplidos)
        P('max_elecciones_celda', 3)          # veces que se puede elegir una celda sin lograr verla
        P('peso_giro_m_por_rad', 0.3).        # m equivalentes por radián de giro al elegir celda
        P('tiempo_regreso_s', 480.0)          # s desde que llega /map
        P('volver_a_la_largada', True)
        P('intentos_regreso', 3)
        P('publicar_done', True)
        P('cantidad_victimas', 5)             # lo setea el launch
        P('frecuencia_control_hz', 5.0)
        P('frecuencia_cobertura_hz', 2.0)

        g = lambda n: self.get_parameter(n).value
        self.map_frame, self.base_frame = g('map_frame'), g('base_frame')
        self.factor = g('factor_escala')
        self.dist_min_valida = g('distancia_minima_valida')
        self.rango_cob = g('rango_cobertura')
        self.fov = math.radians(g('fov_camara_grados'))
        self.submuestreo = g('submuestreo_cobertura')
        self.radio_robot, self.margen_seg = g('radio_robot'), g('margen_seguridad')
        self.margen_obj = g('margen_objetivo')
        self.t_asent = g('tiempo_asentamiento_s')
        self.t_max_obj = g('tiempo_max_objetivo_s')
        self.max_fallos = g('max_fallos_celda')
        self.w_orientar = g('giro_orientar')
        self.tol_orientar = math.radians(g('tolerancia_orientar_grados'))
        self.t_max_orientar = g('tiempo_max_orientar_s')
        self.radio_orientar = g('radio_orientar')
        self.radio_ya_visto = g('radio_ya_visto')
        self.dist_min_destino = g('distancia_min_destino')
        self.max_elecciones = g('max_elecciones_celda')
        self.peso_giro = g('peso_giro_m_por_rad')
        self.t_sin_avance = g('tiempo_sin_avance_s')
        self.t_regreso = g('tiempo_regreso_s')
        self.volver = g('volver_a_la_largada')
        self.max_intentos_regreso = g('intentos_regreso')
        self.publicar_done = g('publicar_done')
        self.n_victimas = g('cantidad_victimas')

        # --- mapa y cobertura (se arman al recibir /map)
        self.mapa_listo = False
        self.t_mapa = None
        # --- estado
        self.estado = ESPERANDO
        self.ultimo_scan = None
        self.reportes = 0
        self.activo = None           # goal en curso: dict(id, tipo, celda, t, gh)
        self.sig_id = 0
        self.llegue = False
        self.no_enviar_hasta = None
        self.pose_largada = None
        self.t_listo = None
        self.regreso_ok = False
        self.intentos_regreso_hechos = 0
        self.yaw_objetivo = None
        self.t_orientar = None
        self.ticks_log = 0
        self.pos_ref = None          # vigilante de progreso: dónde estaba hace un rato
        self.t_ref = None

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        qos_mapa = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.recibir_mapa, qos_mapa)
        self.create_subscription(LaserScan, '/scan', self.recibir_scan, qos_profile_sensor_data)
        self.create_subscription(PointStamped, '/victimas', self.contar_reporte, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
        self.cliente = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.create_timer(1.0 / g('frecuencia_control_hz'), self.ciclo)
        self.create_timer(1.0 / g('frecuencia_cobertura_hz'), self.marcar_cobertura)
        self.get_logger().info('Exploración lista. Esperando /map, TF y Nav2...')

    # =========================================================== callbacks
    def contar_reporte(self, _msg):
        self.reportes += 1

    def recibir_scan(self, msg):
        self.ultimo_scan = msg

    def recibir_mapa(self, msg: OccupancyGrid):
        if self.mapa_listo:
            return
        alto, ancho = msg.info.height, msg.info.width
        datos = np.array(msg.data, dtype=np.int16).reshape((alto, ancho))
        res, f = msg.info.resolution, self.factor
        libre_fino = datos == 0
        # m a la pared (o celda desconocida) más cercana
        dist = nd.distance_transform_edt(datos == 0) * res

        # Celdas navegables para el BFS (con margen del chasis) y "snap" a la más cercana
        self.libre_nav = libre_fino & (dist >= self.radio_robot + self.margen_seg)
        _, self.idx_nav = nd.distance_transform_edt(~self.libre_nav, return_indices=True)

        # Grilla gruesa de cobertura
        ac, an = alto // f, ancho // f
        bloques = libre_fino[:ac * f, :an * f].reshape(ac, f, an, f)
        self.libre_c = bloques.mean(axis=(1, 3)) > 0.5
        self.cubierta = np.zeros((ac, an), dtype=bool)
        self.fallos = np.zeros((ac, an), dtype=int)
        self.elecciones = np.zeros((ac, an), dtype=int)
        self.res_c = res * f
        self.res = res
        self.ox, self.oy = msg.info.origin.position.x, msg.info.origin.position.y

        # Punto-meta de cada celda gruesa: la celda fina libre más alejada de las paredes
        libre_meta = libre_fino & (dist >= self.margen_obj)
        self.meta = np.full((ac, an, 2), -1, dtype=int)
        for i in range(ac):
            for j in range(an):
                blq = libre_meta[i * f:(i + 1) * f, j * f:(j + 1) * f]
                if blq.any():
                    d = np.where(blq, dist[i * f:(i + 1) * f, j * f:(j + 1) * f], -1.0)
                    fi, fj = divmod(int(np.argmax(d)), f)
                    self.meta[i, j] = (i * f + fi, j * f + fj)
        self.tiene_meta = self.meta[:, :, 0] >= 0

        self.mapa_listo = True
        self.t_mapa = self.get_clock().now()
        self.get_logger().info(
            f'Mapa recibido ({ancho}x{alto}). Cobertura: {ac}x{an} celdas, '
            f'{int(np.sum(self.libre_c))} libres, {int(np.sum(self.libre_c & self.tiene_meta))} con destino posible.')

    # =========================================================== geometría
    def pose_robot(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
        except Exception:
            return None
        t = tf.transform.translation
        return t.x, t.y, yaw_de_quaternion(tf.transform.rotation)

    def segundos_desde_mapa(self):
        if self.t_mapa is None:
            return 0.0
        return (self.get_clock().now() - self.t_mapa).nanoseconds * 1e-9

    def fino_a_mundo(self, fila, col):
        return self.ox + (col + 0.5) * self.res, self.oy + (fila + 0.5) * self.res

    def mundo_a_fino(self, x, y):
        alto, ancho = self.libre_nav.shape
        col = int(np.clip(int((x - self.ox) / self.res), 0, ancho - 1))
        fila = int(np.clip(int((y - self.oy) / self.res), 0, alto - 1))
        return fila, col

    def mundo_a_grueso(self, x, y):
        return int((y - self.oy) / self.res_c), int((x - self.ox) / self.res_c)

    def dentro_grueso(self, fila, col):
        return 0 <= fila < self.cubierta.shape[0] and 0 <= col < self.cubierta.shape[1]

    # =========================================================== cobertura
    def marcar_cobertura(self):
        """Marca como vistas las celdas dentro del campo de visión de la cámara."""
        try:
            if (not self.mapa_listo or self.ultimo_scan is None
                    or self.estado not in (ASENTANDO, EXPLORANDO, ORIENTANDO)):
                return
            msg = self.ultimo_scan
            try:
                tf_l = self.tf_buffer.lookup_transform(self.map_frame, msg.header.frame_id, Time())
                tf_b = self.tf_buffer.lookup_transform(self.map_frame, self.base_frame, Time())
            except Exception:
                return
            lx, ly = tf_l.transform.translation.x, tf_l.transform.translation.y
            yaw_l = yaw_de_quaternion(tf_l.transform.rotation)
            yaw_b = yaw_de_quaternion(tf_b.transform.rotation)
            f0, c0 = self.mundo_a_grueso(lx, ly)
            if not self.dentro_grueso(f0, c0):
                return

            # Alrededor del robot se da por visto: la cámara no enfoca tan cerca y ahí
            # no puede haber una caja sin que el robot la esté tocando.
            bx, by = tf_b.transform.translation.x, tf_b.transform.translation.y
            fb, cb = self.mundo_a_grueso(bx, by)
            r = int(math.ceil(self.radio_ya_visto / self.res_c))
            for df in range(-r, r + 1):
                for dc in range(-r, r + 1):
                    if math.hypot(df, dc) * self.res_c <= self.radio_ya_visto:
                        fila, col = fb + df, cb + dc
                        if self.dentro_grueso(fila, col) and self.libre_c[fila, col]:
                            self.cubierta[fila, col] = True

            rangos = np.array(msg.ranges)
            n = len(rangos)
            ang = msg.angle_min + np.arange(n) * msg.angle_increment
            # solo los rayos que caen dentro del campo de visión de la cámara (frente del robot)
            rel = np.array([normalizar_angulo(a + yaw_l - yaw_b) for a in ang])
            en_fov = np.abs(rel) <= self.fov / 2.0

            minimo = max(msg.range_min, self.dist_min_valida)
            finito = np.isfinite(rangos)
            con_ret = finito & (rangos >= minimo) & (rangos <= msg.range_max)
            rebote = finito & (rangos < minimo)
            alcance = np.where(con_ret, np.minimum(rangos, self.rango_cob), self.rango_cob)

            for i in range(0, n, self.submuestreo):
                if not en_fov[i] or rebote[i]:
                    continue
                a = yaw_l + ang[i]
                xf, yf = lx + alcance[i] * math.cos(a), ly + alcance[i] * math.sin(a)
                f1, c1 = self.mundo_a_grueso(xf, yf)
                for fila, col in trazar_rayo(f0, c0, f1, c1):
                    if not self.dentro_grueso(fila, col):
                        break
                    if self.libre_c[fila, col]:
                        self.cubierta[fila, col] = True
        except Exception as e:
            self.get_logger().error(f'marcar_cobertura: {e}', throttle_duration_sec=2.0)

    # =========================================================== elección de destino
    def distancias_bfs(self, fila, col):
        alto, ancho = self.libre_nav.shape
        fila, col = int(self.idx_nav[0][fila, col]), int(self.idx_nav[1][fila, col])
        libre = self.libre_nav.tolist()
        d = [[-1] * ancho for _ in range(alto)]
        d[fila][col] = 0
        cola = deque([(fila, col)])
        while cola:
            f, c = cola.popleft()
            dn = d[f][c] + 1
            for df in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if df == 0 and dc == 0:
                        continue
                    nf, nc = f + df, c + dc
                    if not (0 <= nf < alto and 0 <= nc < ancho) or d[nf][nc] >= 0 or not libre[nf][nc]:
                        continue
                    if df and dc and not (libre[f + df][c] and libre[f][c + dc]):
                        continue   # no cortar esquinas de pared
                    d[nf][nc] = dn
                    cola.append((nf, nc))
        return np.array(d)

        def elegir_celda(self, x, y, yaw):
        """Celda pendiente con menor costo = metros de camino + peso_giro * radianes a girar."""
        pend = self.libre_c & ~self.cubierta & self.tiene_meta & (self.fallos < self.max_fallos)
        if not pend.any():
            return None
        d = self.distancias_bfs(*self.mundo_a_fino(x, y))
        mejor, mejor_costo = None, None
        for i, j in zip(*np.where(pend)):
            fi, fj = self.meta[i, j]
            dd = d[fi, fj]
            if dd < 0:
                continue          # inalcanzable
            gx, gy = self.fino_a_mundo(fi, fj)
            if math.hypot(gx - x, gy - y) < self.dist_min_destino:
                continue          # pegado al robot: Nav2 lo daría por cumplido sin moverse
            camino_m = dd * self.res
            giro = abs(normalizar_angulo(math.atan2(gy - y, gx - x) - yaw))
            costo = camino_m + self.peso_giro * giro
            if mejor is None or costo < mejor_costo:
                mejor, mejor_costo = (int(i), int(j)), costo
        if mejor is None:
            self.get_logger().info(
                f'{int(pend.sum())} celdas restantes inalcanzables o pegadas al robot: se descartan.')
            self.cubierta |= pend
        return mejor

    def yaw_hacia_lo_no_visto(self, x, y, yaw_actual):
        """Rumbo (de 24) que deja más celdas sin ver dentro del campo de visión de la cámara."""
        filas, cols = np.where(self.libre_c & ~self.cubierta)
        if len(filas) == 0:
            return None
        cx = self.ox + (cols + 0.5) * self.res_c
        cy = self.oy + (filas + 0.5) * self.res_c
        dx, dy = cx - x, cy - y
        dist = np.hypot(dx, dy)
        cerca = (dist > 0.15) & (dist <= self.radio_orientar)
        if cerca.sum() < 2:
            return None
        angs = np.arctan2(dy[cerca], dx[cerca])
        mejor, mejor_n = None, 2     # solo se gira si hay al menos 3 celdas sin ver en esa dirección
        for k in range(24):
            h = -math.pi + k * (2 * math.pi / 24)
            cuenta = int(np.sum(np.abs(np.arctan2(np.sin(angs - h), np.cos(angs - h))) <= self.fov / 2.0))
            mas_cerca = mejor is not None and abs(normalizar_angulo(h - yaw_actual)) < abs(normalizar_angulo(mejor - yaw_actual))
            if cuenta > mejor_n or (cuenta == mejor_n and mas_cerca):
                mejor, mejor_n = h, cuenta
        # Si la cámara ya apunta casi para ahí, no hace falta girar
        if mejor is not None and abs(normalizar_angulo(mejor - yaw_actual)) < math.radians(25):
            return None
        return mejor

    # =========================================================== goals a Nav2
    def enviar_goal(self, x, y, yaw, tipo, celda=None):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.map_frame
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation = yaw_a_quaternion(yaw)
        self.sig_id += 1
        self.pos_ref = None
        self.activo = dict(id=self.sig_id, tipo=tipo, celda=celda, gh=None,
                           t=self.get_clock().now())
        self.get_logger().info(f'Goal #{self.sig_id} ({tipo}) -> ({x:.2f}, {y:.2f})')
        fut = self.cliente.send_goal_async(goal)
        fut.add_done_callback(partial(self.respuesta_goal, self.sig_id))

    def respuesta_goal(self, goal_id, fut):
        try:
            gh = fut.result()
            vigente = self.activo is not None and self.activo['id'] == goal_id
            if not gh.accepted:
                if vigente:
                    self.activo = None
                    self.no_enviar_hasta = self.get_clock().now().nanoseconds + int(2e9)
                    self.get_logger().warn('Nav2 rechazó el goal; reintento en 2 s.')
                return
            if not vigente:
                gh.cancel_goal_async()   # lo cancelamos mientras viajaba la respuesta
                return
            self.activo['gh'] = gh
            gh.get_result_async().add_done_callback(partial(self.resultado_goal, goal_id))
        except Exception as e:
            self.get_logger().error(f'respuesta_goal: {e}')

    def resultado_goal(self, goal_id, fut):
        try:
            if self.activo is None or self.activo['id'] != goal_id:
                return
            act, self.activo = self.activo, None
            ok = fut.result().status == GoalStatus.STATUS_SUCCEEDED
            if act['tipo'] == 'cobertura':
                if ok:
                    self.llegue = True
                else:
                    self.fallos[act['celda']] += 1
                    self.get_logger().warn(f'Goal #{goal_id} falló (celda {act["celda"]}, '
                                           f'fallos={self.fallos[act["celda"]]}).')
            else:
                if ok:
                    self.regreso_ok = True
                else:
                    self.intentos_regreso_hechos += 1
        except Exception as e:
            self.get_logger().error(f'resultado_goal: {e}')

    def cancelar_activo(self):
        if self.activo is not None:
            gh = self.activo.get('gh')
            self.activo = None
            if gh is not None:
                gh.cancel_goal_async()

    # =========================================================== máquina de estados
    def cambiar(self, nuevo):
        if nuevo != self.estado:
            self.get_logger().info(f'Estado: {self.estado} -> {nuevo}')
            self.estado = nuevo

    def terminar(self):
        self.pub_cmd.publish(Twist())
        if self.publicar_done:
            self.pub_done.publish(Empty())
            self.get_logger().info('/done publicado.')
        self.cambiar(TERMINADO)

    def iniciar_regreso(self):
        self.cancelar_activo()
        self.pub_cmd.publish(Twist())
        self.regreso_ok = False
        self.intentos_regreso_hechos = 0
        if self.volver:
            self.cambiar(REGRESANDO)
        else:
            self.terminar()

    def ciclo(self):
        try:
            if not self.mapa_listo or self.estado == TERMINADO:
                return
            pose = self.pose_robot()
            if pose is None:
                return
            ahora = self.get_clock().now()
            x, y, yaw = pose

            if self.estado == ESPERANDO:
                if self.cliente.server_is_ready():
                    self.t_listo = ahora
                    self.cambiar(ASENTANDO)
                return

            if self.estado == ASENTANDO:
                if (ahora - self.t_listo).nanoseconds * 1e-9 >= self.t_asent:
                    self.pose_largada = (x, y)
                    self.get_logger().info(f'Largada guardada en ({x:.2f}, {y:.2f}).')
                    self.cambiar(EXPLORANDO)
                return

            # Cortes globales: ya se reportaron todas, o se acaba el tiempo
            if self.estado in (EXPLORANDO, ORIENTANDO):
                if self.reportes >= self.n_victimas or self.segundos_desde_mapa() >= self.t_regreso:
                    self.get_logger().info(
                        f'Fin de exploración (reportes={self.reportes}/{self.n_victimas}, '
                        f't={self.segundos_desde_mapa():.0f}s).')
                    self.iniciar_regreso()
                    return

            self.ticks_log += 1
            if self.ticks_log % 50 == 0 and self.estado == EXPLORANDO:
                total = int(np.sum(self.libre_c & self.tiene_meta))
                vistas = int(np.sum(self.libre_c & self.tiene_meta & self.cubierta))
                self.get_logger().info(f'Cobertura: {vistas}/{total} celdas.')

            if self.estado == EXPLORANDO:
                self.explorar(x, y, yaw, ahora)
            elif self.estado == ORIENTANDO:
                self.orientar(yaw, ahora)
            elif self.estado == REGRESANDO:
                self.regresar(x, y)
        except Exception as e:
            self.get_logger().error(f'ciclo: {e}', throttle_duration_sec=2.0)

    def explorar(self, x, y, yaw, ahora):
        if self.llegue:     # llegó a un destino: apuntar la cámara a lo no visto
            self.llegue = False
            self.yaw_objetivo = self.yaw_hacia_lo_no_visto(x, y, yaw)
            if self.yaw_objetivo is not None:
                self.t_orientar = ahora
                self.cambiar(ORIENTANDO)
                return

        if self.activo is not None:
            celda = self.activo['celda']
            # Vigilante de progreso: si no avanzó 0.15 m en t_sin_avance, abandonar el goal
            if self.pos_ref is None or math.hypot(x - self.pos_ref[0], y - self.pos_ref[1]) > 0.15:
                self.pos_ref, self.t_ref = (x, y), ahora
            elif (ahora - self.t_ref).nanoseconds * 1e-9 > self.t_sin_avance:
                self.get_logger().warn(f'Sin avance hace {self.t_sin_avance:.0f} s; abandono la celda {celda}.')
                self.fallos[celda] += 1
                self.cancelar_activo()
                self.pos_ref = None
                return
            if (ahora - self.activo['t']).nanoseconds * 1e-9 > self.t_max_obj:
                self.get_logger().warn('Goal con timeout; lo cancelo.')
                self.fallos[celda] += 1
                self.cancelar_activo()
                return
            elif self.cubierta[celda]:
                # Se vio de paso: seguir con otro destino SIN cancelar. Al llegar el goal
                # nuevo, Nav2 reemplaza al actual sin frenar el robot.
                self.activo = None
            else:
                return

        if self.no_enviar_hasta is not None and ahora.nanoseconds < self.no_enviar_hasta:
            return
        celda = self.elegir_celda(x, y, yaw)
        if celda is None:
            self.get_logger().info('No queda nada por cubrir.')
            self.iniciar_regreso()
            return
        self.elecciones[celda] += 1
        if self.elecciones[celda] > self.max_elecciones:
            self.get_logger().warn(f'Celda {celda} elegida {self.max_elecciones} veces sin verse: la descarto.')
            self.cubierta[celda] = True
            return
        fi, fj = self.meta[celda]
        gx, gy = self.fino_a_mundo(fi, fj)
        self.enviar_goal(gx, gy, math.atan2(gy - y, gx - x), 'cobertura', celda)

    def orientar(self, yaw, ahora):
        err = normalizar_angulo(self.yaw_objetivo - yaw)
        if abs(err) < self.tol_orientar or (ahora - self.t_orientar).nanoseconds * 1e-9 > self.t_max_orientar:
            self.pub_cmd.publish(Twist())
            self.cambiar(EXPLORANDO)
            return
        w = float(np.clip(1.5 * err, -self.w_orientar, self.w_orientar))
        if abs(w) < 0.1:
            w = math.copysign(0.1, err)
        cmd = Twist()
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)

    def regresar(self, x, y):
        gx, gy = self.pose_largada
        if self.activo is not None:
            return
        if self.regreso_ok or math.hypot(gx - x, gy - y) <= 0.15:
            self.terminar()
        elif self.intentos_regreso_hechos >= self.max_intentos_regreso:
            self.get_logger().warn('No pude volver a la largada; termino igual.')
            self.terminar()
        else:
            self.enviar_goal(gx, gy, 0.0, 'regreso')


def main(args=None):
    rclpy.init(args=args)
    nodo = Exploracion()
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