# exploracion: el lidar encuentra cosas y la camara confirma si son rojas
import math
from functools import partial

import numpy as np
import rclpy
from action_msgs.msg import GoalStatus
from geometry_msgs.msg import PointStamped, Quaternion, Twist
from nav2_msgs.action import NavigateToPose
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Empty
from tf2_ros import Buffer, TransformListener

from equipo_jar.reglas_anomalias import (INALCANZABLE, NO_ROJO, PENDIENTE, ROJO,
                                         DetectorAnomalias, RegistroCandidatos,
                                         puntos_en_mapa)
from equipo_jar.reglas_cobertura import (Cobertura, elegir_punto_de_vista,
                                         normalizar_angulo, punto_para_mirar)
from equipo_jar.reglas_escape import (despeje_en_direccion, direccion_escape,
                                      tiempo_reserva)
from equipo_jar.reglas_estrategia import (INSPECCIONAR, REGRESAR,
                                          decidir, elegir_inspeccion)
from equipo_jar.reglas_mapa import (Grilla, distancias_camino, grilla_navegable,
                                    punto_intermedio)

ESPERANDO = 'esperando'
ASENTANDO = 'asentando'
DECIDIR = 'decidir'
NAVEGANDO = 'navegando'
ORIENTANDO = 'orientando'
MIRANDO = 'mirando'
ESCAPANDO = 'escapando'
REGRESANDO = 'regresando'
AJUSTE_FINAL = 'ajuste_final'
FINALIZANDO = 'finalizando'
TERMINADO = 'terminado'


def yaw_de_quaternion(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))


def yaw_a_quaternion(yaw):
    q = Quaternion()
    q.w = math.cos(yaw * 0.5)
    q.z = math.sin(yaw * 0.5)
    return q


class Exploracion(Node):

    def __init__(self):
        super().__init__('exploracion')
        P = self.declare_parameter
        P('map_frame', 'map')
        P('base_frame', 'base_footprint')
        P('cantidad_victimas', 5)
        P('duracion_corrida_s', 600.0)
        P('margen_final_s', 12.0)
        P('tiempo_asentamiento_s', 3.0)
        P('frecuencia_control_hz', 5.0)
        P('procesar_cada_n_scans', 1)
        P('radio_nav', 0.16)
        P('radio_paso', 0.12)
        P('radio_giro', 0.25)
        P('footprint_adelante', 0.13)
        P('footprint_atras', 0.19)
        P('footprint_costado', 0.11)
        P('tau', 0.10)
        P('tau_por_metro', 0.03)
        P('rango_max_votos', 2.5)
        P('min_votos', 3)
        P('razon_min', 0.5)
        P('tam_min', 0.06)
        P('tam_max', 0.50)
        P('salud_min', 0.6)
        P('radio_asociacion', 0.35)
        P('radio_rojo', 0.45)
        P('rango_cobertura', 3.0)
        P('ganancia_min', 25)
        P('costo_fijo_m', 1.0)
        P('paso_muestreo', 5)
        P('dist_min_destino', 0.5)
        P('tiempo_mirar_s', 2.5)
        P('max_intentos_mirar', 2)
        P('giro_orientar', 0.4)
        P('tolerancia_orientar_grados', 8.0)
        P('tiempo_max_orientar_s', 8.0)
        P('tiempo_sin_progreso_s', 8.0)
        P('progreso_min_m', 0.10)
        P('radio_zona_prohibida', 0.40)
        P('duracion_zona_prohibida_s', 30.0)
        P('escape_velocidad', 0.08)
        P('escape_distancia', 0.12)
        P('escape_despeje_min', 0.12)
        P('escape_tiempo_max_s', 3.0)
        P('volver_a_la_largada', True)
        P('publicar_done', True)
        P('velocidad_efectiva', 0.15)
        P('factor_reserva', 1.5)
        P('margen_reserva_s', 30.0)
        P('intentos_regreso', 3)
        P('progreso_regreso_m', 0.30)
        P('tramo_regreso_m', 1.2)
        P('tramo_exploracion_m', 1.2)
        P('tolerancia_ajuste_final', 0.06)
        P('tiempo_ajuste_final_s', 8.0)
        P('velocidad_ajuste_final', 0.08)

        self.p = {n: par.value for n, par in self.get_parameters_by_prefix('').items()}
        self.n_victimas = int(self.p['cantidad_victimas'])

        self.estado = ESPERANDO
        self.grilla = None
        self.det = None
        self.cob = None
        self.reg = RegistroCandidatos(self.p['radio_asociacion'])
        self.t_mapa = None
        self.t_estado = None
        self.largada = None                     # (x, y, yaw)
        self.reportes = 0
        self.activo = None
        self.sig_id = 0
        self.zonas_prohibidas = []              # (x, y, radio, t_vence_ns)
        self.inspeccion = None
        self.escape = None
        self.intentos_regreso = 0
        self.mejor_camino_regreso = None
        self.tramo_proximo = False
        self.ultimo_scan = None
        self.cuenta_scans = 0
        self.laser_en_base = None               # (lx, ly, yaw) estatico
        self.t_finalizando = None
        self.n_pendientes_al_salir = 0
        self.t_chequeo_meta = None
        self.reportes_hechos = []

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        qos_mapa = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST,
                              durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.create_subscription(OccupancyGrid, '/map', self.al_recibir_mapa, qos_mapa)
        self.create_subscription(LaserScan, '/scan', self.al_recibir_scan, qos_profile_sensor_data)
        self.create_subscription(PointStamped, '/victimas', self.al_recibir_reporte, 10)
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_done = self.create_publisher(Empty, '/done', 10)
        self.pub_sugerencias = self.create_publisher(PointStamped, '/exploracion/sugerencias', 50)
        self.pub_rellenar = self.create_publisher(Empty, '/exploracion/rellenar', 10)
        self.cliente = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        self.limpiar_local = self.create_client(
            ClearEntireCostmap, '/local_costmap/clear_entirely_local_costmap')
        self.limpiar_global = self.create_client(
            ClearEntireCostmap, '/global_costmap/clear_entirely_global_costmap')

        self.create_timer(1.0 / self.p['frecuencia_control_hz'], self.ciclo)
        self.create_timer(1.0, self.actualizar_candidatos)
        self.get_logger().info(f'Exploración lista (N={self.n_victimas}). Esperando /map, TF y Nav2...')

    def ahora_ns(self):
        return self.get_clock().now().nanoseconds

    def segundos_desde(self, t_ns):
        return (self.ahora_ns() - t_ns) * 1e-9

    def t_corrida(self):
        return 0.0 if self.t_mapa is None else self.segundos_desde(self.t_mapa)

    def cambiar(self, nuevo):
        if nuevo != self.estado:
            self.get_logger().info(f'Estado: {self.estado} -> {nuevo} (t={self.t_corrida():.0f}s)')
            self.estado = nuevo
            self.t_estado = self.ahora_ns()

    def pose_robot(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.p['map_frame'], self.p['base_frame'], Time())
        except Exception:
            return None
        t = tf.transform.translation
        return t.x, t.y, yaw_de_quaternion(tf.transform.rotation)

    def frenar(self):
        self.pub_cmd.publish(Twist())

    def esquinas_desde_laser(self):
        lx, ly, _ = self.laser_en_base
        a, b, s = self.p['footprint_adelante'], self.p['footprint_atras'], self.p['footprint_costado']
        return [(a - lx, s - ly), (a - lx, -s - ly), (-b - lx, -s - ly), (-b - lx, s - ly)]

    def zonas_vigentes(self):
        ahora = self.ahora_ns()
        self.zonas_prohibidas = [z for z in self.zonas_prohibidas if z[3] > ahora]
        return [(x, y, r) for x, y, r, _ in self.zonas_prohibidas]

    def prohibir_zona(self, x, y):
        vence = self.ahora_ns() + int(self.p['duracion_zona_prohibida_s'] * 1e9)
        self.zonas_prohibidas.append((x, y, self.p['radio_zona_prohibida'], vence))

    def al_recibir_mapa(self, msg):
        if self.grilla is not None:
            return
        datos = np.array(msg.data, dtype=np.int16).reshape((msg.info.height, msg.info.width))
        self.grilla = Grilla(datos, msg.info.resolution,
                             msg.info.origin.position.x, msg.info.origin.position.y)
        p = self.p
        self.det = DetectorAnomalias(self.grilla, tau=p['tau'], tau_por_metro=p['tau_por_metro'],
                                     rango_max_votos=p['rango_max_votos'], min_votos=p['min_votos'],
                                     razon_min=p['razon_min'], tam_min=p['tam_min'],
                                     tam_max=p['tam_max'])
        self.cob = Cobertura(self.grilla)
        self.t_mapa = self.ahora_ns()
        self.get_logger().info(f'Mapa {msg.info.width}x{msg.info.height}: arranca la corrida.')

    def al_recibir_reporte(self, msg):
        # recien cuando el gestor reporta lo damos por rojo, si no el robot se iba antes de confirmar
        self.reportes += 1
        x, y = msg.point.x, msg.point.y
        self.reportes_hechos.append((x, y))
        cand = self.reg.marcar_rojo_cerca(x, y, self.p['radio_rojo'])
        if cand is not None:
            self.get_logger().info(f'Candidato {cand["id"]} es ROJO: reportado en ({x:.2f}, {y:.2f}).')

    def al_recibir_scan(self, msg):
        if self.grilla is None:
            return
        try:
            if self.laser_en_base is None:
                tf_lb = self.tf_buffer.lookup_transform(self.p['base_frame'], msg.header.frame_id, Time())
                self.laser_en_base = (tf_lb.transform.translation.x, tf_lb.transform.translation.y,
                                      yaw_de_quaternion(tf_lb.transform.rotation))
            try:
                tf = self.tf_buffer.lookup_transform(self.p['map_frame'], msg.header.frame_id,
                                                     Time.from_msg(msg.header.stamp))
            except Exception:
                tf = self.tf_buffer.lookup_transform(self.p['map_frame'], msg.header.frame_id, Time())
        except Exception:
            return
        xl, yl = tf.transform.translation.x, tf.transform.translation.y
        yawl = yaw_de_quaternion(tf.transform.rotation)
        self.ultimo_scan = (msg, xl, yl, yawl)
        self.cuenta_scans += 1
        if self.cuenta_scans % self.p['procesar_cada_n_scans'] != 0:
            return
        if self.estado == TERMINADO:
            return
        try:
            n = len(msg.ranges)
            ang = msg.angle_min + np.arange(n) * msg.angle_increment
            rangos = np.array(msg.ranges, dtype=float)
            rmin = max(msg.range_min, 0.05)
            xs, ys, val = puntos_en_mapa(rangos, ang, xl, yl, yawl, rmin, msg.range_max)
            salud = self.det.salud(xs, ys, val)
            if salud >= self.p['salud_min']:
                self.det.agregar_scan(xs, ys, val, xl, yl, rangos, ang, yawl)
            else:
                self.get_logger().warn(f'Localización dudosa (salud {salud:.2f}): no voto anomalías.',
                                       throttle_duration_sec=5.0)
            self.cob.marcar_scan(xl, yl, yawl, ang, rangos, rmin, self.p['rango_cobertura'],
                                 self.grilla.pared)
        except Exception as e:
            self.get_logger().error(f'al_recibir_scan: {e}', throttle_duration_sec=2.0)

    def actualizar_candidatos(self):
        if self.det is None:
            return
        try:
            for c in self.reg.actualizar(self.det.clusters()):
                if any(math.hypot(c['x'] - rx, c['y'] - ry) <= self.p['radio_rojo']
                       for rx, ry in self.reportes_hechos):
                    c['estado'] = ROJO
                self.get_logger().info(f'Candidato nuevo {c["id"]} en ({c["x"]:.2f}, {c["y"]:.2f}), '
                                       f'tam {c["tam"]:.2f} m, {c["estado"]}.')
        except Exception as e:
            self.get_logger().error(f'actualizar_candidatos: {e}', throttle_duration_sec=2.0)

    def planificar(self, x, y):
        # esto es lo pesado, solo se llama al decidir
        obst = self.det.obstaculos_de_clusters()
        nav, dist_total = grilla_navegable(self.grilla, obst, self.p['radio_nav'])
        nav_giro = nav & (dist_total >= self.p['radio_giro'])
        nav_paso, _ = grilla_navegable(self.grilla, obst,
                                       min(self.p['radio_paso'], self.p['radio_nav']))
        fr, cr = self.grilla.mundo_a_celda(x, y)
        d = distancias_camino(nav_paso, fr, cr, self.grilla.res)
        return obst, nav, nav_giro, d

    def decidir_ahora(self, x, y):
        g, p = self.grilla, self.p
        obst, nav, nav_giro, d = self.planificar(x, y)
        zonas = self.zonas_vigentes()
        fl, cl = g.mundo_a_celda(self.largada[0], self.largada[1])
        camino_largada = float(d[fl, cl]) if g.dentro(fl, cl) and np.isfinite(d[fl, cl]) else \
            math.hypot(x - self.largada[0], y - self.largada[1]) * 1.5
        reserva = tiempo_reserva(camino_largada, p['velocidad_efectiva'], p['factor_reserva'],
                                 p['margen_reserva_s'])

        def pm(cand, excl=()):
            return punto_para_mirar(g, nav_giro, d, g.pared, cand['x'], cand['y'],
                                    angulos_excluidos=excl, zonas_prohibidas=zonas)
        cand, pv, sin_punto = elegir_inspeccion(self.reg.con_estado(PENDIENTE), pm)
        for s in sin_punto:
            s['intentos'] += 1
            if s['intentos'] >= 3:
                s['estado'] = INALCANZABLE
                self.get_logger().warn(f'Candidato {s["id"]}: sin lugar desde donde mirarlo.')
        insp = (cand, pv) if cand is not None else None
        cobp = None
        if insp is None:
            cobp = elegir_punto_de_vista(self.cob, nav, d, g.pared | obst, p['rango_cobertura'],
                                         p['paso_muestreo'], p['dist_min_destino'], p['costo_fijo_m'],
                                         p['ganancia_min'], zonas)
        reinsp = None
        if insp is None and cobp is None:
            otros = [c for c in self.reg.con_estado(NO_ROJO) if c['intentos'] < p['max_intentos_mirar']]
            c2, pv2, _ = elegir_inspeccion(otros, lambda c: pm(c, c['angulos_usados']))
            if c2 is not None:
                reinsp = (c2, pv2)
        accion, dato, motivo = decidir(self.t_corrida(), p['duracion_corrida_s'], self.reportes,
                                       self.n_victimas, reserva, insp, cobp, reinsp)
        self.get_logger().info(
            f'Decisión: {accion} ({motivo}). Cobertura {self.cob.fraccion_vista():.0%}, '
            f'candidatos {len(self.reg.lista)}, reportes {self.reportes}/{self.n_victimas}, '
            f'reserva {reserva:.0f}s.')
        return accion, dato

    def enviar_goal(self, x, y, yaw, tipo, **extra):
        goal = NavigateToPose.Goal()
        goal.pose.header.frame_id = self.p['map_frame']
        goal.pose.header.stamp = self.get_clock().now().to_msg()
        goal.pose.pose.position.x, goal.pose.pose.position.y = float(x), float(y)
        goal.pose.pose.orientation = yaw_a_quaternion(yaw)
        self.sig_id += 1
        self.activo = dict(id=self.sig_id, tipo=tipo, x=x, y=y, yaw=yaw, gh=None,
                           t=self.ahora_ns(), resultado=None, dist_restante=None,
                           mejor_dist=math.inf, t_mejora=self.ahora_ns(), **extra)
        self.t_chequeo_meta = self.ahora_ns()
        self.get_logger().info(f'Meta #{self.sig_id} ({tipo}) -> ({x:.2f}, {y:.2f})')
        fut = self.cliente.send_goal_async(
            goal, feedback_callback=partial(self.feedback_goal, self.sig_id))
        fut.add_done_callback(partial(self.respuesta_goal, self.sig_id))

    def feedback_goal(self, goal_id, msg):
        if self.activo is not None and self.activo['id'] == goal_id:
            d = float(msg.feedback.distance_remaining)
            if math.isfinite(d) and d > 0.0:
                self.activo['dist_restante'] = d

    def respuesta_goal(self, goal_id, fut):
        try:
            gh = fut.result()
            vigente = self.activo is not None and self.activo['id'] == goal_id
            if not gh.accepted:
                if vigente:
                    self.activo['resultado'] = 'rechazada'
                return
            if not vigente:
                gh.cancel_goal_async()
                return
            self.activo['gh'] = gh
            gh.get_result_async().add_done_callback(partial(self.resultado_goal, goal_id))
        except Exception as e:
            self.get_logger().error(f'respuesta_goal: {e}')

    def resultado_goal(self, goal_id, fut):
        try:
            if self.activo is None or self.activo['id'] != goal_id:
                return
            ok = fut.result().status == GoalStatus.STATUS_SUCCEEDED
            self.activo['resultado'] = 'ok' if ok else 'fallo'
        except Exception as e:
            self.get_logger().error(f'resultado_goal: {e}')

    def cancelar_activo(self):
        if self.activo is not None:
            gh = self.activo.get('gh')
            self.activo = None
            if gh is not None:
                gh.cancel_goal_async()

    def limpiar_costmaps(self):
        for cli in (self.limpiar_local, self.limpiar_global):
            if cli.service_is_ready():
                cli.call_async(ClearEntireCostmap.Request())

    def ciclo(self):
        try:
            if self.grilla is None or self.estado == TERMINADO:
                return
            pose = self.pose_robot()
            if pose is None:
                return
            x, y, yaw = pose

            if self.estado == ESPERANDO:
                if self.cliente.server_is_ready():
                    self.cambiar(ASENTANDO)
                return
            if self.estado == ASENTANDO:
                if self.segundos_desde(self.t_estado) >= self.p['tiempo_asentamiento_s']:
                    self.largada = (x, y, yaw)
                    self.get_logger().info(f'Largada guardada en ({x:.2f}, {y:.2f}).')
                    self.cambiar(DECIDIR)
                return

            limite = self.p['duracion_corrida_s'] - self.p['margen_final_s']
            if self.t_corrida() >= limite and self.estado != FINALIZANDO:
                self.get_logger().warn('Se acaba el tiempo: cierro la corrida.')
                self.cancelar_activo()
                self.frenar()
                self.cambiar(FINALIZANDO)

            {
                DECIDIR: self.paso_decidir,
                NAVEGANDO: self.paso_navegando,
                ORIENTANDO: self.paso_orientando,
                MIRANDO: self.paso_mirando,
                ESCAPANDO: self.paso_escapando,
                REGRESANDO: self.paso_regresando,
                AJUSTE_FINAL: self.paso_ajuste_final,
                FINALIZANDO: self.paso_finalizando,
            }[self.estado](x, y, yaw)
        except Exception as e:
            self.get_logger().error(f'ciclo ({self.estado}): {e}', throttle_duration_sec=2.0)

    def paso_decidir(self, x, y, yaw):
        accion, dato = self.decidir_ahora(x, y)
        if accion == REGRESAR:
            self.intentos_regreso = 0
            self.mejor_camino_regreso = None
            if self.p['volver_a_la_largada']:
                self.cambiar(REGRESANDO)
            else:
                self.cambiar(FINALIZANDO)
            return
        destino = dato[1] if accion == INSPECCIONAR else dato
        if self.tramo_proximo:
            tramo = self.punto_de_tramo_hacia(x, y, destino['x'], destino['y'],
                                              self.p['tramo_exploracion_m'])
            if tramo is not None and math.hypot(tramo[0] - x, tramo[1] - y) >= self.p['dist_min_destino'] \
                    and math.hypot(tramo[0] - destino['x'], tramo[1] - destino['y']) > 0.3:
                tx, ty = tramo
                self.inspeccion = None
                self.get_logger().info(f'Por tramos: punto intermedio ({tx:.2f}, {ty:.2f}).')
                self.enviar_goal(tx, ty, math.atan2(ty - y, tx - x), 'tramo',
                                 camino=self.p['tramo_exploracion_m'] * 1.5)
                self.cambiar(NAVEGANDO)
                return
        if accion == INSPECCIONAR:
            cand, pv = dato
            self.inspeccion = dict(cand=cand, pv=pv)
            self.enviar_goal(pv['x'], pv['y'], pv['yaw'], 'inspeccion', cand_id=cand['id'],
                             camino=pv['camino'])
        else:
            self.inspeccion = None
            self.n_pendientes_al_salir = len(self.reg.con_estado(PENDIENTE))
            self.enviar_goal(dato['x'], dato['y'], math.atan2(dato['y'] - y, dato['x'] - x),
                             'cobertura', fila=dato['fila'], col=dato['col'], camino=dato['camino'])
        self.cambiar(NAVEGANDO)

    def vigilar_progreso(self, x, y):
        # trabado = no se acerco a la meta en un rato (ir y venir no cuenta)
        act = self.activo
        d = act['dist_restante']
        if d is None:
            d = math.hypot(act['x'] - x, act['y'] - y)
        if d > act['mejor_dist'] + 0.5:
            act['mejor_dist'], act['t_mejora'] = d, self.ahora_ns()
            return False
        if d < act['mejor_dist'] - self.p['progreso_min_m']:
            act['mejor_dist'], act['t_mejora'] = d, self.ahora_ns()
            return False
        return self.segundos_desde(act['t_mejora']) > self.p['tiempo_sin_progreso_s']

    def paso_navegando(self, x, y, yaw):
        act = self.activo
        if act is None:
            self.cambiar(DECIDIR)
            return
        if act['resultado'] == 'ok':
            self.activo = None
            self.tramo_proximo = False
            if act['tipo'] == 'inspeccion':
                self.cambiar(ORIENTANDO)
            else:
                self.cambiar(DECIDIR)
            return
        if act['resultado'] in ('fallo', 'rechazada'):
            self.get_logger().warn(f'Meta #{act["id"]} {act["resultado"]}.')
            self.activo = None
            self.tramo_proximo = act['tipo'] != 'tramo'
            self.registrar_fallo(act)
            self.cambiar(DECIDIR)
            return
        if self.reportes >= self.n_victimas:
            self.cancelar_activo()
            self.cambiar(DECIDIR)
            return
        if act['tipo'] == 'inspeccion':
            if self.inspeccion['cand']['estado'] == ROJO:
                self.get_logger().info('El candidato se confirmó rojo en el camino.')
                self.cancelar_activo()
                self.cambiar(DECIDIR)
                return
        elif act['tipo'] == 'cobertura':
            if len(self.reg.con_estado(PENDIENTE)) > self.n_pendientes_al_salir:
                self.get_logger().info('Apareció un candidato nuevo: voy a mirarlo.')
                self.cancelar_activo()
                self.cambiar(DECIDIR)
                return
            if self.segundos_desde(self.t_chequeo_meta) > 2.0:
                self.t_chequeo_meta = self.ahora_ns()
                obst = self.det.obstaculos_de_clusters()
                gan = self.cob.ganancia(act['fila'], act['col'], self.grilla.pared | obst,
                                        self.p['rango_cobertura'])
                if gan < self.p['ganancia_min']:
                    self.get_logger().info('Lo que había para ver en la meta ya se vio.')
                    self.cancelar_activo()
                    self.cambiar(DECIDIR)
                    return
        limite = max(20.0, 1.5 * act['camino'] / self.p['velocidad_efectiva'] + 10.0)
        if self.vigilar_progreso(x, y) or self.segundos_desde(act['t']) > limite:
            self.get_logger().warn(f'Meta #{act["id"]}: sin avance o fuera de tiempo -> escape.')
            self.tramo_proximo = act['tipo'] != 'tramo'
            self.registrar_fallo(act)
            self.cancelar_activo()
            self.iniciar_escape(siguiente=DECIDIR)

    def registrar_fallo(self, act):
        self.prohibir_zona(act['x'], act['y'])
        if act['tipo'] == 'inspeccion' and self.inspeccion is not None:
            cand = self.inspeccion['cand']
            cand['intentos'] += 1
            cand['angulos_usados'].append(self.inspeccion['pv']['angulo'])
            if cand['intentos'] >= 3 and cand['estado'] == PENDIENTE:
                cand['estado'] = INALCANZABLE

    def giro_seguro(self):
        # no girar si algo queda adentro del circulo que barre el chasis
        if self.ultimo_scan is None or self.laser_en_base is None:
            return False
        msg = self.ultimo_scan[0]
        lx, ly, lyaw = self.laser_en_base
        r = np.array(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment + lyaw
        ok = np.isfinite(r) & (r >= max(msg.range_min, 0.05))
        px, py = lx + r[ok] * np.cos(a[ok]), ly + r[ok] * np.sin(a[ok])
        radio = math.hypot(self.p['footprint_atras'], self.p['footprint_costado']) + 0.02
        return bool(np.all(np.hypot(px, py) > radio))

    def paso_orientando(self, x, y, yaw):
        objetivo = self.inspeccion['pv']['yaw']
        err = normalizar_angulo(objetivo - yaw)
        vencido = self.segundos_desde(self.t_estado) > self.p['tiempo_max_orientar_s']
        if abs(err) < math.radians(self.p['tolerancia_orientar_grados']) or vencido or not self.giro_seguro():
            self.frenar()
            self.cambiar(MIRANDO)
            return
        w = float(np.clip(1.5 * err, -self.p['giro_orientar'], self.p['giro_orientar']))
        if abs(w) < 0.1:
            w = math.copysign(0.1, err)
        cmd = Twist()
        cmd.angular.z = w
        self.pub_cmd.publish(cmd)

    def paso_mirando(self, x, y, yaw):
        cand = self.inspeccion['cand']
        if cand['estado'] == ROJO:
            self.cambiar(DECIDIR)
            return
        if self.segundos_desde(self.t_estado) < self.p['tiempo_mirar_s']:
            return
        cand['intentos'] += 1
        cand['angulos_usados'].append(self.inspeccion['pv']['angulo'])
        cand['estado'] = NO_ROJO
        self.get_logger().info(f'Candidato {cand["id"]}: no es rojo (intento {cand["intentos"]}).')
        self.cambiar(DECIDIR)

    def iniciar_escape(self, siguiente):
        self.frenar()
        self.escape = dict(siguiente=siguiente, direccion=None, inicio=None)
        self.cambiar(ESCAPANDO)

    def angulos_scan_en_base(self):
        msg = self.ultimo_scan[0]
        r = np.array(msg.ranges, dtype=float)
        a = msg.angle_min + np.arange(len(r)) * msg.angle_increment + self.laser_en_base[2]
        return r, a, max(msg.range_min, 0.05)

    def paso_escapando(self, x, y, yaw):
        e = self.escape
        if self.segundos_desde(self.t_estado) < 0.4:
            self.frenar()
            return
        if self.ultimo_scan is None or self.laser_en_base is None:
            self.terminar_escape()
            return
        r, a, rmin = self.angulos_scan_en_base()
        esquinas = self.esquinas_desde_laser()
        if e['direccion'] is None:
            sal = direccion_escape(r, a, esquinas, rmin, despeje_min=self.p['escape_despeje_min'])
            if sal is None:
                self.get_logger().warn('Escape: no hay lado libre; limpio costmaps y sigo.')
                self.terminar_escape()
                return
            e['direccion'], e['inicio'] = sal[0], (x, y)
            self.get_logger().info(f'Escape hacia {math.degrees(sal[0]):.0f}° (despeje {sal[1]:.2f} m).')
        movido = math.hypot(x - e['inicio'][0], y - e['inicio'][1])
        despeje = despeje_en_direccion(r, a, esquinas, rmin, e['direccion'])
        vencido = self.segundos_desde(self.t_estado) > self.p['escape_tiempo_max_s'] + 0.4
        if movido >= self.p['escape_distancia'] or despeje < 0.05 or vencido:
            self.terminar_escape()
            return
        cmd = Twist()
        cmd.linear.x = self.p['escape_velocidad'] * math.cos(e['direccion'])
        cmd.linear.y = self.p['escape_velocidad'] * math.sin(e['direccion'])
        self.pub_cmd.publish(cmd)

    def terminar_escape(self):
        self.frenar()
        self.limpiar_costmaps()
        siguiente = self.escape['siguiente'] if self.escape else DECIDIR
        self.escape = None
        self.cambiar(siguiente)

    def camino_a_largada(self, x, y):
        g = self.grilla
        _, _, _, d = self.planificar(x, y)
        fl, cl = g.mundo_a_celda(self.largada[0], self.largada[1])
        if g.dentro(fl, cl) and np.isfinite(d[fl, cl]):
            return float(d[fl, cl])
        return math.hypot(x - self.largada[0], y - self.largada[1]) * 1.5

    def punto_de_tramo_hacia(self, x, y, dx, dy, largo):
        # punto intermedio a `largo` m por el camino, para no mandar metas lejos
        g = self.grilla
        obst = self.det.obstaculos_de_clusters()
        fd, cd = g.mundo_a_celda(dx, dy)
        fr, cr = g.mundo_a_celda(x, y)
        for radio in (self.p['radio_giro'], self.p['radio_nav']):
            nav, _ = grilla_navegable(g, obst, radio)
            d_destino = distancias_camino(nav, fd, cd, g.res)
            celda = punto_intermedio(d_destino, nav, fr, cr, largo)
            if celda is not None:
                return g.celda_a_mundo(*celda)
        return None

    def punto_de_tramo(self, x, y):
        return self.punto_de_tramo_hacia(x, y, self.largada[0], self.largada[1],
                                         self.p['tramo_regreso_m'])

    def paso_regresando(self, x, y, yaw):
        gx, gy, gyaw = self.largada
        act = self.activo
        if act is None:
            if math.hypot(gx - x, gy - y) <= 0.15:
                self.cambiar(AJUSTE_FINAL)
                return
            camino = self.camino_a_largada(x, y)
            if self.mejor_camino_regreso is None or \
                    camino < self.mejor_camino_regreso - self.p['progreso_regreso_m']:
                if self.mejor_camino_regreso is not None:
                    self.get_logger().info(f'Regreso: me acerqué ({camino:.1f} m de camino), '
                                           f'renuevo los intentos.')
                self.mejor_camino_regreso = camino
                self.intentos_regreso = 0
            if self.intentos_regreso >= self.p['intentos_regreso']:
                self.get_logger().warn('No pude volver a la largada; cierro igual.')
                self.cambiar(FINALIZANDO)
                return
            usar_tramo = self.intentos_regreso > 0
            self.intentos_regreso += 1
            if usar_tramo:
                tramo = self.punto_de_tramo(x, y)
                if tramo is not None and math.hypot(tramo[0] - x, tramo[1] - y) >= self.p['dist_min_destino']:
                    tx, ty = tramo
                    self.get_logger().info(f'Regreso por tramos: punto intermedio ({tx:.2f}, {ty:.2f}).')
                    self.enviar_goal(tx, ty, math.atan2(ty - y, tx - x), 'regreso', tramo=True,
                                     camino=self.p['tramo_regreso_m'] * 1.5)
                    return
            self.enviar_goal(gx, gy, gyaw, 'regreso', camino=math.hypot(gx - x, gy - y) * 1.5)
            return
        if act['resultado'] == 'ok':
            self.activo = None
            if not act.get('tramo'):
                self.cambiar(AJUSTE_FINAL)
            return
        if act['resultado'] in ('fallo', 'rechazada'):
            self.activo = None
            self.iniciar_escape(siguiente=REGRESANDO)
            return
        if self.vigilar_progreso(x, y):
            self.cancelar_activo()
            self.iniciar_escape(siguiente=REGRESANDO)

    def paso_ajuste_final(self, x, y, yaw):
        # ultimos cm a mano, nav2 deja a 15cm y el bono pide 25
        gx, gy, _ = self.largada
        ex, ey = gx - x, gy - y
        dist = math.hypot(ex, ey)
        vencido = self.segundos_desde(self.t_estado) > self.p['tiempo_ajuste_final_s']
        if dist <= self.p['tolerancia_ajuste_final'] or vencido:
            self.frenar()
            self.cambiar(FINALIZANDO)
            return
        bx = math.cos(yaw) * ex + math.sin(yaw) * ey
        by = -math.sin(yaw) * ex + math.cos(yaw) * ey
        direccion = math.atan2(by, bx)
        if self.ultimo_scan is not None and self.laser_en_base is not None:
            r, a, rmin = self.angulos_scan_en_base()
            if despeje_en_direccion(r, a, self.esquinas_desde_laser(), rmin, direccion) < 0.05:
                self.frenar()
                self.cambiar(FINALIZANDO)
                return
        v = min(self.p['velocidad_ajuste_final'], max(0.03, 0.8 * dist))
        cmd = Twist()
        cmd.linear.x = v * bx / dist
        cmd.linear.y = v * by / dist
        self.pub_cmd.publish(cmd)

    def paso_finalizando(self, x, y, yaw):
        if self.t_finalizando is None:
            self.frenar()
            self.t_finalizando = self.ahora_ns()
            sug = self.reg.sugerencias_para_relleno()
            for sx, sy in sug:
                m = PointStamped()
                m.header.frame_id = 'map'
                m.header.stamp = self.get_clock().now().to_msg()
                m.point.x, m.point.y = float(sx), float(sy)
                self.pub_sugerencias.publish(m)
            self.get_logger().info(f'Le paso {len(sug)} sugerencias al gestor.')
            self.relleno_pedido = False
            return
        if not self.relleno_pedido:
            if self.segundos_desde(self.t_finalizando) < 0.5:
                return
            self.relleno_pedido = True
            self.get_logger().info('Pido relleno.')
            self.pub_rellenar.publish(Empty())
            return
        if self.segundos_desde(self.t_finalizando) < 2.0:
            return
        if self.p['publicar_done']:
            self.pub_done.publish(Empty())
            self.get_logger().info(f'/done publicado (t={self.t_corrida():.0f}s).')
        self.cambiar(TERMINADO)


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
