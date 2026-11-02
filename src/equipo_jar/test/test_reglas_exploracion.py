import math

import numpy as np

from equipo_jar.reglas_anomalias import (NO_ROJO, PENDIENTE, ROJO, DetectorAnomalias,
                                         RegistroCandidatos, puntos_en_mapa)
from equipo_jar.reglas_cobertura import Cobertura, elegir_punto_de_vista, punto_para_mirar
from equipo_jar.reglas_escape import (direccion_escape, soporte_footprint,
                                      tiempo_reserva)
from equipo_jar.reglas_mapa import (Grilla, celdas_vistas, distancias_camino,
                                    grilla_navegable, linea_libre)

RES = 0.05


def mundo():
    alto, ancho = 40, 80                       # 2 m x 4 m
    datos = np.zeros((alto, ancho), dtype=np.int16)
    datos[0, :] = datos[-1, :] = 100
    datos[:, 0] = datos[:, -1] = 100
    grilla = Grilla(datos, RES, 0.0, 0.0)
    real = grilla.pared.copy()
    f, c = grilla.mundo_a_celdas(np.array([1.9, 2.1]), np.array([0.9, 1.1]))
    real[f[0]:f[1] + 1, c[0]:c[1] + 1] = True
    return grilla, real


def scan_sintetico(real, x, y, yaw, n=1080, rmax=12.0):
    angulos = -math.pi + np.arange(n) * 2 * math.pi / n
    rangos = np.full(n, np.inf)
    for i, a in enumerate(angulos):
        for s in np.arange(0.0, rmax, RES / 4):
            px, py = x + s * math.cos(yaw + a), y + s * math.sin(yaw + a)
            f, c = int(math.floor(py / RES)), int(math.floor(px / RES))
            if not (0 <= f < real.shape[0] and 0 <= c < real.shape[1]) or real[f, c]:
                rangos[i] = s
                break
    return angulos, rangos


def test_distancia_pared_y_navegable():
    g, _ = mundo()
    assert g.dist_pared[20, 40] > 0.9                     # centro del cuarto
    nav, _ = grilla_navegable(g, np.zeros_like(g.pared), radio_nav=0.16)
    assert nav[20, 40] and not nav[2, 40]


def test_dijkstra_rodea_obstaculo():
    g, real = mundo()
    extra = real & ~g.pared
    nav, _ = grilla_navegable(g, extra, radio_nav=0.16)
    d = distancias_camino(nav, 20, 10, RES)               # desde (0.5, 1.0)
    recta = 3.0
    assert np.isfinite(d[20, 70]) and d[20, 70] > recta   # tiene que rodear la caja


def test_linea_libre_y_rayos():
    g, real = mundo()
    assert linea_libre(real, 20, 10, 20, 30)
    assert not linea_libre(real, 20, 10, 20, 70)          # la caja tapa
    f, c = celdas_vistas(real, 20, 10, [0.0], [3.5], RES)
    assert c.max() < 38


def test_detecta_la_caja_y_no_las_paredes():
    g, real = mundo()
    det = DetectorAnomalias(g)
    for x in (0.6, 0.8, 1.0):
        ang, r = scan_sintetico(real, x, 1.0, 0.0)
        xs, ys, val = puntos_en_mapa(r, ang, x, 1.0, 0.0, 0.05, 12.0)
        assert det.salud(xs, ys, val) > 0.8
        det.agregar_scan(xs, ys, val)
    cl = det.clusters()
    assert len(cl) == 1
    assert math.hypot(cl[0]['x'] - 1.9, cl[0]['y'] - 1.0) < 0.10   # cara visible de la caja
    assert 0.1 < cl[0]['tam'] < 0.4


def test_salud_baja_si_la_pose_esta_corrida():
    g, real = mundo()
    det = DetectorAnomalias(g)
    ang, r = scan_sintetico(real, 1.0, 1.0, 0.0)
    xs, ys, val = puntos_en_mapa(r, ang, 1.0, 1.0, 0.25, 0.05, 12.0)   # yaw errado 0.25 rad
    assert det.salud(xs, ys, val) < 0.6


def test_registro_mantiene_identidad_y_marca_rojo():
    reg = RegistroCandidatos()
    nuevos = reg.actualizar([dict(x=1.0, y=1.0, tam=0.25, votos=10, celdas=5)])
    assert len(nuevos) == 1 and nuevos[0]['estado'] == PENDIENTE
    assert reg.actualizar([dict(x=1.05, y=1.0, tam=0.25, votos=20, celdas=6)]) == []
    assert reg.marcar_rojo_cerca(1.1, 1.0, radio=0.4)['estado'] == ROJO
    reg.actualizar([dict(x=3.0, y=1.0, tam=0.30, votos=8, celdas=5)])
    reg.lista[1]['estado'] = NO_ROJO
    assert reg.sugerencias_para_relleno() == [(3.0, 1.0)]


def test_cobertura_avanza_y_termina():
    g, real = mundo()
    cob = Cobertura(g)
    ang, r = scan_sintetico(real, 0.5, 1.0, 0.0)
    cob.marcar_scan(0.5, 1.0, 0.0, ang, r, 0.05, 3.0, g.pared)
    parcial = cob.fraccion_vista()
    assert 0.2 < parcial < 1.0                             # la caja tapa lo de atras
    extra = real & ~g.pared
    nav, _ = grilla_navegable(g, extra, 0.16)
    bloq = g.pared | extra
    d = distancias_camino(nav, *g.mundo_a_celda(0.5, 1.0), RES)
    pv = elegir_punto_de_vista(cob, nav, d, bloq, 3.0)
    assert pv is not None and pv['x'] > 1.5
    ang, r = scan_sintetico(real, pv['x'], pv['y'], 0.0)
    cob.marcar_scan(pv['x'], pv['y'], 0.0, ang, r, 0.05, 3.0, g.pared)
    assert cob.fraccion_vista() > parcial


def test_punto_para_mirar_de_frente():
    g, real = mundo()
    extra = real & ~g.pared
    nav, _ = grilla_navegable(g, extra, 0.16)
    d = distancias_camino(nav, *g.mundo_a_celda(0.5, 1.0), RES)
    pm = punto_para_mirar(g, nav, d, g.pared, 2.0, 1.0)
    assert pm is not None
    dist = math.hypot(pm['x'] - 2.0, pm['y'] - 1.0)
    assert 0.55 < dist < 0.95
    assert abs(math.atan2(1.0 - pm['y'], 2.0 - pm['x']) - pm['yaw']) < 1e-6
    otro = punto_para_mirar(g, nav, d, g.pared, 2.0, 1.0, angulos_excluidos=[pm['angulo']])
    assert otro is not None and abs(otro['angulo'] - pm['angulo']) >= math.radians(60) - 1e-6


ESQUINAS = [(0.08, 0.11), (0.08, -0.11), (-0.23, -0.11), (-0.23, 0.11)]   # chasis desde el lidar


def test_soporte_footprint():
    assert abs(soporte_footprint(ESQUINAS, 0.0) - 0.08) < 1e-9          # adelante
    assert abs(soporte_footprint(ESQUINAS, math.pi) - 0.23) < 1e-9      # atras
    assert abs(soporte_footprint(ESQUINAS, math.pi / 2) - 0.11) < 1e-9  # costado


def test_escape_elige_el_lado_libre():
    ang = -math.pi + np.arange(360) * 2 * math.pi / 360
    r = np.full(360, 2.0)
    r[np.abs(ang) < math.radians(100)] = 0.12          # pared pegada adelante
    sal = direccion_escape(r, ang, ESQUINAS, 0.05)
    assert sal is not None and abs(abs(sal[0]) - math.pi) < math.radians(50)   # hacia atras


def test_escape_sin_salida():
    ang = -math.pi + np.arange(360) * 2 * math.pi / 360
    r = np.full(360, 0.20)                              # rodeado a 20 cm
    assert direccion_escape(r, ang, ESQUINAS, 0.05) is None


def test_reserva():
    assert abs(tiempo_reserva(5.0) - 80.0) < 1e-6


def test_error_de_yaw_no_inventa_objetos():
    g, _ = mundo()
    vacio = g.pared.copy()
    det = DetectorAnomalias(g)
    for k, x in enumerate((0.6, 1.2, 1.8, 2.4, 3.0, 3.4)):
        err = math.radians(3.0) * (1 if k % 2 else -1)
        ang, r = scan_sintetico(vacio, x, 1.0, 0.0)
        xs, ys, val = puntos_en_mapa(r, ang, x, 1.0, err, 0.05, 12.0)
        det.agregar_scan(xs, ys, val, x, 1.0, r, ang, err)
    assert det.clusters() == []


def test_punto_intermedio_avanza_por_el_camino_hacia_la_meta():
    from equipo_jar.reglas_mapa import punto_intermedio
    grilla, _ = mundo()
    nav, _ = grilla_navegable(grilla, np.zeros_like(grilla.pared), 0.16)
    fm, cm = grilla.mundo_a_celda(0.5, 1.0)                      # meta (la largada)
    d_meta = distancias_camino(nav, fm, cm, RES)
    fr, cr = grilla.mundo_a_celda(3.5, 1.0)                      # robot, 3 m de la meta
    f, c = punto_intermedio(d_meta, nav, fr, cr, 1.2)
    x, _ = grilla.celda_a_mundo(f, c)
    assert abs(x - 2.3) < 0.1                                    # 1.2 m mas cerca
    f, c = punto_intermedio(d_meta, nav, fr, cr, 10.0)
    assert (f, c) == (fm, cm)
