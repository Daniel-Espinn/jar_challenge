# cobertura: que piso ya barrio el lidar y a donde ir a mirar
import math

import numpy as np

from equipo_jar.reglas_mapa import celdas_vistas, linea_libre


def normalizar_angulo(a):
    return math.atan2(math.sin(a), math.cos(a))


class Cobertura:

    def __init__(self, grilla, dist_min_pared=0.10):
        self.g = grilla
        self.objetivo = grilla.libre & (grilla.dist_pared >= dist_min_pared)
        self.visto = np.zeros_like(self.objetivo)
        self.total = int(self.objetivo.sum())

    def marcar_scan(self, x_laser, y_laser, yaw_laser, angulos, rangos, rango_min,
                    rango_cobertura, bloqueado, submuestreo=3):
        f0, c0 = self.g.mundo_a_celda(x_laser, y_laser)
        if not self.g.dentro(f0, c0):
            return 0
        rangos = np.asarray(rangos, dtype=float)[::submuestreo]
        angulos = np.asarray(angulos, dtype=float)[::submuestreo]
        rebote = np.isfinite(rangos) & (rangos < rango_min)
        alcance = np.where(np.isfinite(rangos), np.minimum(rangos, rango_cobertura), rango_cobertura)
        alcance, ang = alcance[~rebote], angulos[~rebote] + yaw_laser
        filas, cols = celdas_vistas(bloqueado, f0, c0, ang, alcance, self.g.res)
        nuevas = self.objetivo[filas, cols] & ~self.visto[filas, cols]
        self.visto[filas, cols] = True
        return int(nuevas.sum())

    def pendientes(self):
        return self.objetivo & ~self.visto

    def fraccion_vista(self):
        return float((self.objetivo & self.visto).sum()) / max(self.total, 1)

    def ganancia(self, fila, col, bloqueado, rango_cobertura, n_rayos=72):
        ang = np.linspace(-math.pi, math.pi, n_rayos, endpoint=False)
        f, c = celdas_vistas(bloqueado, fila, col, ang, np.full(n_rayos, rango_cobertura), self.g.res)
        return int((self.objetivo[f, c] & ~self.visto[f, c]).sum())


def elegir_punto_de_vista(cobertura, navegable, dist_camino, bloqueado, rango_cobertura,
                          paso_muestreo=5, dist_min_destino=0.5, costo_fijo_m=1.0,
                          ganancia_min=25, zonas_prohibidas=()):
    # el que mas piso nuevo ve por metro de camino
    g = cobertura.g
    filas, cols = np.nonzero(navegable & np.isfinite(dist_camino))
    sel = (filas % paso_muestreo == 0) & (cols % paso_muestreo == 0)
    filas, cols = filas[sel], cols[sel]
    mejor = None
    for f, c in zip(filas, cols):
        camino = float(dist_camino[f, c])
        if camino < dist_min_destino:
            continue
        x, y = g.celda_a_mundo(f, c)
        if any(math.hypot(x - zx, y - zy) < zr for zx, zy, zr in zonas_prohibidas):
            continue
        gan = cobertura.ganancia(f, c, bloqueado, rango_cobertura)
        if gan < ganancia_min:
            continue
        puntaje = gan / (camino + costo_fijo_m)
        if mejor is None or puntaje > mejor['puntaje']:
            mejor = dict(x=x, y=y, fila=int(f), col=int(c), ganancia=gan, camino=camino,
                         puntaje=puntaje)
    return mejor


def punto_para_mirar(grilla, navegable, dist_camino, bloqueado_paredes, cx, cy,
                     radios=(0.60, 0.75, 0.90), n_angulos=24, angulos_excluidos=(),
                     sep_angulo=math.radians(60), zonas_prohibidas=()):
    # desde donde mirar el candidato con la camara (0.6-0.9 m y vista libre)
    fc, cc = grilla.mundo_a_celda(cx, cy)
    mejor = None
    for r in radios:
        for k in range(n_angulos):
            ang = -math.pi + k * 2 * math.pi / n_angulos
            if any(abs(normalizar_angulo(ang - a)) < sep_angulo for a in angulos_excluidos):
                continue
            px, py = cx + r * math.cos(ang), cy + r * math.sin(ang)
            f, c = grilla.mundo_a_celda(px, py)
            if not grilla.dentro(f, c) or not navegable[f, c] or not np.isfinite(dist_camino[f, c]):
                continue
            if any(math.hypot(px - zx, py - zy) < zr for zx, zy, zr in zonas_prohibidas):
                continue
            if not linea_libre(bloqueado_paredes, f, c, fc, cc):
                continue
            costo = float(dist_camino[f, c]) + 0.5 * abs(r - 0.75)
            if mejor is None or costo < mejor['costo']:
                mejor = dict(x=px, y=py, yaw=math.atan2(cy - py, cx - px), angulo=ang,
                             camino=float(dist_camino[f, c]), costo=costo)
    return mejor
