# anomalias del lidar: puntos lejos de las paredes del mapa = posibles cajas
import math

import numpy as np
import scipy.ndimage as nd

from equipo_jar.reglas_mapa import celdas_vistas

PENDIENTE = 'pendiente'
ROJO = 'rojo'                    # ya reportado
NO_ROJO = 'no_rojo'
INALCANZABLE = 'inalcanzable'    # no hay desde donde mirarla


def puntos_en_mapa(rangos, angulos, x_laser, y_laser, yaw_laser, rango_min, rango_max):
    rangos = np.asarray(rangos, dtype=float)
    validos = np.isfinite(rangos) & (rangos >= rango_min) & (rangos <= rango_max)
    r = np.where(validos, rangos, 0.0)
    a = yaw_laser + np.asarray(angulos, dtype=float)
    return x_laser + r * np.cos(a), y_laser + r * np.sin(a), validos


class DetectorAnomalias:

    def __init__(self, grilla, tau=0.10, tau_por_metro=0.03, rango_max_votos=2.5,
                 distancia_salud=0.10, min_votos=3, razon_min=0.5,
                 tam_min=0.06, tam_max=0.50):
        self.g = grilla
        self.tau = tau
        self.tau_por_metro = tau_por_metro
        self.rango_max_votos = rango_max_votos
        self.distancia_salud = distancia_salud
        self.min_votos = min_votos
        self.razon_min = razon_min
        self.tam_min = tam_min
        self.tam_max = tam_max
        self.votos = np.zeros((grilla.alto, grilla.ancho), dtype=np.int32)
        self.pasadas = np.zeros((grilla.alto, grilla.ancho), dtype=np.int32)

    def salud(self, xs, ys, validos):
        # si el scan no calza con las paredes la pose esta mal y no votamos
        f, c = self.g.mundo_a_celdas(xs[validos], ys[validos])
        ok = self.g.dentro_v(f, c)
        if ok.sum() < 20:
            return 0.0
        return float(np.mean(self.g.dist_pared[f[ok], c[ok]] <= self.distancia_salud))

    def agregar_scan(self, xs, ys, validos, x_laser=None, y_laser=None, rangos=None,
                     angulos=None, yaw_laser=None):
        validos = np.asarray(validos, dtype=bool)
        if x_laser is not None:
            r_punto = np.hypot(xs - x_laser, ys - y_laser)
        else:
            r_punto = np.zeros_like(xs)
        usar = validos & (r_punto <= self.rango_max_votos)
        f, c = self.g.mundo_a_celdas(xs[usar], ys[usar])
        r_punto = r_punto[usar]
        ok = self.g.dentro_v(f, c)
        f, c, r_punto = f[ok], c[ok], r_punto[ok]
        tol = self.tau + self.tau_por_metro * r_punto
        lejos = self.g.libre[f, c] & (self.g.dist_pared[f, c] > tol)
        claves = np.unique(f[lejos] * self.g.ancho + c[lejos])
        self.votos.flat[claves] += 1
        if rangos is not None:
            self._sumar_pasadas(x_laser, y_laser, yaw_laser, rangos, angulos, claves)
        return int(claves.size)

    def _sumar_pasadas(self, x_laser, y_laser, yaw_laser, rangos, angulos, claves_impacto):
        rangos = np.asarray(rangos, dtype=float)
        alcance = np.where(np.isfinite(rangos), rangos - 1.5 * self.g.res, self.rango_max_votos)
        alcance = np.clip(alcance, 0.0, self.rango_max_votos)
        ang = np.asarray(angulos, dtype=float)[::2] + yaw_laser
        alcance = alcance[::2]
        f0, c0 = self.g.mundo_a_celda(x_laser, y_laser)
        if not self.g.dentro(f0, c0):
            return
        vacio = np.zeros_like(self.g.pared)
        filas, cols = celdas_vistas(vacio, f0, c0, ang, alcance, self.g.res)
        claves = np.setdiff1d(filas * self.g.ancho + cols, claves_impacto)
        self.pasadas.flat[claves] += 1

    def ocupadas(self):
        total = self.votos + self.pasadas
        razon = np.where(total > 0, self.votos / np.maximum(total, 1), 0.0)
        return (self.votos >= self.min_votos) & (razon >= self.razon_min)

    def clusters(self):
        mascara = self.ocupadas()
        if not mascara.any():
            return []
        unida = nd.binary_dilation(mascara, structure=np.ones((3, 3), bool))
        etiquetas, n = nd.label(unida, structure=np.ones((3, 3), bool))
        salida = []
        for k in range(1, n + 1):
            filas, cols = np.nonzero((etiquetas == k) & mascara)
            if filas.size == 0:
                continue
            w = self.votos[filas, cols].astype(float)
            xs = self.g.ox + (cols + 0.5) * self.g.res
            ys = self.g.oy + (filas + 0.5) * self.g.res
            tam = math.hypot(xs.max() - xs.min(), ys.max() - ys.min()) + self.g.res
            if not (self.tam_min <= tam <= self.tam_max):
                continue
            salida.append(dict(x=float(np.average(xs, weights=w)),
                               y=float(np.average(ys, weights=w)),
                               tam=float(tam), votos=int(w.sum()), celdas=int(filas.size)))
        return salida

    def obstaculos_de_clusters(self):
        mascara = self.ocupadas()
        if not mascara.any():
            return mascara
        unida = nd.binary_dilation(mascara, structure=np.ones((3, 3), bool))
        etiquetas, n = nd.label(unida, structure=np.ones((3, 3), bool))
        salida = np.zeros_like(mascara)
        for k in range(1, n + 1):
            sel = (etiquetas == k) & mascara
            filas, cols = np.nonzero(sel)
            tam = math.hypot(np.ptp(cols), np.ptp(filas)) * self.g.res + self.g.res
            if self.tam_min <= tam <= self.tam_max:
                salida |= sel
        return salida


class RegistroCandidatos:
    # le da un id fijo a cada cluster entre actualizaciones

    def __init__(self, radio_asociacion=0.35):
        self.radio = radio_asociacion
        self.lista = []
        self._sig = 0

    def actualizar(self, clusters):
        nuevos = []
        for cl in clusters:
            mejor, dmin = None, self.radio
            for cand in self.lista:
                d = math.hypot(cl['x'] - cand['x'], cl['y'] - cand['y'])
                if d < dmin:
                    mejor, dmin = cand, d
            if mejor is None:
                self._sig += 1
                cand = dict(id=self._sig, x=cl['x'], y=cl['y'], tam=cl['tam'],
                            votos=cl['votos'], estado=PENDIENTE, intentos=0,
                            angulos_usados=[])
                self.lista.append(cand)
                nuevos.append(cand)
            else:
                mejor.update(x=cl['x'], y=cl['y'], tam=cl['tam'], votos=cl['votos'])
        return nuevos

    def marcar_rojo_cerca(self, x, y, radio):
        mejor, dmin = None, radio
        for cand in self.lista:
            d = math.hypot(x - cand['x'], y - cand['y'])
            if d < dmin:
                mejor, dmin = cand, d
        if mejor is not None and mejor['estado'] != ROJO:
            mejor['estado'] = ROJO
            return mejor
        return None

    def con_estado(self, *estados):
        return [c for c in self.lista if c['estado'] in estados]

    def sugerencias_para_relleno(self, tam_caja_min=0.06, tam_caja_max=0.50):
        # lo que no se confirmo, para gastar los reportes que sobran al final
        def tam_ok(c):
            return tam_caja_min <= c['tam'] <= tam_caja_max
        primero = sorted([c for c in self.lista if c['estado'] in (PENDIENTE, INALCANZABLE) and tam_ok(c)],
                         key=lambda c: -c['votos'])
        despues = sorted([c for c in self.lista if c['estado'] == NO_ROJO and tam_ok(c)],
                         key=lambda c: -c['votos'])
        return [(c['x'], c['y']) for c in primero + despues]
