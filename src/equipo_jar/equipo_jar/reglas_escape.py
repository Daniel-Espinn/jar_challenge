# escape cuando se traba + reserva para volver
import math

import numpy as np


def soporte_footprint(esquinas, angulo):
    u = np.array([math.cos(angulo), math.sin(angulo)])
    return float(max(np.dot(np.asarray(e, dtype=float), u) for e in esquinas))


def _soporte_por_rayo(esquinas, angulos):
    e = np.asarray(esquinas, dtype=float)                     # (k, 2)
    u = np.stack([np.cos(angulos), np.sin(angulos)], axis=1)  # (n, 2)
    return (u @ e.T).max(axis=1)


def despeje_en_direccion(rangos, angulos_base, esquinas_desde_laser, rango_min_valido,
                         direccion, ancho_sector=math.radians(50)):
    # lugar libre medido desde el borde del chasis, no desde el lidar
    rangos = np.asarray(rangos, dtype=float)
    angulos_base = np.asarray(angulos_base, dtype=float)
    validos = np.isfinite(rangos) & (rangos >= rango_min_valido)
    dif = np.abs(np.arctan2(np.sin(angulos_base - direccion), np.cos(angulos_base - direccion)))
    en_sector = validos & (dif <= ancho_sector / 2)
    if not en_sector.any():
        return 10.0
    sop = _soporte_por_rayo(esquinas_desde_laser, angulos_base[en_sector])
    return float(np.min(rangos[en_sector] - sop))


def direccion_escape(rangos, angulos_base, esquinas_desde_laser, rango_min_valido,
                     n_direcciones=8, ancho_sector=math.radians(50), despeje_min=0.12):
    mejor = None
    for k in range(n_direcciones):
        d = -math.pi + k * 2 * math.pi / n_direcciones
        despeje = despeje_en_direccion(rangos, angulos_base, esquinas_desde_laser,
                                       rango_min_valido, d, ancho_sector)
        if despeje >= despeje_min and (mejor is None or despeje > mejor[1]):
            mejor = (d, despeje)
    return mejor


def tiempo_reserva(camino_m, velocidad_efectiva=0.15, factor=1.5, margen_s=30.0):
    # factor * camino / vel + margen
    return factor * camino_m / velocidad_efectiva + margen_s
