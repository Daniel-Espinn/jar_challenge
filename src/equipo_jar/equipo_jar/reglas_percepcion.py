import numpy as np
import cv2

def agrupar_en_rachas(booleanos):

    indices = np.flatnonzero(booleanos)
    if len(indices) == 0:
        return []

    diferencias = np.diff(indices)
    saltos = np.flatnonzero(diferencias > 1) 

    cortes = saltos + 1
    rachas = np.split(indices, cortes)

    if len(rachas) > 1 and rachas[0][0] == 0 and rachas[-1][-1] == (len(booleanos) - 1): 
        rachas[0] = np.concatenate([rachas[-1], rachas[0]])
        rachas.pop()

    return rachas


def tramo_mas_cercano(racha, rangos, salto_max, rayos_minimos):
    # corta la racha donde salta el rango (la pared de atras no es la caja)
    racha = np.asarray(racha)
    if len(racha) == 0:
        return racha
    cortes = np.flatnonzero(np.abs(np.diff(rangos[racha])) > salto_max) + 1
    tramos = [t for t in np.split(racha, cortes) if len(t) >= rayos_minimos]
    if not tramos:
        return racha[:0]
    return min(tramos, key=lambda t: float(np.median(rangos[t])))


def columnas_de_rayos(rangos, angulos, R, t, fx, cx):
    rangos_seguros = np.where(np.isfinite(rangos), rangos, 0.0)
    punto_laser = np.stack([rangos_seguros * np.cos(angulos),
                            rangos_seguros * np.sin(angulos),
                            np.zeros_like(rangos_seguros)], axis=1)
    puntos_cam = punto_laser @ R.T + t
    X, Z = puntos_cam[:, 0], puntos_cam[:, 2]
    en_frente = np.isfinite(rangos) & (Z > 0)
    Z_seguro = np.where(en_frente, Z, 1.0)
    columnas = np.round((X / Z_seguro) * fx + cx).astype(int)
    return np.where(en_frente, columnas, -1)


def algo_muy_cerca_en_imagen(rangos, columnas, ancho_imagen, rango_min):
    en_imagen = (columnas >= 0) & (columnas < ancho_imagen)
    cortos = np.isfinite(rangos) & (rangos < rango_min)
    return bool(np.any(en_imagen & cortos))


def toca_borde(racha, columnas, ancho_imagen, margen_px):
    c = columnas[np.asarray(racha)]
    return bool(c.min() < margen_px or c.max() > ancho_imagen - 1 - margen_px)


def puntos_en_mapa(rangos, angulos, tx, ty, yaw, rango_min, rango_max):
    validos = (np.isfinite(rangos)) & (rangos >= rango_min) & (rangos <= rango_max)

    rangos = np.where(validos,rangos,0.0)

    xs_laser = rangos * np.cos(angulos)
    ys_laser = rangos * np.sin(angulos)
    
    c, s = np.cos(yaw), np.sin(yaw)

    xs_map = tx + c*xs_laser - s*ys_laser
    ys_map = ty + s*xs_laser + c*ys_laser
    return xs_map, ys_map, validos

def estimar_mitad_caja(ancho, ancho_min, ancho_max):
    valor_ancho = np.clip(ancho, ancho_min, ancho_max)
    mitad_caja = valor_ancho / 2
    return mitad_caja

def corregir_al_centro(x_cara, y_cara, x_robot, y_robot, mitad_caja):
    # se puntua contra el centro, asi que corremos media caja para atras
    dx = x_cara - x_robot
    dy = y_cara - y_robot
    largo = np.hypot(dx, dy)

    if largo == 0.0:
        return (x_cara, y_cara)

    else:
        x_centro = x_cara + mitad_caja * (dx / largo)
        y_centro = y_cara + mitad_caja * (dy / largo)
        return (x_centro, y_centro)

def mascara_rojo(imagen_bgr, hue_rojo_bajo_1, hue_rojo_alto_1, hue_rojo_bajo_2, hue_rojo_alto_2,
                  saturacion_min, valor_min) -> np.ndarray:
    # el rojo esta en las dos puntas del hue, por eso dos rangos
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    bajo1 = np.array([hue_rojo_bajo_1, saturacion_min, valor_min])
    alto1 = np.array([hue_rojo_alto_1, 255, 255])

    bajo2 = np.array([hue_rojo_bajo_2, saturacion_min, valor_min])
    alto2 = np.array([hue_rojo_alto_2, 255, 255])

    mascara1 = cv2.inRange(imagen_hsv, bajo1, alto1)
    mascara2 = cv2.inRange(imagen_hsv, bajo2, alto2)

    return cv2.bitwise_or(mascara1, mascara2)

def rayos_sobre_rojo(rangos, angulos, R, t, fx, fy, cx, cy, mascara) -> np.array:

    rangos_seguros = np.where(np.isfinite(rangos), rangos, 0.0)
    x = rangos_seguros * np.cos(angulos)
    y = rangos_seguros * np.sin(angulos)

    punto_laser = np.stack([x, y, np.zeros_like(x)], axis=1)

    puntos_cam = punto_laser @ R.T + t
    X,Y,Z = puntos_cam[:,0], puntos_cam[:,1], puntos_cam[:,2]

    en_frente = (np.isfinite(rangos)) & (Z > 0)
    Z_seguro = np.where(en_frente, Z, 1.0)
    columnas = np.round((X/Z_seguro) * fx + cx).astype(int)
    filas = np.round((Y/Z_seguro) * fy + cy).astype(int)

    alto,ancho = mascara.shape
    dentro = (columnas >= 0) & (columnas < ancho) & (filas >= 0) & (filas < alto)

    ok = en_frente & dentro
    sobre_rojo = np.zeros(len(rangos), dtype=bool)
    sobre_rojo[ok] = mascara[filas[ok], columnas[ok]] > 0
    return sobre_rojo

def cuaternion_a_matriz(x, y, z, w):
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])