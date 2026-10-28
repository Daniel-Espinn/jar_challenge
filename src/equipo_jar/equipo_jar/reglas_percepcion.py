'''
Funciones puras de percepcion sin involucrar ROS
Van a ser llamadas en percepcion.py
'''
import numpy as np
import cv2

def agrupar_en_rachas(booleanos):
    """
    Agrupar los índices contiguos de un array booleano en "rachas".

    Una racha que tiene True en el ultimo y primer elemento se hace
    una sola
    Parámetro: `booleanos`, un array de N bools (uno por rayo del scan,
    True donde ese rayo cayó en un punto "no mapeado").
    Devuelve una lista de arrays de índices, uno por racha
    """

    # devuelvo las posiciones de los True
    indices = np.flatnonzero(booleanos)
    if len(indices) == 0:
        return []

    # defino las diferencias entre vecinos y sus saltos
    # donde hubo cortes de racha
    diferencias = np.diff(indices)
    saltos = np.flatnonzero(diferencias > 1) 

    # defino donde se cortan y las rachas en base a los cortes
    cortes = saltos + 1
    rachas = np.split(indices, cortes)

    # como scan es circular si hay True en bordes concatenamos
    if len(rachas) > 1 and rachas[0][0] == 0 and rachas[-1][-1] == (len(booleanos) - 1): 
        rachas[0] = np.concatenate([rachas[-1], rachas[0]])
        rachas.pop()

    return rachas


def puntos_en_mapa(rangos, angulos, tx, ty, yaw, rango_min, rango_max):
    """
    traduccion de los scans y posicion actual del robot a coordenadas de los objetos
    Como necesitamos donde esta cada objeto en el mapa y el lidar
    nos da a que distancia del sensor esta cada cosa, esta funcion hace
    esa traduccion.
    """
    # filtramos rayos validos
    validos = (np.isfinite(rangos)) & (rangos >= rango_min) & (rangos <= rango_max)

    # rangos invalidos en cero
    rangos = np.where(validos,rangos,0.0)

    # punto desde lidar
    xs_laser = rangos * np.cos(angulos)
    ys_laser = rangos * np.sin(angulos)
    
    # centro de columna y fila
    c, s = np.cos(yaw), np.sin(yaw)

    # obtenemos puntos en mapa
    xs_map = tx + c*xs_laser - s*ys_laser
    ys_map = ty + s*xs_laser + c*ys_laser
    return xs_map, ys_map, validos

def estimar_mitad_caja(ancho, ancho_min, ancho_max):
    '''
    
    '''
    valor_ancho = np.clip(ancho, ancho_min, ancho_max)
    mitad_caja = valor_ancho / 2
    return mitad_caja

def corregir_al_centro(x_cara, y_cara, x_robot, y_robot, mitad_caja):
    '''
    Correo el punto de la cara media caja mas lejos del robot
    porque estimamos que ahi esta el centro
    '''
    # calculo distancias e hipot
    dx = x_cara - x_robot
    dy = y_cara - y_robot
    largo = np.hypot(dx, dy)

    # si estamos pegados devolvemos esas coords
    if largo == 0.0:
        return (x_cara, y_cara)

    # si no calculamos centor de caja
    else:
        x_centro = x_cara + mitad_caja * (dx / largo)
        y_centro = y_cara + mitad_caja * (dy / largo)
        return (x_centro, y_centro)

def mascara_rojo(imagen_bgr, hue_rojo_bajo_1, hue_rojo_alto_1, hue_rojo_bajo_2, hue_rojo_alto_2,
                  saturacion_min, valor_min) -> np.ndarray:
    """Máscara binaria (255 = rojo)."""
    # bgr a hsv
    imagen_hsv = cv2.cvtColor(imagen_bgr, cv2.COLOR_BGR2HSV)

    bajo1 = np.array([hue_rojo_bajo_1, saturacion_min, valor_min])
    alto1 = np.array([hue_rojo_alto_1, 255, 255])

    bajo2 = np.array([hue_rojo_bajo_2, saturacion_min, valor_min])
    alto2 = np.array([hue_rojo_alto_2, 255, 255])

    # creamos rangos de color bajo y alto
    mascara1 = cv2.inRange(imagen_hsv, bajo1, alto1)
    mascara2 = cv2.inRange(imagen_hsv, bajo2, alto2)

    # devolvemos mascara pixel a pixel
    return cv2.bitwise_or(mascara1, mascara2)

def rayos_sobre_rojo(rangos, angulos, R, t, fx, fy, cx, cy, mascara) -> np.array:
    '''
    Salida, un array con True en los rayos donde caiga en un pixel rojo y False donde sea distinto.
    IN:
        rangos: rayos del scan (m)
        angulos: angulos de los rayos (en rad)
        R (3x3): rotacion, transformacion laser al frame optico de la camara
        t: traslacion vector de 3 coords. transformacion laser al frame optico de la camara
        fx, fy, cx, cy: son los zooms y los centros. vienen de camara_info
    '''

    # punto lidar
    rangos_seguros = np.where(np.isfinite(rangos), rangos, 0.0)
    x = rangos_seguros * np.cos(angulos)
    y = rangos_seguros * np.sin(angulos)

    # fila [x,y,0] por rayo
    punto_laser = np.stack([x, y, np.zeros_like(x)], axis=1)

    # mismo punto visto desde la camara
    puntos_cam = punto_laser @ R.T + t
    # derecha abajo y adelante
    X,Y,Z = puntos_cam[:,0], puntos_cam[:,1], puntos_cam[:,2]

    # del punto al vidiro 
    en_frente = (np.isfinite(rangos)) & (Z > 0)
    # no dividir por cero
    Z_seguro = np.where(en_frente, Z, 1.0)
    # obtenemos pixel con columnas y filas
    columnas = np.round((X/Z_seguro) * fx + cx).astype(int)
    filas = np.round((Y/Z_seguro) * fy + cy).astype(int)

    # pasamos a ancho y alto la mascara de 255/0
    alto,ancho = mascara.shape
    dentro = (columnas >= 0) & (columnas < ancho) & (filas >= 0) & (filas < alto)

    ok = en_frente & dentro
    # arrancamos en false y rellenamos
    sobre_rojo = np.zeros(len(rangos), dtype=bool)
    sobre_rojo[ok] = mascara[filas[ok], columnas[ok]] > 0
    return sobre_rojo

def cuaternion_a_matriz(x, y, z, w):
    """Matriz de rotación 3x3 de un cuaternión (x, y, z, w)."""
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])