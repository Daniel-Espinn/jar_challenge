'''
Funciones puras de percepcion sin involucrar ROS
Van a ser llamadas en percepcion.py
'''
import numpy as np

def agrupar_en_rachas():
    """
    Agrupar los índices contiguos de un array booleano en "rachas".

    Un obstáculo real casi nunca genera un solo punto "no mapeado" suelto:
    genera una racha de varios rayos seguidos del lidar pegando en la misma
    cara del objeto (igual que ya viste con `/scan_rojo` en semana 04). Acá
    identificamos esas rachas para después convertir cada una en un objeto.

    Parámetro: `booleanos`, un array de N bools (uno por rayo del scan,
    True donde ese rayo cayó en un punto "no mapeado").

    Devolvé una lista de arrays de índices, uno por racha — por ejemplo,
    para [F,F,T,T,T,F,T,T,F] tendría que devolver [[2,3,4], [6,7]].
    """

    # Paso 1: devuelvo las posiciones de los True
    # los True son aquellos rayos que detectaron algo
    # que no estaba en el mapa
    indices = np.flatnonzero(booleanos)
    if len(indices) == 0:
        return []

    # Paso 2: diferencias entre vecinos y dónde hay salto
    # la diferencia es como la diferencia de espacio entre postes
    # si hay mas de un poste de diferencia entre postes
    # es porque hubo un corte y ahi habria una coma que divide dos rachas
    # los saltos nos devuelve efectivamente aquellos lugares en los
    # que la diferencia es mayor a 1, es decir posiciones
    # donde hubo cortes de racha
    diferencias = np.diff(indices)
    saltos = np.flatnonzero(diferencias > 1)                   # posiciones (dentro de diferencias) donde la diferencia es > 1

    # Paso 3: cortar (acordate de los postes y el alambrado)
    # ahora lo que hadcemos es separar en rachas para que queden
    # divididas con un split que divida en base a los indices y cortes
    cortes = saltos + 1
    rachas = np.split(indices, cortes)

    # Paso 4: el scan es circular
    # si el largo de la lista de rachas es mayor a 1
    # y si el primer elemento de la primer racha es igual
    # a 0 y si el ultimo elemento de la ultima racha es igual
    # al indice de del ultimo indice de la lista de booleanos
    # concatenamos la primer racha sumando el final al inicio y
    # y el incio al final, es la union entre 360 y 0
    # va primero la ultima, y popeamos
    if len(rachas) > 1 and rachas[0][0] == 0 and rachas[-1][-1] == (len(booleanos) - 1):     # primera empieza en 0  /  última termina en len(booleanos) - 1
        rachas[0] = np.concatenate([rachas[-1], rachas[0]])   # ¿cuál va primero?
        rachas.pop()

    # Paso 5 devolvemos la lista d erachas
    return rachas


def puntos_en_mapa(rangos, angulos, tx, ty, yaw, rango_min, rango_max):
    """
    Como necesitamos donde esta cada objeto en el mapa y el lidar
    nos da a que distancia del sensor esta cada cosa, esta funcion hace
    esa traduccion.
    """
    # Nos quedamos con aquellos rayos que cumplen que 1.son finitos, 2.cuya distancia
    # es mayor a la distancia minima 3.y cuya distancia es menor a la maxima
    validos = (np.isfinite(rangos)) & (rangos >= rango_min) & (rangos <= rango_max)
    # modificamos rangos para que los validos queden en su posicion
    # y aquellos que no sean validos se le spone cero para que no molesten
    rangos = np.where(validos,rangos,0.0)
    # ?
    xs_laser = rangos * np.cos(angulos)
    ys_laser = rangos * np.sin(angulos)
    # ?
    c, s = np.cos(yaw), np.sin(yaw)
    # Aplicamos la formula para obtener los puntos en el mapa
    # usando las transformaciones. Tambien devolvemos los 
    # los valores validos
    xs_map = tx + c*xs_laser - s*ys_laser
    ys_map = ty + s*xs_laser + c*ys_laser
    return xs_map, ys_map, validos

def estimar_mitad_caja(ancho, ancho_min, ancho_max):
    valor_ancho = np.clip(ancho, ancho_min, ancho_max)
    mitad_caja = valor_ancho / 2
    return mitad_caja

def corregir_al_centro(x_cara, y_cara, x_robot, y_robot, mitad_caja):
    '''
    Correo el punto de la cara media caja mas lejos del robot
    porque estimamos que ahi esta el centro
    '''
    dx = x_cara - x_robot
    dy = y_cara - y_robot
    largo = np.hypot(dx, dy)

    if largo == 0.0:
        return (x_cara, y_cara)
    else:
        x_centro = x_cara + mitad_caja * (dx / largo)
        y_centro = y_cara + mitad_caja * (dy / largo)
        return (x_centro, y_centro)
