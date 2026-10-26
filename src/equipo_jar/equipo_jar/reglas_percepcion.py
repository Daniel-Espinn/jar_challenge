'''
Funciones puras de percepcion sin involucrar ROS
Van a ser llamadas en percepcion.py
'''
import numpy as np

def agrupar_en_rachas(booleanos: np.ndarray):
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