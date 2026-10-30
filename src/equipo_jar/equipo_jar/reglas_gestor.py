'''
Funciones para implementar gestion de victimas
'''
import numpy as np

def posicion(candidata) -> tuple:
    '''
    Donde esta en el mapa?
    Posicion estimada de la candidata.
    Tomamos el x e y mediano de caja eje recibido para evitar
    sacar promedio con valoraciones locas como cuando giramos
    '''
    x = float(np.median(candidata['xs']))
    y = float(np.median(candidata['ys']))
    return (x,y)

def dispersion(candidata: dict) -> float:
    '''
    Que tan juntos estan los avisos?
    '''
    x_mediana = float(np.median(candidata['xs']))
    y_mediana = float(np.median(candidata['ys']))
    
    # convertimos lista de candidatas a array
    x_diferencias = np.array(candidata['xs']) - x_mediana
    y_diferencias = np.array(candidata['ys']) - y_mediana

    distancias = np.hypot(y_diferencias, x_diferencias)
    distancia_mediana = float(np.median(distancias))

    return distancia_mediana
    

def buscar_candidata(pizarron:list, x:float, y:float, radio:float) -> int|None:
    '''
    Este aviso es de alguna que ya tengo?
    '''
    mejor_cand = None
    distancia = radio
    for i,candidata in enumerate(pizarron):
        px,py = posicion(candidata)
        distancia_i = np.hypot(x-px, y-py)
        if distancia_i < distancia:
            mejor_cand = i
            distancia = distancia_i
    return mejor_cand


def esta_confirmada(candidata:dict, min_avisos:int, disp_max:int) -> bool:
    '''
    ya estoy seguro de la candidata?
    '''
    return len(candidata['xs']) <= min_avisos and dispersion(candidata) >= disp_max
    

def es_duplicada(pizarron:list, x:float, y:float, radio_dup:float) -> bool:
    '''
    ya reporte algo muy cerca de aca?
    '''
    for candidata in pizarron:
        if candidata['reportada']:
            px,py = posicion(candidata)
            distancia_i = np.hypot(x-px, y-py)
            if distancia_i <= radio_dup:
                return True
    return False