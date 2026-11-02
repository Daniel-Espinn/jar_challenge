import numpy as np

def posicion(candidata) -> tuple:
    x = float(np.median(candidata['xs']))
    y = float(np.median(candidata['ys']))
    return (x,y)

def dispersion(candidata: dict) -> float:
    # mediana de las distancias a la mediana, asi un aviso loco no la arruina
    x_mediana = float(np.median(candidata['xs']))
    y_mediana = float(np.median(candidata['ys']))
    
    x_diferencias = np.array(candidata['xs']) - x_mediana
    y_diferencias = np.array(candidata['ys']) - y_mediana

    distancias = np.hypot(y_diferencias, x_diferencias)
    distancia_mediana = float(np.median(distancias))

    return distancia_mediana
    

def buscar_candidata(pizarron:list, x:float, y:float, radio:float) -> int|None:
    mejor_cand = None
    distancia = radio
    for i,candidata in enumerate(pizarron):
        px,py = posicion(candidata)
        distancia_i = np.hypot(x-px, y-py)
        if distancia_i < distancia:
            mejor_cand = i
            distancia = distancia_i
    return mejor_cand


def esta_confirmada(candidata:dict, min_avisos:int, disp_max:float) -> bool:
    return len(candidata['xs']) >= min_avisos and dispersion(candidata) <= disp_max
    

def es_duplicada(pizarron:list, x:float, y:float, radio_dup:float) -> bool:
    for candidata in pizarron:
        if candidata['reportada']:
            px,py = posicion(candidata)
            distancia_i = np.hypot(x-px, y-py)
            if distancia_i <= radio_dup:
                return True
    return False

def elegir_relleno(pizarron: list, sugerencias: list, quedan: int, radio_dup: float,
                   min_avisos_relleno: int = 1) -> list:
    elegidos = [posicion(c) for c in pizarron if c['reportada']]
    salida = []

    def lejos_de_todo(x, y):
        return all(np.hypot(x - ex, y - ey) > radio_dup for ex, ey in elegidos)

    no_reportadas = sorted([c for c in pizarron if not c['reportada']
                            and len(c['xs']) >= min_avisos_relleno],
                           key=lambda c: -len(c['xs']))
    for c in no_reportadas:
        if len(salida) >= quedan:
            return salida
        x, y = posicion(c)
        if lejos_de_todo(x, y):
            salida.append((x, y))
            elegidos.append((x, y))
    for x, y in sugerencias:
        if len(salida) >= quedan:
            return salida
        if np.isfinite(x) and np.isfinite(y) and lejos_de_todo(x, y):
            salida.append((float(x), float(y)))
            elegidos.append((x, y))
    return salida
