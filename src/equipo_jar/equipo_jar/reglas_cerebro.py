import math
from dataclasses import dataclass
PI = math.pi

# usmaos dataclass para no tener que crear
# metodos basicos
@dataclass(frozen=True)
# Cada posicion que llegue al cerebro se pasa
# a Pose2D y es inmutable
class Pose2D:
    x: float
    y: float
    yaw: float = 0

def distancia(a: Pose2D, b: Pose2D) -> float:
    """
    Calculamos distancia entre dos posiciones
    IN: a=posiicon1 y b=posicion2
    OUT: distancia entre los vectores
    """
    hipot = ((b.x - a.x)**2 + (b.y - a.y)**2)**(1/2)
    return hipot

def normalizar_angulo(theta: float) -> float:
    """
    Llevar cualquier angulo al rango (-pi, pi]
    IN: un angulo theta
    OUT: un angulo normalizado
    """
    proyeccion_x = math.cos(theta)
    proyeccion_y = math.sin(theta)
    normalizacion = math.atan2(proyeccion_y, proyeccion_x)
    return normalizacion

def yaw_a_cuaternion(yaw: float) -> tuple[float, float]:
    """
    TF y Nav2 piden cuaterniones. Son necesarios para
    que el robot solo gire sobre el eje vertical
    IN: angulo
    OUT: tupla de floats z y w porque
    x e y, quedan ifjos ya que estamos en el piso.
    """
    z = math.sin(yaw/2)
    w = math.cos(yaw/2)
    return z, w

def cuaternion_a_yaw(z:float, w:float) -> float:
    """
    Convertir el cuaternion en angulo
    IN: z y w, las ultimas posiicones del cuaternion
    x e y estan fijos en 0 porque no hay inclinacion
    OUT: angulo
    """
    theta = 2 * math.atan2(z, w)
    return theta
    
def tiempo_restante(t_inicio: float, t_ahora: float, duracion_s: float = 600.0) -> float:
    """
    Obtener el tiempo restante de cronometro
    IN: tiempo de inicio, tiempo de ahora mismo, duracion del cronometro
    OUT: tiempo restante
    """
    t_restante = duracion_s - (t_ahora - t_inicio)
    if t_restante < 0:
        t_restante = 0
    return t_restante

def calcular_reserva(distancia_m: float, velocidad_ms: float, factor_cambio:float = 1.5, margen_s: float = 20.0) -> float:
    """
    Segundos que hay que reservar para volver a la largada
    IN:
        distancia_m: distancia que recorrimos en m
        velocidad_ms: velocidad promedio del robot
        factor_camino: float estimado para tener en cuenta paredes y vueltas
        margen_s: el margen que añadimos si tenemos en cuenta giros,
        sobre el eje, reversa y posibles factores que hagan mas 
        lento el robot durante la vuelta
    OUT: 
        reserva_s: segundos de reserva para la vuelta
    Detalle: si la reserva sale demasiado chica tenemos poco tiemopo de vuelta y hay que ver si debemos volver
    si es muy grande debemos seguir explorando.
    """
    if not velocidad_ms > 0:
        raise ValueError(f"La velocidad debe ser positiva, se recibio {velocidad_ms}")
    if not distancia_m >= 0:
        raise ValueError(f"La distancia a la largada no puede ser negativa, se recibió {distancia_m}")
    reserva_s = (distancia_m / velocidad_ms) * factor_cambio + margen_s
    return reserva_s

def hay_que_volver(tiempo_restante_s: float, reserva_s:float, reportes_hechos:int, n_victimas:int, quedan_viewpoints:bool) -> bool:
    """
    Si se cumple que la reserva en s es menor al tiempo restante en s
    o si ya hicmos N reportes,
    o si no quedan mas viewpoints por explorar:
    hay que volver y devuelve True
    IN:
        tiempo_restante_en_s
        reserva_s: segundos estimados de reserva para la vuelta
        reportes_hechos
        n_victimas: viene fijo por el juez
    OUT:
        voler_a_inicio: bool, si hay que volver al inicio o no.
    """
    volver_a_inicio = False
    if tiempo_restante_s < reserva_s or n_victimas == reportes_hechos or quedan_viewpoints == 0:
        volver_a_inicio = True
        return volver_a_inicio
    else:
        return volver_a_inicio

def reportes_disponibles(reportes_hechos:int, n_victimas:int) -> int:
    reportes_restantes = n_victimas - reportes_hechos
    return max(0,reportes_restantes)


def demasiado_tiempo_quieto(t_ultimo_movimiento_s: float, t_ahora_s:float, limite_s: float = 40.0) -> bool:
    mucho_tiempo_quieto = False
    if (t_ahora_s - t_ultimo_movimiento_s) >= limite_s:
        mucho_tiempo_quieto = True
        return mucho_tiempo_quieto
    else:
        return mucho_tiempo_quieto

    


