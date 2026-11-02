import math
from dataclasses import dataclass
PI = math.pi

@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float = 0

def distancia(a: Pose2D, b: Pose2D) -> float:
    hipot = ((b.x - a.x)**2 + (b.y - a.y)**2)**(1/2)
    return hipot

def normalizar_angulo(theta: float) -> float:
    proyeccion_x = math.cos(theta)
    proyeccion_y = math.sin(theta)
    normalizacion = math.atan2(proyeccion_y, proyeccion_x)
    return normalizacion

def yaw_a_cuaternion(yaw: float) -> tuple[float, float]:
    z = math.sin(yaw/2)
    w = math.cos(yaw/2)
    return z, w

def cuaternion_a_yaw(z:float, w:float) -> float:
    theta = 2 * math.atan2(z, w)
    return theta
    
def tiempo_restante(t_inicio: float, t_ahora: float, duracion_s: float = 600.0) -> float:
    t_restante = duracion_s - (t_ahora - t_inicio)
    if t_restante < 0:
        t_restante = 0
    return t_restante

def calcular_reserva(distancia_m: float, velocidad_ms: float, factor_cambio:float = 1.5, margen_s: float = 20.0) -> float:
    if not velocidad_ms > 0:
        raise ValueError(f"La velocidad debe ser positiva, se recibio {velocidad_ms}")
    if not distancia_m >= 0:
        raise ValueError(f"La distancia a la largada no puede ser negativa, se recibió {distancia_m}")
    reserva_s = (distancia_m / velocidad_ms) * factor_cambio + margen_s
    return reserva_s

def hay_que_volver(tiempo_restante_s: float, reserva_s:float, reportes_hechos:int, n_victimas:int, quedan_viewpoints:bool) -> bool:
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

def toca_loguear(t_ultimo_log_s:float | None, t_ahora_s:float, periodo_s:float = 10.0) -> bool:
    loguear = False
    if t_ultimo_log_s == None or (t_ahora_s - t_ultimo_log_s) >= periodo_s:
        loguear = True
    return loguear

    


