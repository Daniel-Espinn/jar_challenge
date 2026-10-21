# idea: Probar todas las funciones de relglas_cerebro.py
# aqui con pytest para evitar tener que launchear gazebo 
# cada vez que queremos probar funciones elementales.
# Evitamos que nos den errores tontos o resultados limites
# de tipo infinito culpa de dividir por cero o que
# queden reservas de segundo en negativo.
# Usamos pytest para testear rapido con tablas y probar todo lo que
# necesitamos en una iteracion de una viendo el resultado de cada una
# y si tiran los errores que deberian tirar o no

import math
import pytest
from equipo_jar.reglas_cerebro import PI, Pose2D, distancia, normalizar_angulo, yaw_a_cuaternion, cuaternion_a_yaw, tiempo_restante, calcular_reserva, hay_que_volver, reportes_disponibles, demasiado_tiempo_quieto, toca_loguear

@pytest.mark.parametrize("a, b, esperado", [
    ((0.0, 0,0), (3.0,4.0), 5.0),
    ((1.0,1.0),(1.0, 1.0), 0.0),
    ((-1.0,-1.0),(2.0,3.0), 5.0),
])
def test_distancia_casos(a,b, esperado):
    pose_a = Pose2D(*a)
    pose_b = Pose2D(*b)
    assert distancia(pose_a, pose_b) == pytest.approx(esperado)

def test_distancia_ignora_el_yaw():
    pose_a = Pose2D(0.0, 0.0, yaw=0.0)
    pose_b = Pose2D(3.0, 4.0, yaw=2.0)
    assert distancia(pose_a, pose_b) == pytest.approx(5.0)

@pytest.mark.parametrize("theta, esperado", [
    ((3*PI/2), -PI/2),
    ((-3*PI/2), PI/2),
    ((0.5), 0.5),
    ((5*PI), PI),
])
def test_normalizar_angulo(theta, esperado):
    assert normalizar_angulo(theta) == pytest.approx(esperado)

@pytest.mark.parametrize("yaw, esperado", [
    ((0), (0,1)),
    ((PI/2), (0.7071, 0.7071)),
    ((PI), (1,0)),
])
def test_yaw_a_cuaternion(yaw, esperado):
    assert yaw_a_cuaternion(yaw) == pytest.approx(esperado, abs=1e-4)

@pytest.mark.parametrize("z, w, esperado", [
    (0, 1, 0),
    ((0.7071), (0.7071), PI/2),
    ((1), (0), PI),
])
def test_cuaternion_a_yaw(z, w, esperado):
    assert cuaternion_a_yaw(z,w) == pytest.approx(esperado)

@pytest.mark.parametrize("t_inicio, t_ahora, duracion_s, esperado", [
    (100, 400, 600, 300),
    (100, 100, 600, 600),
    (100, 700, 600, 0),
    (100, 800, 600, 0),
])
def test_tiempo_restante(t_inicio, t_ahora, duracion_s, esperado):
    assert tiempo_restante(t_inicio, t_ahora, duracion_s) == pytest.approx(esperado)

@pytest.mark.parametrize("distancia_m, velocidad_ms, factor_camino, margen_s, esperado", [
    (8.0, 0.25, 1.5, 20.0, 68.0),
    (0.0, 0.25, 1.5, 20.0, 20.0),
])
def test_calcular_reserva(distancia_m, velocidad_ms, factor_camino, margen_s, esperado):
    assert calcular_reserva(distancia_m, velocidad_ms, factor_camino, margen_s) == pytest.approx(esperado)


@pytest.mark.parametrize("distancia_m, velocidad_ms", [
    (8.0, 0.0),
    (8.0, -0.1),
    (-1.0, 0.25),
])
def test_calcular_reserva_invalida(distancia_m, velocidad_ms):
    with pytest.raises(ValueError):
        calcular_reserva(distancia_m, velocidad_ms)
@pytest.mark.parametrize("tiempo_restante_s, reserva_s, reportes_hechos, n_victimas, quedan_viewpoints, esperado", [
    (300, 68, 1, 3, True, False),
    (67, 68, 1, 3, True, True),
    (68, 68, 1, 3, True, False),
    (300, 68, 3, 3, True, True),
    (300, 68, 1, 3, False, True),
])
def test_hay_que_volver(tiempo_restante_s, reserva_s, reportes_hechos, n_victimas, quedan_viewpoints, esperado):
    assert hay_que_volver(tiempo_restante_s, reserva_s, reportes_hechos, n_victimas, quedan_viewpoints) == pytest.approx(esperado)

@pytest.mark.parametrize("reportes_hechos, n_victimas, esperado", [
    (0,3,3),
    (1,3,2),
    (3,3,0),
    (4,3,0),
])
def test_reportes_disponibles(reportes_hechos, n_victimas, esperado):
    assert reportes_disponibles(reportes_hechos, n_victimas) == pytest.approx(esperado)

@pytest.mark.parametrize("t_ultimo_movimiento_s, t_ahora_s, esperado", [
    (100, 130, False),
    (100, 140, True),
    (100, 200, True),
])
def test_demasiado_tiempo_quieto(t_ultimo_movimiento_s, t_ahora_s, esperado):
    assert demasiado_tiempo_quieto(t_ultimo_movimiento_s, t_ahora_s) == pytest.approx(esperado)

@pytest.mark.parametrize("t_ultimo_log_s, t_ahora_s, periodo_s, esperado", [
    (None, 5, 10, True),
    (100, 105, 10, False),
    (100, 110, 10, True),
    (100, 115, 10, True),
])
def test_toca_loguear(t_ultimo_log_s, t_ahora_s, periodo_s, esperado):
    assert toca_loguear(t_ultimo_log_s, t_ahora_s, periodo_s) == pytest.approx(esperado)