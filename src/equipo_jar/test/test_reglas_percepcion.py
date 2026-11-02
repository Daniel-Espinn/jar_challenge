import numpy as np
from equipo_jar.reglas_percepcion import agrupar_en_rachas, puntos_en_mapa
import pytest

@pytest.mark.parametrize("booleanos, esperado", [
    ([0]*8, []),
    ([0, 0, 0, 1, 0, 0, 0, 0], [[3]]),
    ([0, 1, 1, 0, 0, 1, 0, 0], [[1, 2], [5]]),
    ([1, 1, 0, 0, 1, 0, 1, 1], [[6, 7, 0, 1], [4]]),
    ([1] * 8, [[0, 1, 2, 3, 4, 5, 6, 7]])
], ids=['vacio', 'un_rayo', 'seperadas', 'circular', 'todo'])
def test_rachas_en_rachas(booleanos, esperado):
    rachas = agrupar_en_rachas(np.array(booleanos, dtype=bool))
    assert [r.tolist() for r in rachas] == esperado

@pytest.mark.parametrize("tx, ty, yaw, angulo, rango, x_esp, y_esp", [
    (0, 0, 0,         0,         2,  2, 0),
    (1, 1, 0,         np.pi / 2, 2,  1, 3),
    (0, 0, np.pi / 2, 0,         1,  0, 1),
    (1, 2, np.pi,     0,         1,  0, 2),
], ids=['origen', 'traslado', 'girado_90', 'girado_180'])
def test_puntos_en_mapa(tx, ty, yaw, angulo, rango, x_esp, y_esp):
    rangos = np.array([rango])
    angulos = np.array([angulo])

    xs, ys, validos = puntos_en_mapa(rangos, angulos, tx, ty, yaw, 0.1, 10)

    assert xs[0] == pytest.approx(x_esp)
    assert ys[0] == pytest.approx(y_esp)
    assert validos[0]


from equipo_jar.reglas_percepcion import tramo_mas_cercano, toca_borde


def test_tramo_mas_cercano_saca_la_pared_de_atras():
    rangos = np.array([3.0, 3.0, 1.00, 1.001, 1.002, 1.003, 1.004, 3.0])
    racha = np.arange(8)
    assert tramo_mas_cercano(racha, rangos, 0.08, 3).tolist() == [2, 3, 4, 5, 6]


def test_tramo_mas_cercano_sin_saltos_deja_todo():
    rangos = np.array([1.0, 1.01, 1.02, 1.03])
    assert tramo_mas_cercano(np.arange(4), rangos, 0.08, 3).tolist() == [0, 1, 2, 3]


def test_tramo_mas_cercano_ignora_tramos_cortos():
    rangos = np.array([0.5, 1.0, 1.0, 1.0, 1.0])
    assert tramo_mas_cercano(np.arange(5), rangos, 0.08, 3).tolist() == [1, 2, 3, 4]
    assert len(tramo_mas_cercano(np.arange(2), rangos, 0.08, 3)) == 0


@pytest.mark.parametrize("cols, esperado", [
    ([100, 150], False), ([2, 50], True), ([200, 318], True), ([4, 315], False),
], ids=['centro', 'borde_izq', 'borde_der', 'justo_adentro'])
def test_toca_borde(cols, esperado):
    columnas = np.array(cols)
    assert toca_borde(np.arange(len(cols)), columnas, 320, 4) == esperado


def test_algo_muy_cerca_en_imagen():
    from equipo_jar.reglas_percepcion import algo_muy_cerca_en_imagen
    rangos = np.array([1.0, 0.17, 2.0, 0.10, np.inf])
    assert algo_muy_cerca_en_imagen(rangos, np.array([10, 50, 90, -1, 200]), 320, 0.25)
    assert not algo_muy_cerca_en_imagen(rangos, np.array([10, 400, 90, -1, 200]), 320, 0.25)
