# tests de agrupar_en_rachas
# para testear debo comparar listas de arrays,
# pero para eso conviene pasar cada array de la lista
# a lista.
import numpy as np
from equipo_jar.reglas_percepcion import agrupar_en_rachas
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

@pytest.mark.parametrize(", esperado"[
    (0, 0, 0,         0,         2,  2, 0),
    (1, 1, 0,         np.pi / 2, 2,  1, 3),
    (0, 0, np.pi / 2, 0,         1,  0, 1),
    (1, 2, np.pi,     0,         1,  0, 2),
], ids=['origen', 'traslado', 'girado_90', 'girado_180'])
def test_puntos,en_mapa(tx, ty, yaw, angulo, rango, x_esp, y_esp):
    # armar un scan de juguete con UN rayo, usando los valores de la fila
    rangos = np.array([rango])
    angulos = np.array([angulo])

    xs, ys, validos = puntos_en_mapa(rangos, angulos, tx, ty, yaw, 0.1, 10)

    # la función devuelve un array por cosa; con 1 rayo, miramos el elemento [0]
    assert xs[0] == pytest.approx(x_esp)
    assert ys[0] == pytest.approx(y_esp)
    assert validos[0]
