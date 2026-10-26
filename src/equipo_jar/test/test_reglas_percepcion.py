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
