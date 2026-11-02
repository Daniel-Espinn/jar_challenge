import heapq
import math

import numpy as np
import scipy.ndimage as nd

RAIZ2 = math.sqrt(2.0)


class Grilla:

    def __init__(self, datos, resolucion, origen_x, origen_y):
        datos = np.asarray(datos, dtype=np.int16)
        self.alto, self.ancho = datos.shape
        self.res = float(resolucion)
        self.ox, self.oy = float(origen_x), float(origen_y)
        self.libre = datos == 0
        self.pared = (datos >= 50) | (datos < 0)
        self.dist_pared = nd.distance_transform_edt(~self.pared) * self.res

    def mundo_a_celda(self, x, y):
        return int(math.floor((y - self.oy) / self.res)), int(math.floor((x - self.ox) / self.res))

    def mundo_a_celdas(self, xs, ys):
        filas = np.floor((np.asarray(ys) - self.oy) / self.res).astype(int)
        cols = np.floor((np.asarray(xs) - self.ox) / self.res).astype(int)
        return filas, cols

    def celda_a_mundo(self, fila, col):
        return self.ox + (col + 0.5) * self.res, self.oy + (fila + 0.5) * self.res

    def dentro(self, fila, col):
        return 0 <= fila < self.alto and 0 <= col < self.ancho

    def dentro_v(self, filas, cols):
        return (filas >= 0) & (filas < self.alto) & (cols >= 0) & (cols < self.ancho)


def grilla_navegable(grilla, obstaculos_extra, radio_nav):
    # donde puede estar el centro del robot
    bloqueado = grilla.pared | obstaculos_extra
    dist_total = nd.distance_transform_edt(~bloqueado) * grilla.res
    navegable = grilla.libre & ~obstaculos_extra & (dist_total >= radio_nav)
    return navegable, dist_total


def celda_navegable_mas_cercana(navegable, fila, col):
    if 0 <= fila < navegable.shape[0] and 0 <= col < navegable.shape[1] and navegable[fila, col]:
        return fila, col
    if not navegable.any():
        return None
    _, (idx_f, idx_c) = nd.distance_transform_edt(~navegable, return_indices=True)
    fila = int(np.clip(fila, 0, navegable.shape[0] - 1))
    col = int(np.clip(col, 0, navegable.shape[1] - 1))
    return int(idx_f[fila, col]), int(idx_c[fila, col])


def distancias_camino(navegable, fila, col, res):
    # dijkstra, metros de camino desde (fila, col)
    alto, ancho = navegable.shape
    dist = np.full((alto, ancho), np.inf)
    inicio = celda_navegable_mas_cercana(navegable, fila, col)
    if inicio is None:
        return dist
    nav = navegable.tolist()
    d = dist.tolist()
    f0, c0 = inicio
    d[f0][c0] = 0.0
    cola = [(0.0, f0, c0)]
    pasos = ((-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
             (-1, -1, RAIZ2), (-1, 1, RAIZ2), (1, -1, RAIZ2), (1, 1, RAIZ2))
    while cola:
        dd, f, c = heapq.heappop(cola)
        if dd > d[f][c]:
            continue
        for df, dc, costo in pasos:
            nf, nc = f + df, c + dc
            if not (0 <= nf < alto and 0 <= nc < ancho) or not nav[nf][nc]:
                continue
            if df and dc and not (nav[f + df][c] and nav[f][c + dc]):
                continue
            nd_ = dd + costo
            if nd_ < d[nf][nc]:
                d[nf][nc] = nd_
                heapq.heappush(cola, (nd_, nf, nc))
    return np.array(d) * res


def punto_intermedio(dist_a_meta, navegable, fila, col, avance_m):
    inicio = celda_navegable_mas_cercana(navegable, fila, col)
    if inicio is None:
        return None
    f, c = inicio
    if not np.isfinite(dist_a_meta[f, c]):
        return None
    objetivo = max(float(dist_a_meta[f, c]) - avance_m, 0.0)
    alto, ancho = dist_a_meta.shape
    while dist_a_meta[f, c] > objetivo:
        mejor = (f, c)
        for df in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nf, nc = f + df, c + dc
                if 0 <= nf < alto and 0 <= nc < ancho and dist_a_meta[nf, nc] < dist_a_meta[mejor]:
                    mejor = (nf, nc)
        if mejor == (f, c):
            break
        f, c = mejor
    return f, c


def linea_libre(bloqueado, fila0, col0, fila1, col1):
    # bresenham
    f0, c0, f1, c1 = int(fila0), int(col0), int(fila1), int(col1)
    dc, df = abs(c1 - c0), abs(f1 - f0)
    sc = 1 if c0 < c1 else -1
    sf = 1 if f0 < f1 else -1
    err = dc - df
    f, c = f0, c0
    while (f, c) != (f1, c1):
        e2 = 2 * err
        if e2 > -df:
            err -= df
            c += sc
        if e2 < dc:
            err += dc
            f += sf
        if (f, c) == (f1, c1):
            break
        if not (0 <= f < bloqueado.shape[0] and 0 <= c < bloqueado.shape[1]) or bloqueado[f, c]:
            return False
    return True


def celdas_vistas(bloqueado, fila0, col0, angulos, alcances_m, res, paso_celdas=0.5):
    angulos = np.asarray(angulos, dtype=float)
    alcances = np.asarray(alcances_m, dtype=float) / res        # en celdas
    if angulos.size == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    max_pasos = int(np.ceil(np.max(alcances) / paso_celdas)) + 1
    s = np.arange(max_pasos) * paso_celdas                       # distancia en celdas
    filas = np.floor(fila0 + 0.5 + np.outer(np.sin(angulos), s)).astype(int)
    cols = np.floor(col0 + 0.5 + np.outer(np.cos(angulos), s)).astype(int)
    alto, ancho = bloqueado.shape
    dentro = (filas >= 0) & (filas < alto) & (cols >= 0) & (cols < ancho)
    fc, cc = np.clip(filas, 0, alto - 1), np.clip(cols, 0, ancho - 1)
    choca = bloqueado[fc, cc] | ~dentro
    choca[:, 0] = False
    hay = choca.any(axis=1)
    primer = np.where(hay, np.argmax(choca, axis=1), max_pasos)
    idx = np.arange(max_pasos)[None, :]
    ok = (idx < primer[:, None]) & (s[None, :] <= alcances[:, None]) & dentro
    claves = np.unique(fc[ok] * ancho + cc[ok])
    return claves // ancho, claves % ancho
