from equipo_jar.reglas_gestor import elegir_relleno, esta_confirmada


def cand(xs, ys, reportada=False):
    return {'xs': list(xs), 'ys': list(ys), 'reportada': reportada}


def test_confirma_con_avisos_juntos():
    c = cand([1.000, 1.001, 0.999, 1.000, 1.001, 1.000, 0.999], [2.0] * 7)
    assert esta_confirmada(c, min_avisos=5, disp_max=0.10)


def test_no_confirma_con_pocos_avisos():
    c = cand([1.0, 1.0, 1.0], [2.0, 2.0, 2.0])
    assert not esta_confirmada(c, min_avisos=5, disp_max=0.10)


def test_no_confirma_si_estan_desparramados():
    c = cand([1.0, 1.3, 0.7, 1.4, 0.6, 1.25], [2.0, 2.3, 1.7, 2.4, 1.6, 2.25])
    assert not esta_confirmada(c, min_avisos=5, disp_max=0.10)


def test_relleno_prioriza_no_reportadas_con_mas_avisos():
    pizarron = [
        cand([0.0] * 9, [0.0] * 9, reportada=True),
        cand([2.0] * 2, [2.0] * 2),          # 2 avisos
        cand([4.0] * 3, [4.0] * 3),          # 3 avisos -> primero
    ]
    salida = elegir_relleno(pizarron, sugerencias=[(6.0, 6.0)], quedan=2, radio_dup=0.6)
    assert salida == [(4.0, 4.0), (2.0, 2.0)]


def test_relleno_usa_sugerencias_y_evita_duplicados():
    pizarron = [cand([0.0] * 9, [0.0] * 9, reportada=True)]
    sugerencias = [(0.3, 0.0), (3.0, 3.0), (3.2, 3.0), (5.0, 1.0)]
    salida = elegir_relleno(pizarron, sugerencias, quedan=5, radio_dup=0.6)
    assert salida == [(3.0, 3.0), (5.0, 1.0)]


def test_relleno_no_supera_lo_que_queda():
    salida = elegir_relleno([], [(1.0, 1.0), (3.0, 3.0)], quedan=1, radio_dup=0.6)
    assert salida == [(1.0, 1.0)]
