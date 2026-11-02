REGRESAR = 'regresar'
INSPECCIONAR = 'inspeccionar'
EXPLORAR = 'explorar'


def decidir(t_corrida_s, duracion_s, reportes, n_victimas, reserva_s,
            inspeccion=None, cobertura=None, reinspeccion=None):
    # orden: todas reportadas > no llego a volver > mirar candidato > explorar > re-mirar > volver
    if reportes >= n_victimas:
        return REGRESAR, None, 'todas las víctimas reportadas'
    if duracion_s - t_corrida_s <= reserva_s:
        return REGRESAR, None, 'reserva de tiempo para volver'
    if inspeccion is not None:
        return INSPECCIONAR, inspeccion, 'objeto sin mirar'
    if cobertura is not None:
        return EXPLORAR, cobertura, 'piso sin ver'
    if reinspeccion is not None:
        return INSPECCIONAR, reinspeccion, 'mirar de nuevo desde otro lado'
    return REGRESAR, None, 'no queda nada por hacer'


def elegir_inspeccion(candidatos, punto_para_mirar_fn):
    mejor, mejor_pv, sin_punto = None, None, []
    for cand in candidatos:
        pv = punto_para_mirar_fn(cand)
        if pv is None:
            sin_punto.append(cand)
            continue
        if mejor is None or pv['camino'] < mejor_pv['camino']:
            mejor, mejor_pv = cand, pv
    return mejor, mejor_pv, sin_punto
