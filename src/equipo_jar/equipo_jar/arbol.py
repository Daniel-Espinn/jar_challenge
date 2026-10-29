'''
Sequence "corrida" (memory=True)
├── EsperarMapa
├── EsperarViewpoints
├── GuardarLargada(espera_s)
└── Selector "mision" (memory=False)
      ├── Sequence "volver" (memory=True)
      │     ├── HayQueVolver(velocidad, factor, margen, duración)
      │     ├── PrepararMeta(destino='largada', clave='meta_largada')
      │     ├── FromBlackboard(key='meta_largada')
      │     └── PublicarDone(publicar)
      └── Sequence "explorar" (memory=True)
            ├── QuedanViewpoints
            ├── PrepararMeta(destino='viewpoint', clave='meta_vp')
            ├── FailureIsSuccess( FromBlackboard(key='meta_vp') )
            └── AvanzarViewpoint
'''

import py_trees
import py_trees_ros
from nav2_msgs.action import NavigateToPose

from equipo_jar.comportamientos import (
    EsperarMapa, EsperarViewpoints, GuardarLargada, HayQueVolver,
    PublicarDone, QuedanViewpoints, AvanzarViewpoint, PrepararMeta)

def armar_arbol(estado, cfg, publicar_donde):
    '''Armar el arbol del cerebro
    estado: EstadoCorrida
    cfg: dict con espera_s, velocidad_ms, factor_cambio, margen_s, duracion_s,
    publicar_done: funcion sin args que publica /done
    '''
    # - rama volver
    hay_que_volver = HayQueVolver('Hay que volver?', estado
                                cfg['velocidad_ms'], cfg['factor_cambio'],
                                cfg['margen_s'], cfg['duracion_s'])

    meta_largada = PrepararMeta('Meta: largada', estado,
                                destino='largada', clave='meta_largada')
    
    # se lo manda a nav
    ir_largada = py_trees_ros.action_clients.FromBlackboard(
    name= 'Ir a a la largada', action_type=NavigateToPose,
    action_name='navigate_to_pose', key='meta_largada')

    # publica /done
    done = PublicarDone('Publicar /done', estado, publicar_done)

    volver = py_trees.composites.Sequence('volver', memory=True,
        children=[hay_que_volver, meta_largada, ir_largada, done])

    # - rama explorar
    quedan_vp = QuedanViewpoints('Quedan viewpoints?', estado)

    meta_vp = PrepararMeta('Meta: viewpoint', estado, destino='viewpoint', clave='meta_vp')

    ir_viewpoint = py_trees_ros.action_clients.FromBlackboard(
        name='Ir a viewpoint', action_type=NavigateToPose,
        action_name='navigate_to_pose')
    # mision y corrida