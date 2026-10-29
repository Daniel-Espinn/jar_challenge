import py_trees
from py_trees.common import Status

from equipo_jar.reglas_cerebro import (
    tiempo_restante, distancia, calcular_reserva, hay_que_volver, yaw_a_cuaternion)
from nav2_msgs.action import NavigateToPose

class EsperarMapa(py_trees.behaviour.Behaviour):
    '''RUNNING hasta que arranco la corrida cuando haya hora de inicio'''
    def __init__(self, name, estado):
        super().__init__(name)
        self.estado = estado

    def update(self):
        if self.estado.t_inicio is None:
            return Status.RUNNING
        return Status.SUCCESS

class EsperarViewpoints(py_trees.behaviour.Behaviour):
    '''RUNNING hasta que llego la lista de viewpoints'''
    def __init__(self, name, estado):
        super().__init__(name)
        self.estado = estado

    def update(self):
        if len(self.estado.viewpoints) == 0:
            return Status.RUNNING
        return Status.SUCCESS

class GuardarLargada(py_trees.behaviour.Behaviour):
    ''' Espera 'espera_s' desde el inicio (convergencia) y anota la pose como largada'''

    def __init__(self, name, estado, espera_s):
        super().__init__(name)
        self.estado = estado
        self.espera_s = espera_s

    def update(self):
        segundos = self.estado.ahora - self.estado.t_inicio
        if segundos < self.espera_s or self.estado.pose is None:
            return Status.RUNNING #convergiendo
        self.estado.largada = self.estado.pose
        self.logger.info(f"Largada guardada en {self.estado.largada}")
        return Status.SUCCESS

class QuedanViewpoints(py_trees.behaviour.Behaviour):
    '''SUCCESS si todavia hay viewpoints por visitar'''
    def __init__(self, name, estado):
        super().__init__(name)
        self.estado = estado

    def update(self):
        if self.estado.indice_vp < len(self.estado.viewpoints):
            return Status.SUCCESS
        return Status.FAILURE
    
class AvanzarViewpoint(py_trees.behaviour.Behaviour):
    def __init__(self, name, estado):
        super().__init__(name)
        self.estado = estado
    
    def update(self):
        self.estado.indice_vp += 1
        return Status.SUCCESS

class HayQueVolver(py_trees.behaviour.Behaviour):
    ''' SUCCESS si hay que volver ya (tiempo, victimas completas o sin viewpoints)'''
    def __init__(self, name, estado, velocidad_ms, factor_cambio, margen_s, duracion_s):
        super().__init__(name)
        self.estado = estado
        self.velocidad_ms = velocidad_ms
        self.factor_cambio = factor_cambio
        self.margen_s = margen_s
        self.duracion_s = duracion_s
        
    def update(self):
        e = self.estado
        if e.pose is None or e.largada is None:
            return Status.FAILURE # sin pose no se puede hacer nada, seguimos asi
        t_restante = tiempo_restante(e.t_inicio, e.ahora, self.duracion_s)
        dist = distancia(e.pose, e.largada)
        reserva = calcular_reserva(dist, self.velocidad_ms, self.factor_cambio, self.margen_s)
        quedan_vp = e.indice_vp < len(e.viewpoints)
        
        if hay_que_volver(t_restante, reserva, e.reportes_hechos, e.n_victimas, quedan_vp):
            return Status.SUCCESS
        return Status.FAILURE

class PublicarDone(py_trees.behaviour.Behaviour):
    ''' Publica /done una sola vez, usando la funcion 'publicar' que
    le pasa el nodo'''
    def __init__(self, name, estado, publicar):
        super().__init__(name)
        self.estado = estado
        self.publicar = publicar

    def update(self):
        if not self.estado.done_publicado:
            self.publicar()
            self.estado.done_publicado = True
        return Status.SUCCESS

class PrepararMeta(py_trees.behaviour.Behaviour):
    '''Arma la meta de Nav2 (viewpoint)'''
    def __init__(self,name, estado, destino, clave):
        super().__init__(name)
        self.estado = estado
        self.destino = destino
        self.clave = clave
        self.bb = self.attach_blackboard_client(name=name)
        self.bb.register_key(key=clave, access=py_trees.common.Access.WRITE)

    def update(self):
        #1. Pose destino
        if self.destino == 'largada':
            p = self.estado.largada
        else:
            p = self.estado.viewpoints[self.estado.indice_vp]

        # mensaje
        meta = NavigateToPose.Goal()
        meta.pose.header.frame_id = 'map'
        meta.pose.pose.position.x = float(p.x)
        meta.pose.pose.position.y = float(p.y)
        z, w = yaw_a_cuaternion(p.yaw)
        meta.pose.pose.orientation.z = z
        meta.pose.pose.orientation.w = w

        # al blackboard
        self.bb.set(self.clave, meta)
        return Status.SUCCESS

