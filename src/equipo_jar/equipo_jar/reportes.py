# formato oficial del reporte (7.2)
import math

from geometry_msgs.msg import PointStamped

FRAME_REPORTES = 'map'


def armar_reporte(x, y, ahora):
    if not (math.isfinite(x) and math.isfinite(y)):
        raise ValueError(f'reporte con valores no finitos: x={x}, y={y}')
    msg = PointStamped()
    msg.header.frame_id = FRAME_REPORTES
    msg.header.stamp = ahora
    msg.point.x = float(x)
    msg.point.y = float(y)
    msg.point.z = 0.0
    return msg
