"""Top-down telemetry visualization from Genesis poses (NOT a Genesis camera image)."""
from __future__ import annotations
from pathlib import Path
import numpy as np

from .core import OdorField, OBSTACLES, ROOM_HALF


def plot_trace(trace:list[dict], source_xy:tuple[float,float], wind_angle:float, path:Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    path.parent.mkdir(parents=True,exist_ok=True)
    field=OdorField(*source_xy,wind_angle=wind_angle)
    grid=np.linspace(-ROOM_HALF,ROOM_HALF,130)
    z=np.asarray([[field.at(x,y) for x in grid] for y in grid])
    fig,ax=plt.subplots(figsize=(7.5,6.8),dpi=145)
    ax.imshow(z,origin='lower',extent=[-ROOM_HALF,ROOM_HALF]*2,
              cmap='YlOrRd',vmin=0,vmax=1,alpha=.75)
    for x,y,sx,sy in OBSTACLES:
        ax.add_patch(Rectangle((x-sx/2,y-sy/2),sx,sy,
                               fc='#39465a',ec='white',lw=1.0))
    if trace:
        x=np.asarray([p['x'] for p in trace]);y=np.asarray([p['y'] for p in trace])
        ax.plot(x,y,c='#1676d2',lw=2.5,label='Genesis odometry')
        ax.scatter([x[0]],[y[0]],c='#1a60a4',s=70,marker='s',label='start')
        ax.scatter([x[-1]],[y[-1]],c='#083e70',s=75,marker='x',label='finish')
    ax.scatter([source_xy[0]],[source_xy[1]],marker='*',c='#ff2f36',s=250,
               edgecolor='white',linewidth=.7,label='hidden source (display only)')
    ax.set(xlim=(-ROOM_HALF,ROOM_HALF),ylim=(-ROOM_HALF,ROOM_HALF),
           xlabel='X / m',ylabel='Y / m',title='OlfaLink Scout | Genesis physical trajectory')
    ax.set_aspect('equal');ax.grid(alpha=.2)
    ax.legend(loc='lower right',fontsize=8,facecolor='white',framealpha=.85)
    fig.tight_layout();fig.savefig(path);plt.close(fig)
