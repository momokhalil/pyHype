from pyhype.utils.visualizer import PlotSetup, Vizualizer
from examples.explosion_multi.config import config

setup = PlotSetup(
    timesteps=[300, 400, 500],
    size_x=10,
    size_y=10,
    x_lim=(0, 10),
    y_lim=(0, 20),
)
viz = Vizualizer(config=config)
viz.plot(setup=setup)
