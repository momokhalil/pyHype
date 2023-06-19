from pyhype.utils.visualizer import PlotSetup, Vizualizer
from examples.explosion_multi.config import config

setup = PlotSetup(
    timesteps=[100],
    size_x=10,
    size_y=10,
    x_lim=(0, 10),
    y_lim=(0, 20),
    variable="density"
)
viz = Vizualizer(config=config)
viz.plot(setup=setup)
