from examples.supersonic_step.config import config
from pyhype.utils.visualizer import Vizualizer, PlotSetup


if __name__ == "__main__":
    viz = Vizualizer(config=config)

    setup = PlotSetup(
        timesteps=list(range(10000, 11500 + 500, 500)),
        size_x=9 * 3,
        size_y=4 * 3,
        x_lim=(0.0, 9.0),
        y_lim=(0.0, 4.0),
    )
    viz.plot(setup=setup)
