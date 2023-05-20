from examples.explosion_multi.config import config
from pyhype.utils.visualizer import Vizualizer


if __name__ == "__main__":
    viz = Vizualizer(config=config)
    viz.plot(time_step=1600)
