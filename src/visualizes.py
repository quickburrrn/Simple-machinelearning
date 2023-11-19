import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pandas as pd

class visualizer:
    def update(self, i):
        try:
            df = pd.read_csv('data.csv')
            print(df)
            plt.scatter(df['X'], df['Y'], c=df['class'], s=40, cmap='brg')
        except Exception as e:
            print(f'failed to read from data.csv Exeption:{e}')


    def __init__(self):
        self.fig = plt.figure()
        self.ax1 = self.fig.add_subplot(1, 1, 1)

        ani = FuncAnimation(plt.gcf(), self.update, interval=1000, cache_frame_data=False)
        plt.show()


if __name__ == '__main__':
    pass
    vs = visualizer()
