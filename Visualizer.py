###
# Date: September 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###


import matplotlib
matplotlib.use("Agg")	# needed for the ssh connection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os

from moviepy.editor import VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage

class Visualizer(object):

    def __init__(self, experiment_name="Experiment",
                 network_dimensions=None,
                 spikes_file="",
                 verbose=False):
        self.experiment_name = experiment_name
        self.network_dimensions = network_dimensions

        self.network_output_file = spikes_file
        self.spikes = []
        if spikes_file:
            with open(self.network_output_file, 'rb') as events:
                for line in events:
                    l = line.split()
                    self.spikes.append((float(l[0]), int(l[1]), int(l[2]), int(l[3])))
        else:
            print("WARNING: Network output file is not set. No data to visualize.")

        self.scatter = None
        self.verbose = verbose



    def disparity_histogram(self, over_time=False, save_figure=True, show_interactive=False):
        if self.verbose:
            print("WARNING: The disparity histogram plotter assumes that the spikes are already sorted by time."
                  "Otherwise, the results might be false.")
        if not over_time:
            spikes_per_disparity = self.spikes
            plt.bar(range(0, self.network_dimensions['max_d'] - self.network_dimensions['min_d'] + 1),
                    spikes_per_disparity, align='center')
        else:
            disps = [x[3] for x in self.spikes]

            x = range(0, len(disps))
            y = disps

            heatmap, xedges, yedges = np.histogram2d(y, x, bins=(self.network_dimensions['max_d'], 100))
            # extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

            ax = plt.figure().add_subplot(111)
            im = ax.imshow(heatmap,
                           extent=[0, 10, 0, self.network_dimensions['max_d']],
                           aspect=0.4,
                           interpolation='none',
                           origin='lower')
            ax.set_xlabel("Time in s")
            ax.set_ylabel("Disparity")
            ax.set_aspect(0.5)

            cbar = plt.colorbar(im, fraction=0.046, pad=0.03)
            cbar.set_label('Number of events per time slot (0.1 s)', rotation=270)
            cbar.ax.get_yaxis().labelpad = 15

        if save_figure:
            if not os.path.exists("./figures"):
                os.makedirs("./figures")
            i = 0
            while os.path.exists("./figures/{0}_{1}.png".format(self.experiment_name, i)):
                i += 1
            plt.savefig("./figures/{0}_{1}.png".format(self.experiment_name, i))
        if show_interactive:
            plt.show()

    def scatter_animation(self, dimension=3, save_animation=True, show_interactive=False, rotate=False):

        if dimension == 2:
            if rotate:
                print("WARNING: The rotate option is available only for the 3D animation. It will be ignored.")

        scatter = _Scatter(visualizer=self,
                           dimension=dimension,
                           relative_frame_length=0.01,
                           speedup_factor=1.0,
                           rotate=rotate,
                           export_format='mp4')

        if show_interactive:
            scatter.show()
        if save_animation:
            scatter.save()


class _Scatter(object):
    """relative frame length is the length of each single frame measured as a fraction of
    the whole experiment duration. """

    def __init__(self, visualizer=None,
                 dimension=3,
                 relative_frame_length=1.0,
                 rotate=False, speedup_factor=10.0,
                 export_format='gif'):

        self.visualizer = visualizer
        self.dimension=dimension
        self.relative_frame_length = relative_frame_length  # should be within (0.0, 1.0]
        self.frame_count = int(1 / self.relative_frame_length)
        self.angle = 0
        self.rotate = rotate
        self.speedup = speedup_factor
        self.scatter_plot = None
        self.data = self._prepare_frames()

        # initialise figure and axes and setup other details
        self._setup_plot()

        self.file_format = export_format

        self.anim = VideoClip(self._update_frame, duration=self.frame_count)

    def _change_angle(self):
        self.angle = (self.angle + 1) % 360

    def _setup_plot(self):
        self.fig = plt.figure()
        if self.dimension == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim3d(0, self.visualizer.network_dimensions['dim_x'])
            self.ax.set_xlabel('depth')
            self.ax.set_ylim3d(0, self.visualizer.network_dimensions['dim_y'])
            self.ax.set_ylabel('x')
            self.ax.set_zlim3d(self.visualizer.network_dimensions['min_d'], self.visualizer.network_dimensions['max_d'])
            self.ax.set_zlabel('y')

        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.visualizer.network_dimensions['dim_x'])
            self.ax.set_xlabel('x')
            self.ax.set_ylim(0, self.visualizer.network_dimensions['dim_y'])
            self.ax.set_ylabel('y')

        self.ax.set_title("Network's subjective sense of reality", fontsize=16)
        self.ax.set_autoscale_on(False)

    def _prepare_frames(self):
        # self.spikes[-1][0]/duration < 1 ! to prevent an index out of bound exception
        duration = float(self.visualizer.spikes[-1][0]) + 0.1
        print("Duration", duration)
        framed_data = [[] for _ in range(self.frame_count)]

        # the data is formatted like:
        # [(timestamp, x, y, disparity)]
        for d in self.visualizer.spikes:
            framed_data[int((d[0] / duration) * self.frame_count) - 1].append((d[1], d[2], d[3]))
        print len(framed_data), framed_data
        framed_data = [np.asarray(x) for x in framed_data]
        return framed_data

    def _update_frame(self, time):
        i = int(time)
        # self.ax.clear()
        if self.dimension == 3:
            if self.rotate:
                self._change_angle()
                self.ax.view_init(30, self.angle)

        if self.data[i].size > 0:
            if self.scatter_plot is not None:
                self.scatter_plot.remove()
            # be careful of the coordinate-axes orientation. up/down should be x!
            if self.dimension == 3:
                self.scatter_plot = self.ax.scatter(self.data[i][:, 2],
                                                    self.data[i][:, 0],
                                                    self.data[i][:, 1],
                                                    s=15)
            else:
                self.scatter_plot = self.ax.scatter(self.data[i][:, 0],
                                                    self.data[i][:, 1],
                                                    s=15,
                                                    c=self.data[i][:, 2],
                                                    cmap=plt.cm.get_cmap("brg",
                                                                         self.visualizer.network_dimensions['max_d']+1))

        return mplfig_to_npimage(self.fig)

    def show(self):
        plt.show()  # or better use plt.draw()?, not sure but I think this might freeze when updating...

    def save(self):
        dim = "3D" if self.dimension == 3 else "2D"

        if not os.path.exists("./animations"):
            os.makedirs("./animations")
        i = 0
        while os.path.exists("./animations/{0}_{2}_{1}.gif".format(self.visualizer.experiment_name, i, dim)) or \
                os.path.exists("./animations/{0}_{2}_{1}.mp4".format(self.visualizer.experiment_name, i, dim)):
            i += 1
        if self.file_format == 'gif':
            self.anim.write_gif(filename="./animations/{0}_{2}_{1}.gif".format(self.visualizer.experiment_name, i, dim),
                                fps=int(15/self.speedup),
                                verbose=self.visualizer.verbose)
        elif self.file_format == 'mp4':
            self.anim.write_videofile(filename="./animations/{0}_{2}_{1}.mp4".format(self.visualizer.experiment_name, i, dim),
                                      fps=int(15/self.speedup),
                                      codec='mpeg4',
                                      audio=False,
                                      verbose=self.visualizer.verbose)
        else:
            print("ERROR: The export format is not supported.")