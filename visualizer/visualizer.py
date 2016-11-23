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

class Visualizer(object):

    def __init__(self, experiment_name="Experiment",
                 network_dimensions=None,
                 spikes_file="",
                 membrane_potential_file="",
                 verbose=False):
        self.experiment_name = experiment_name.replace(" ", "_")
        self.network_dimensions = network_dimensions

        self.network_spikes_file = spikes_file
        self.network_voltage_file = membrane_potential_file
        self.spikes = []

        if spikes_file:
            with open(self.network_spikes_file, 'rb') as events:
                is_data = False
                for line in events:
                    # skip preambles and other logged information
                    if not "DATA START" in line and not "DATA END" in line:
                        if not is_data:
                            continue
                        else:
                            l = line.split()
                            self.spikes.append((float(l[0]), int(l[1]), int(l[2]), int(l[3])))
                    else:
                        is_data = True
        else:
            print("WARNING: Network spikes output file is not set. No spike data to visualize.")
        # self.spikes = np.asarray(self.spikes)

        self.membrane_potential = {"bl": [], "br": [], "c": []}

        # NOTE: it is assumed that if the membrane potential is to be ploted, then only one microensemble is recorded
        if membrane_potential_file:
            with open(self.network_voltage_file, 'rb') as voltages:
                is_data = False
                for line in voltages:
                    # skip preambles and other logged information
                    if not "DATA START" in line and not "DATA END" in line:
                        if not is_data:
                            continue
                        else:
                            l = line.split()
                            if l[0] == 'b':
                                if l[2] == '0':
                                    self.membrane_potential["bl"].append((float(l[3]), float(l[4])))
                                else:
                                    self.membrane_potential["br"].append((float(l[3]), float(l[4])))
                            else:
                                self.membrane_potential["c"].append((float(l[3]), float(l[4])))
                    else:
                        is_data = True
        self.membrane_potential["br"] = np.asarray(self.membrane_potential["br"])
        self.membrane_potential["bl"] = np.asarray(self.membrane_potential["bl"])
        self.membrane_potential["c"] = np.asarray(self.membrane_potential["c"])

        self.scatter = None
        self.verbose = verbose

    def microensemble_voltage_plot(self, save_figure=True, show_interactive=False):

        fig = plt.figure()
        ax1 = fig.add_subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)

        ax1.plot(self.membrane_potential["bl"][: , 0], self.membrane_potential["bl"][:, 1])
        ax2.plot(self.membrane_potential["br"][:, 0], self.membrane_potential["br"][:, 1])
        ax3.plot(self.membrane_potential["c"][:, 0], self.membrane_potential["c"][:, 1])

        plt.subplots_adjust(hspace=1)

        ax1.set_title('Blocker left')
        ax2.set_title('Blocker right')
        ax3.set_title('Collector')

        # set common labels
        fig.text(0.5, 0.04, 'time in ms', ha='center', va='center')
        fig.text(0.06, 0.5, 'voltage in mV', ha='center', va='center', rotation='vertical')

        if save_figure:
            if not os.path.exists("./figures"):
                os.makedirs("./figures")
            i = 0
            while os.path.exists("./figures/{0}_voltage_{1}.png".format(self.experiment_name, i)):
                i += 1
            plt.savefig("./figures/{0}_voltage_{1}.png".format(self.experiment_name, i))
        if show_interactive:
            plt.show()

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

        if show_interactive:
            plt.show()

        if save_figure:
            if not os.path.exists("./figures"):
                os.makedirs("./figures")
            i = 0
            while os.path.exists("./figures/{0}_{1}.png".format(self.experiment_name, i)):
                i += 1
            plt.savefig("./figures/{0}_{1}.png".format(self.experiment_name, i))


        return [disps.count(c) for c in range(self.network_dimensions['min_d'], self.network_dimensions['max_d'])]

    def scatter_animation(self, dimension=3, save_animation=True, show_interactive=False, rotate=False):

        if dimension == 2:
            if rotate:
                print("WARNING: The rotate option is available only for the 3D animation. It will be ignored.")

        scatter = _Scatter(visualizer=self,
                           dimension=dimension,
                           relative_frame_length=0.05,
                           smoothness_factor=0.03,
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
                 relative_frame_length=0.01,
                 rotate=False, speedup_factor=1.0, smoothness_factor=0.5,
                 export_format='mp4'):

        self.visualizer = visualizer
        self.dimension=dimension

        self.relative_frame_length = relative_frame_length  # should be within (0.0, 1.0]

        self.duration = self.visualizer.spikes[-1][0]       # in ms
        self.angle = -90
        self.rotate = rotate
        self.speedup = speedup_factor


        self.scatter_plot = None

        self.window_size = self.duration * relative_frame_length            # some quantity in ms
        self.window_step = self.window_size * smoothness_factor      # some quantity in ms
        self.window_start = 0
        self.window_end = self.window_start + self.window_size

        self.frame_count = int((self.duration - self.window_size) / self.window_step) + 1
        self.fps = int((self.frame_count / (self.duration / 1000)) * self.speedup)

        if self.fps == 0:
            self.fps = 1

        # initialise figure and axes and setup other details
        self.cbar_not_added = True
        self._setup_plot()

        self.file_format = export_format
        # print("duration", int((self.duration/1000+1)/speedup_factor))
        from moviepy.editor import VideoClip
        self.anim = VideoClip(self._update_frame, duration=int((self.duration/1000+1)/speedup_factor))

    def _change_angle(self):
        self.angle = (self.angle + 0.5) % 360

    def _setup_plot(self):
        self.fig = plt.figure()
        if self.dimension == 3:
            self.ax = self.fig.add_subplot(111, projection='3d')
            self.ax.set_xlim3d(0, self.visualizer.network_dimensions['dim_x'])
            self.ax.set_xlabel('x')
            self.ax.set_ylim3d(self.visualizer.network_dimensions['max_d'], self.visualizer.network_dimensions['min_d'])
            self.ax.set_ylabel('depth')
            self.ax.set_zlim3d(0, self.visualizer.network_dimensions['dim_y'])
            self.ax.set_zlabel('y')

        else:
            self.ax = self.fig.add_subplot(111)
            self.ax.set_xlim(0, self.visualizer.network_dimensions['dim_x'])
            self.ax.set_xlabel('x')
            self.ax.set_ylim(0, self.visualizer.network_dimensions['dim_y'])
            self.ax.set_ylabel('y')

        self.ax.set_title("Network's subjective sense of reality", fontsize=16)
        self.ax.set_autoscale_on(False)

    def _update_frame(self, time):
        from moviepy.video.io.bindings import mplfig_to_npimage
        if self.dimension == 3:
            if self.rotate:
                self._change_angle()
            self.ax.view_init(30, self.angle)

        # clear the plot from the old data
        if self.scatter_plot is not None:
            self.scatter_plot.remove()

        # reimplement in a more clever way so that movie generation is in total O(n) not O(nm)
        current_frame = []
        for s in self.visualizer.spikes:
            if s[0] > self.window_end:
                self.window_start += self.window_step
                self.window_end = self.window_start + self.window_size
                break
            if s[0] > self.window_start:
                current_frame.append((s[1], s[2], s[3]))
        current_frame = np.asarray(current_frame)

        if current_frame.size > 0:

            # be careful of the coordinate-axes orientation. up/down should be x!
            if self.dimension == 3:
                self.scatter_plot = self.ax.scatter(current_frame[:, 0],
                                                    current_frame[:, 2],
                                                    current_frame[:, 1],
                                                    s=25,
                                                    marker='s',
                                                    lw=0,
                                                    c=current_frame[:, 2],
                                                    cmap=plt.cm.get_cmap("brg",
                                                                         self.visualizer.network_dimensions[
                                                                             'max_d'] + 1),
                                                    vmin=self.visualizer.network_dimensions['min_d'],
                                                    vmax=self.visualizer.network_dimensions['max_d']
                                                    )
                if self.cbar_not_added:
                    cbar = self.fig.colorbar(self.scatter_plot)
                    cbar.set_label('Perceived disparity in pixel units', rotation=270)
                    cbar.ax.get_yaxis().labelpad = 15
                    self.cbar_not_added = False
            else:
                self.scatter_plot = self.ax.scatter(current_frame[:, 0],
                                                    current_frame[:, 1],
                                                    s=25,
                                                    marker='s',
                                                    lw=0,
                                                    c=current_frame[:, 2],
                                                    cmap=plt.cm.get_cmap("brg",
                                                                         self.visualizer.network_dimensions['max_d']+1),
                                                    vmin=self.visualizer.network_dimensions['min_d'],
                                                    vmax=self.visualizer.network_dimensions['max_d'])
                # find some better solution than this flag hack
                if self.cbar_not_added:
                    cbar = self.fig.colorbar(self.scatter_plot)
                    cbar.set_label('Perceived disparity in pixel units', rotation=270)
                    cbar.ax.get_yaxis().labelpad = 15
                    self.cbar_not_added = False
        else:
            self.scatter_plot = self.ax.scatter([], [])

        return mplfig_to_npimage(self.fig)

    def show(self):
        plt.show()  # or better use plt.draw()?, not sure but I think this might freeze when updating...

    def save(self):
        dim = "3D" if self.dimension == 3 else "2D"

        if not os.path.exists("./animations"):
            os.makedirs("./animations")
        i = 0
        while os.path.exists("./animations/{0}_{2}_{1}.gif"
                                     .format(self.visualizer.experiment_name, i, dim)) or \
                os.path.exists("./animations/{0}_{2}_{1}.mp4"
                                       .format(self.visualizer.experiment_name, i, dim)):
            i += 1
        if self.file_format == 'gif':
            self.anim.write_gif(filename="./animations/{0}_{2}_{1}.gif"
                                .format(self.visualizer.experiment_name, i, dim),
                                fps=self.fps,
                                verbose=self.visualizer.verbose)
        elif self.file_format == 'mp4':
            print("INFO: Generating movie with duration of {0}s at {1}fps."
                  .format(int((self.duration/1000+1)/self.speedup), self.fps))
            self.anim.write_videofile(filename="./animations/{0}_{2}_{1}.mp4"
                                      .format(self.visualizer.experiment_name, i, dim),
                                      fps=self.fps,
                                      codec='mpeg4',
                                      bitrate='2000k',
                                      audio=False,
                                      verbose=self.visualizer.verbose)
        else:
            print("ERROR: The export format is not supported.")
