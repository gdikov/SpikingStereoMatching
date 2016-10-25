###
# Date: August 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###

import os

class Retina(object):
    def __init__(self, label="Retina", dimension_x=1, dimension_y=1,
                 use_prerecorded_input=True, spike_times=None,
                 min_disparity=0, experiment_name=None,
                 record_spikes=False, verbose=False):
        assert len(spike_times) >= dimension_x and len(spike_times[0]) >= dimension_y, \
            "ERROR: Dimensionality of retina's spiking times is bad. Retina initialization failed."

        if min_disparity != 0:
            print "WARNING: The minimum disparity is not 0. " \
                  "This may lead to nonsensical results or failure as the network is not tested thoroughly."

        self.label = label
        self.experiment_name = experiment_name
        self.dim_x = dimension_x
        self.dim_y = dimension_y
        self.use_prerecorded_input = use_prerecorded_input

        if verbose:
            print "INFO: Creating Spike Source: {0}".format(label)

        import spynnaker.pyNN as ps

        self.pixel_columns = []
        self.labels = []

        if use_prerecorded_input or spike_times is not None:
            for x in range(0, dimension_x - min_disparity):
                retina_label = "{0}_{1}".format(label, x)
                col_of_pixels = ps.Population(dimension_y, ps.SpikeSourceArray, {'spike_times': spike_times[x]},
                                               label=retina_label, structure=ps.Line())

                self.pixel_columns.append(col_of_pixels)
                self.labels.append(retina_label)
                if record_spikes:
                    col_of_pixels.record()

        else:
            import spynnaker_external_devices_plugin.pyNN as sedp

            # constants and variables for the live injector mode
            init_portnum = 12000 if "l" in label.lower() else 13000
            remaining_pixel_cols = dimension_x
            MAX_INJNEURONS_IN_POPULATION = 255

            for x in range(0, int((dimension_x-min_disparity-1)/(MAX_INJNEURONS_IN_POPULATION/dimension_y))+1):
                retina_label = "{0}_{1}".format(label, x)

                # some hacks to pack the injector neurons densely in the populations (since they are limited!)
                if remaining_pixel_cols > int(MAX_INJNEURONS_IN_POPULATION / dimension_y):
                    col_height = int(MAX_INJNEURONS_IN_POPULATION / dimension_y) * dimension_y
                else:
                    col_height = remaining_pixel_cols * dimension_y

                col_of_pixels = ps.Population(col_height,
                                              sedp.SpikeInjector,
                                              {'port': init_portnum + x},
                                              label=retina_label)
                remaining_pixel_cols -= int(MAX_INJNEURONS_IN_POPULATION / dimension_y)

                self.pixel_columns.append(col_of_pixels)
                self.labels.append(retina_label)
                if record_spikes:
                    col_of_pixels.record()

    def get_spikes(self, sort_by_time=True, save_spikes=True):
        spikes_per_population = [x.getSpikes() for x in self.pixel_columns]
        spikes = list()
        for col_index, col in enumerate(spikes_per_population, 0):  # it is 0-indexed
            # for each spike in the population extract the timestamp and x,y coordinates
            x_coord = col_index
            for spike in col:
                y_coord = int(spike[0])
                spikes.append((round(spike[1], 1), x_coord+1, y_coord+1))	# pixel coordinates are 1-indexed
        if sort_by_time:
            spikes.sort(key=lambda x: x[0])
        if save_spikes:
            if not os.path.exists("./spikes"):
                os.makedirs("./spikes")
            i = 0
            while os.path.exists("./spikes/{0}_{1}_spikes_{2}.dat".format(self.experiment_name, i, self.label)):
                i += 1
            with open('./spikes/{0}_{1}_spikes_{2}.dat'.format(self.experiment_name, i, self.label), 'w') as fs:
                fs.write("### DATA FORMAT ###\n"
                        "# Description: These are the spikes a retina has produced (see file name for exact retina label).\n"
                        "# Each row contains: "
                        "Time stamp -- x-coordinate -- y-coordinate\n"
                        "### DATA START ###\n")
                for s in spikes:
                    fs.write(str(s[0]) + " " + str(s[1]) + " " + str(s[2]) + "\n")
                fs.write("### DATA END ###")
                fs.close()
        return spikes
