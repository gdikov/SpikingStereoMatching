###
# Date: August 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###

class Retina(object):
    def __init__(self, label="Retina", dimension_x=1, dimension_y=1,
                 use_prerecorded_input=True, spike_times=None,
                 min_disparity=0, record_spikes=False, verbose=False):
        assert len(spike_times) >= dimension_x and len(spike_times[0]) >= dimension_y, \
            "ERROR: Dimensionality of retina's spiking times is bad. Retina initialization failed."

        if min_disparity != 0:
            print "WARNING: The minimum disparity is not 0. " \
                  "This may lead to nonsensical results or failure as the network is not tested thoroughly."

        self.label = label

        self.dim_x = dimension_x
        self.dim_y = dimension_y
        self.use_prerecorded_input = use_prerecorded_input

        if verbose:
            print "INFO: Creating Spike Source: {0}".format(label)

        import spynnaker.pyNN as ps

        self.pixel_columns = []
        self.labels = []

        if use_prerecorded_input:
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

