###
# Date: August 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###

class Retina(object):
    def __init__(self, label="Retina", dimension_x=1, dimension_y=1, spike_times=None,
                 min_disparity=0, record_spikes=False, verbose=False):
        assert len(spike_times) >= dimension_x and len(spike_times[0]) >= dimension_y, \
            "ERROR: Dimensionality of retina's spiking times is bad. Retina initialization failed."

        if min_disparity != 0:
            print "WARNING: The minimum disparity is not 0. " \
                  "This may lead to nonsensical results or failure as the network is not tested thoroughly."

        self.dim_x = dimension_x
        self.dim_y = dimension_y

        if verbose:
            print "INFO: Creating Spike Source: {0}".format(label)

        import pyNN.spiNNaker as ps

        self.pixel_columns = []
        for x in range(0, dimension_x - min_disparity):
            col_of_pixels = ps.Population(dimension_y, ps.SpikeSourceArray, {'spike_times': spike_times[x]},
                                       label="{0} - Population {1}".format(label, x), structure=ps.Line())
            self.pixel_columns.append(col_of_pixels)
            if record_spikes:
                col_of_pixels.record()
