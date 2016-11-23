###
# Date: August 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###

import spynnaker.pyNN as ps

class SNNSimulation(object):
    def __init__(self, simulation_time=1000, simulation_time_step=0.2, threads_count=4):
        self.simulation_time = simulation_time
        self.time_step = simulation_time_step
        # setup timestep of simulation and minimum and maximum synaptic delays
        ps.setup(timestep=simulation_time_step,
                 min_delay=simulation_time_step,
                 max_delay=10*simulation_time_step,
                 threads=threads_count)

    def run(self):
        # run simulation for time in milliseconds
        ps.run(self.simulation_time)

    def end(self):
        # finalise program and simulation
        ps.end()