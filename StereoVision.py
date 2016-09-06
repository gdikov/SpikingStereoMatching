###
# Date: August 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###

import CooperativeNetwork as net
import Retina as ret
import ExternalInputReader as eir
import Simulation as sim
import Visualizer as vis

if __name__ == "__main__":

    experiment_name = "Test"
    experiment_duration = 10000
    dx = 60
    dy = 40
    max_d = 30

    # Setup the simulation
    Simulation = sim.SNNSimulation(simulation_time=experiment_duration)

    # Define the input source
    ExternalRetinaInput = \
        eir.ExternalInputReader(url="https://raw.githubusercontent.com/gdikov/"
                                "StereoMatching/master/Data/Input_Events/"
                                "NSTlogo_disp12-8-3.dat",
                                crop_window_x=dx,
                                crop_window_y=dy,
                                dim_x=dx,
                                dim_y=dy,
                                sim_time=experiment_duration)

    # Create two instances of Retinas with the respective inputs
    RetinaL = ret.Retina(label="RetL", dimension_x=dx, dimension_y=dy,
                         spike_times=ExternalRetinaInput.retinaLeft)
    RetinaR = ret.Retina(label="RetR", dimension_x=dx, dimension_y=dy,
                         spike_times=ExternalRetinaInput.retinaRight)

    # Create a cooperative network for stereo vision from retinal disparity
    SNN_Network = net.CooperativeNetwork(retinae={'left': RetinaL, 'right': RetinaR},
                                         max_disparity=max_d,
                                         record_spikes=True,
                                         experiment_name=experiment_name)

    # Start the simulation
    Simulation.run()

    # Store the results in a file
    disparities = SNN_Network.get_spikes(sort_by_time=True, save_spikes=True)

    # Finish the simulation
    Simulation.end()

    # Visualize the results (disparity histograms and 3D scatter animation)
    # network_dimensions = SNN_Network.get_network_dimensions()
    # Results = vis.Visualizer(network_dimensions=network_dimensions,
    #                          experiment_name=experiment_name,
    #                          spikes_file="./spikes/{0}_0.dat".format(experiment_name))
    # Results.disparity_histogram(over_time=True, save_figure=True)
    # Results.scatter_animation(dimension=2, save_animation=True, rotate=False)




