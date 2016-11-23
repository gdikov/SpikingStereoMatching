###
# Date: August 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###

# import CooperativeNetwork as net
# import Retina as ret
# import ExternalInputReader as eir
# import Simulation as sim
import Visualizer as vis

if __name__ == "__main__":
    experiment_name = "Back_Front"
    experiment_duration = 10000  # in ms
    dx = 60                    # in pixels
    dy = 40                    # in pixels
    max_d = 20                 # in pixels
    crop_xmin = 45              # in pixels
    crop_ymin = 26              # in pixels

    # Setup the simulation
    # Simulation = sim.SNNSimulation(simulation_time=experiment_duration)

    # Define the input source
    # ExternalRetinaInput = eir.ExternalInputReader(file_path="input_data/Back_On_02_2xscaled.npz",
    #                                               dim_x=dx,
    #                                               dim_y=dy,
    #                                               crop_xmin=crop_xmin,
    #                                               crop_xmax=crop_xmin+dx,
    #                                               crop_ymin=crop_ymin,
    #                                               crop_ymax=crop_ymin+dy,
    #                                               sim_time=experiment_duration,
    #                                               is_rawdata_time_in_ms=False)

    # Create two instances of Retinas with the respective inputs
    # RetinaL = ret.Retina(label="RetL", dimension_x=dx, dimension_y=dy,
    #                      spike_times=ExternalRetinaInput.retinaLeft,
    #                      record_spikes=False,
    #                      experiment_name=experiment_name)
    # RetinaR = ret.Retina(label="RetR", dimension_x=dx, dimension_y=dy,
    #                      spike_times=ExternalRetinaInput.retinaRight,
    #                      record_spikes=False,
    #                      experiment_name=experiment_name)
#    RetinaL = ret.Retina(label="RetL", dimension_x=dx, dimension_y=dy,
#                         spike_times=[[1, 25, 26, 27, 40]])
#    RetinaR = ret.Retina(label="RetR", dimension_x=dx, dimension_y=dy,
#                         spike_times=[[9, 41]])

    # Create a cooperative network for stereo vision from retinal disparity
    # SNN_Network = net.CooperativeNetwork(retinae={'left': RetinaL, 'right': RetinaR},
    #                                      max_disparity=max_d,
    #                                      record_spikes=True,
    #                                      record_v=False,
    #                                      experiment_name=experiment_name)

    # Start the simulation
    # Simulation.run()

    # Store the results in a file
    # disparities = SNN_Network.get_spikes(sort_by_time=True, save_spikes=True)
#     ret_left_spikes = RetinaL.get_spikes(sort_by_time=True, save_spikes=True)
#     ret_right_spikes = RetinaR.get_spikes(sort_by_time=True, save_spikes=True)
#    membrane_potential = SNN_Network.get_v(save_v=True)
#     print(disparities)
#    print(membrane_potential)

    # Finish the simulation
    # Simulation.end()

    # Visualize the results (disparity histograms and 3D scatter animation)
    # network_dimensions = SNN_Network.get_network_dimensions()
    network_dimensions = {'dim_x':dx, 'dim_y':dy, 'min_d':0, 'max_d':max_d}
    Results = vis.Visualizer(network_dimensions=network_dimensions,
                             experiment_name=experiment_name,
                             spikes_file="./spikes/Back_Front_0_spikes.dat")
    # Results.microensemble_voltage_plot(save_figure=True)
    disps = Results.disparity_histogram(over_time=True, save_figure=True, show_interactive=True)
    print(sum(disps))
    for i in range(max_d):
        print(i, 100*disps[i]/float(sum(disps)))
    # Results.scatter_animation(dimension=3, save_animation=True, rotate=True)