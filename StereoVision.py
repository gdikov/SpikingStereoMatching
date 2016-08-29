import CooperativeNetwork as net
import Retina as ret
import ExternalInputReader as eir
import Simulation as sim

if __name__ == "__main__":

    # Setup the simulation
    Simulation = sim.SNNSimulation(simulation_time=100)

    # Define the input source
    ExternalRetinaInput = \
        eir.ExternalInputReader(url="https://raw.githubusercontent.com/gdikov/"
                                "StereoMatching/master/Data/Input_Events/"
                                "Small_input_test.dat", dim_x=4, dim_y=4, sim_time=200)

    # Create two instances of Retinas with the respective inputs
    RetinaL = ret.Retina(label="RetL", dimension_x=4, dimension_y=4,
                         spike_times=ExternalRetinaInput.retinaLeft)
    RetinaR = ret.Retina(label="RetR", dimension_x=4, dimension_y=4,
                         spike_times=ExternalRetinaInput.retinaRight)

    # Create a cooperative network for stereo vision from retinal disparity
    SNN_Network = net.CooperativeNetwork(retinae={'left': RetinaL, 'right': RetinaR},
                                         max_disparity=3,
                                         record_spikes=True, visualize_spikes=True, experiment_name="Test")

    # Start the simulation
    Simulation.run()

    # Fetch the results and visualize them
    disparities = SNN_Network.get_spikes(sort_by_time=True, save_spikes=True)
    SNN_Network.visualizer.disparity_histogram(over_time=True, save_figure=True)

    # Finish the simulation
    Simulation.end()


