###
# Date: August 2016
# Author: Georgi Dikov
# email: gvdikov93@gmail.com
###

import spynnaker.pyNN as ps
import numpy as np
import os

class CooperativeNetwork(object):

    def __init__(self, retinae=None,
                 max_disparity=0, cell_params=None,
                 record_spikes=True, record_v=False, experiment_name="Experiment",
                 verbose=False):
        # IMPORTANT NOTE: This implementation assumes min_disparity = 0

        assert retinae['left'] is not None and retinae['right'] is not None, \
            "ERROR: Retinas are not initialised! Creating Network Failed."

        dx = retinae['left'].dim_x
        assert dx > max_disparity >= 0, "ERROR: Maximum Disparity Constant is illegal!"
        self.max_disparity = max_disparity
        self.min_disparity = 0
        self.size = (2 * (dx - self.min_disparity) * (self.max_disparity - self.min_disparity + 1)
                     - (self.max_disparity - self.min_disparity + 1) ** 2
                     + self.max_disparity - self.min_disparity + 1) / 2
        self.dim_x = dx
        self.dim_y = retinae['left'].dim_y

        # check this assertion before the actual network generation, since the former
        # might take very long to complete.
        assert retinae['left'].dim_x == retinae['right'].dim_x and \
               retinae['left'].dim_y == retinae['right'].dim_y, \
            "ERROR: Left and Right retina dimensions are not matching. Connecting Spike Sources to Network Failed."

        # TODO: make parameter values dependent on the simulation time step
        # (for the case 0.1 it is not tested completely and should serve more like an example)

        # the notation for the synaptic parameters is as follows:
        # B blocker, C collector, S spike source, (2, 4)
        # w weight, d delay, (1)
        # a one's own, z other, (3)
        # i inhibition, e excitation  (5)
        # If B is before C than the connection is from B to C.
        # Example: dSaB would mean a dealy from a spike source to the one's own blocker neuron, and
        # wSzB would be the weight from a spike source to the heterolateral blocker neuron.
        params = {'neural': dict(), 'synaptic': dict(), 'topological': dict()}
        simulation_time_step = 0.2
        if simulation_time_step == 0.2:
            params['neural'] = {'tau_E': 2.0,
                                'tau_I': 2.0,
                                'tau_mem': 2.07,
                                'v_reset_blocker': -84.0,
                                'v_reset_collector': -90.0}
            params['synaptic'] = {'wBC': 20.5,  #-20.5: negative won't work. However keep in mind that it is inhibitory!
                                  'dBC': simulation_time_step,
                                  'wSC': 20.5,
                                  'dSC': 1.6,
                                  'wSaB': 22.5,
                                  'dSaB': simulation_time_step,
                                  'wSzB': 22.5,    # same story here
                                  'dSzB': simulation_time_step,
                                  'wCCi': 50.0,    # and again
                                  'dCCi': simulation_time_step,
                                  'wCCe': 3.0,
                                  'dCCe': simulation_time_step}
            params['topological'] = {'radius_e': 1,
                                     'radius_i': max(self.dim_x, self.dim_y)}
        elif simulation_time_step == 0.1:
            params['neural'] = {'tau_E': 1.0,
                                'tau_I': 1.0,
                                'tau_mem': 1.07,
                                'v_reset_blocker': -92.0,
                                'v_reset_collector': -102.0}
            params['synaptic'] = {'wBC': 39.5, #weight should be positive numbers, indicated as inhibitory synapses (if necessary)!
                                  'dBC': simulation_time_step,
                                  'wSC': 39.5,
                                  'dSC': 0.8,
                                  'wSaB': 49.5,
                                  'dSaB': simulation_time_step,
                                  'wSzB': 39.5, # same here
                                  'dSzB': simulation_time_step,
                                  'wCCi': 50.0, # and here
                                  'dCCi': simulation_time_step,
                                  'wCCe': 3.0,
                                  'dCCe': simulation_time_step}
            params['topological'] = {'radius_e': 1,
                                     'radius_i': max(self.dim_x, self.dim_y)}

        self.cell_params = params if cell_params is None else cell_params

        self.network = self._create_network(record_spikes=record_spikes,
                                            record_v=record_v,
                                            verbose=verbose)

        self._connect_spike_sources(retinae=retinae, verbose=verbose)

        self.experiment_name = experiment_name.replace(" ", "_")

    def _create_network(self, record_spikes=False, record_v=False, verbose=False):

        if verbose:
            print("INFO: Creating Cooperative Network of size {0}".format(self.size))

        if record_spikes:
            from pyNN.spiNNaker import record

        network = []
        neural_params = self.cell_params['neural']
        for x in range(0, self.size):
            blocker_columns = ps.Population(self.dim_y * 2,
                                            ps.IF_curr_exp,
                                            {'tau_syn_E': neural_params['tau_E'],
                                             'tau_syn_I': neural_params['tau_I'],
                                             'tau_m': neural_params['tau_mem'],
                                             'v_reset': neural_params['v_reset_blocker']},
                                            label="Blocker {0}".format(x))

            collector_column = ps.Population(self.dim_y,
                                             ps.IF_curr_exp,
                                             {'tau_syn_E': neural_params['tau_E'],
                                              'tau_syn_I': neural_params['tau_I'],
                                              'tau_m': neural_params['tau_mem'],
                                              'v_reset': neural_params['v_reset_collector']},
                                             label="Collector {0}".format(x))

            if record_spikes:
                collector_column.record()  # records only the spikes
            if record_v:
                collector_column.record_v()  # records the membrane potential -- very resource demanding!
                blocker_columns.record_v()

            network.append((blocker_columns, collector_column))

        self._interconnect_neurons(network, verbose=verbose)
        if self.dim_x > 1:
            self._interconnect_neurons_inhexc(network, verbose)
        else:
            global _retina_proj_l, _retina_proj_r, same_disparity_indices
            _retina_proj_l = [[0]]
            _retina_proj_r = [[0]]
            same_disparity_indices = [[0]]
            
        return network

    def _interconnect_neurons(self, network, verbose=False):

        assert network is not None, \
            "ERROR: Network is not initialised! Interconnecting failed."

        synaptic_params = self.cell_params['synaptic']

        # generate connectivity list: 0 untill dimensionRetinaY-1 for the left
        # and dimensionRetinaY till dimensionRetinaY*2 - 1 for the right
        connList = []
        for y in range(0, self.dim_y):
            connList.append((y, y, synaptic_params['wBC'], synaptic_params['dBC']))
            connList.append((y + self.dim_y, y, synaptic_params['wBC'], synaptic_params['dBC']))

        # connect the inhibitory neurons to the cell output neurons
        if verbose:
            print "INFO: Interconnecting Neurons. This may take a while."
        for ensemble in network:
            ps.Projection(ensemble[0], ensemble[1], ps.FromListConnector(connList), target='inhibitory')

    def _interconnect_neurons_inhexc(self, network, verbose=False):

        assert network is not None, \
            "ERROR: Network is not initialised! Interconnecting for inhibitory and excitatory patterns failed."

        if verbose and self.cell_params['topological']['radius_i'] < self.dim_x:
            print "WARNING: Bad radius of inhibition. Uniquness constraint cannot be satisfied."
        if verbose and 0 <= self.cell_params['topological']['radius_e'] > self.dim_x:
            print "WARNING: Bad radius of excitation. "

        # create lists with inhibitory along the Retina Right projective line
        nbhoodInhL = []
        nbhoodInhR = []
        nbhoodExcX = []
        nbhoodEcxY = []
        # used for the triangular form of the matrix in order to remain within the square
        if verbose:
            print "INFO: Generating inhibitory and excitatory connectivity patterns."
        # generate rows
        limiter = self.max_disparity - self.min_disparity + 1
        ensembleIndex = 0

        while ensembleIndex < len(network):
            if ensembleIndex / (self.max_disparity - self.min_disparity + 1) > \
                                    (self.dim_x - self.min_disparity) - (self.max_disparity - self.min_disparity) - 1:
                limiter -= 1
                if limiter == 0:
                    break
            nbhoodInhL.append([ensembleIndex + disp for disp in range(0, limiter)])
            ensembleIndex += limiter

        ensembleIndex = len(network)

        # generate columns
        nbhoodInhR = [[x] for x in nbhoodInhL[0]]
        shiftGlob = 0
        for x in nbhoodInhL[1:]:
            shiftGlob += 1
            shift = 0

            for e in x:
                if (shift + 1) % (self.max_disparity - self.min_disparity + 1) == 0:
                    nbhoodInhR.append([e])
                else:
                    nbhoodInhR[shift + shiftGlob].append(e)
                shift += 1

        # generate all diagonals
        for diag in map(None, *nbhoodInhL):
            sublist = []
            for elem in diag:
                if elem is not None:
                    sublist.append(elem)
            nbhoodExcX.append(sublist)

        # generate all y-axis excitation
        for x in range(0, self.dim_y):
            for e in range(1, self.cell_params['topological']['radius_e'] + 1):
                if x + e < self.dim_y:
                    nbhoodEcxY.append(
                        (x, x + e, self.cell_params['synaptic']['wCCe'], self.cell_params['synaptic']['dCCe']))
                if x - e >= 0:
                    nbhoodEcxY.append(
                        (x, x - e, self.cell_params['synaptic']['wCCe'], self.cell_params['synaptic']['dCCe']))

        # Store these lists as global parameters as they can be used to quickly match the spiking collector neuron
        # with the corresponding pixel xy coordinates (same_disparity_indices)
        # TODO: think of a better way to encode pixels: closed form formula would be perfect
        # These are also used when connecting the spike sources to the network! (retina_proj_l, retina_proj_r)

        global _retina_proj_l, _retina_proj_r, same_disparity_indices

        _retina_proj_l = nbhoodInhL
        _retina_proj_r = nbhoodInhR
        same_disparity_indices = nbhoodExcX

        if verbose:
            print "INFO: Connecting neurons for internal excitation and inhibition."

        for row in nbhoodInhL:
            for pop in row:
                for nb in row:
                    if nb != pop:
                        ps.Projection(network[pop][1],
                                      network[nb][1],
                                      ps.OneToOneConnector(weights=self.cell_params['synaptic']['wCCi'],
                                                           delays=self.cell_params['synaptic']['dCCi']),
                                      target='inhibitory')
        for col in nbhoodInhR:
            for pop in col:
                for nb in col:
                    if nb != pop:
                        ps.Projection(network[pop][1],
                                      network[nb][1],
                                      ps.OneToOneConnector(weights=self.cell_params['synaptic']['wCCi'],
                                                           delays=self.cell_params['synaptic']['dCCi']),
                                      target='inhibitory')

        for diag in nbhoodExcX:
            for pop in diag:
                for nb in range(1, self.cell_params['topological']['radius_e'] + 1):
                    if diag.index(pop) + nb < len(diag):
                        ps.Projection(network[pop][1],
                                      network[diag[diag.index(pop) + nb]][1],
                                      ps.OneToOneConnector(weights=self.cell_params['synaptic']['wCCe'],
                                                           delays=self.cell_params['synaptic']['dCCe']),
                                      target='excitatory')
                    if diag.index(pop) - nb >= 0:
                        ps.Projection(network[pop][1],
                                      network[diag[diag.index(pop) - nb]][1],
                                      ps.OneToOneConnector(weights=self.cell_params['synaptic']['wCCe'],
                                                           delays=self.cell_params['synaptic']['dCCe']),
                                      target='excitatory')

        for ensemble in network:
            ps.Projection(ensemble[1], ensemble[1], ps.FromListConnector(nbhoodEcxY), target='excitatory')

    def _connect_spike_sources(self, retinae=None, verbose=False):

        if verbose:
            print "INFO: Connecting Spike Sources to Network."

        global _retina_proj_l, _retina_proj_r

        # left is 0--dimensionRetinaY-1; right is dimensionRetinaY--dimensionRetinaY*2-1
        connListRetLBlockerL = []
        connListRetLBlockerR = []
        connListRetRBlockerL = []
        connListRetRBlockerR = []
        for y in range(0, self.dim_y):
            connListRetLBlockerL.append((y, y,
                                         self.cell_params['synaptic']['wSaB'],
                                         self.cell_params['synaptic']['dSaB']))
            connListRetLBlockerR.append((y, y + self.dim_y,
                                         self.cell_params['synaptic']['wSzB'],
                                         self.cell_params['synaptic']['dSzB']))
            connListRetRBlockerL.append((y, y,
                                         self.cell_params['synaptic']['wSzB'],
                                         self.cell_params['synaptic']['dSzB']))
            connListRetRBlockerR.append((y, y + self.dim_y,
                                         self.cell_params['synaptic']['wSaB'],
                                         self.cell_params['synaptic']['dSaB']))

        retinaLeft = retinae['left'].pixel_columns
        retinaRight = retinae['right'].pixel_columns
        pixel = 0
        for row in _retina_proj_l:
            for pop in row:
                ps.Projection(retinaLeft[pixel],
                              self.network[pop][1],
                              ps.OneToOneConnector(weights=self.cell_params['synaptic']['wSC'],
                                                   delays=self.cell_params['synaptic']['dSC']),
                              target='excitatory')
                ps.Projection(retinaLeft[pixel],
                              self.network[pop][0],
                              ps.FromListConnector(connListRetLBlockerL),
                              target='excitatory')
                ps.Projection(retinaLeft[pixel],
                              self.network[pop][0],
                              ps.FromListConnector(connListRetLBlockerR),
                              target='inhibitory')
            pixel += 1

        pixel = 0
        for col in _retina_proj_r:
            for pop in col:
                ps.Projection(retinaRight[pixel], self.network[pop][1],
                              ps.OneToOneConnector(weights=self.cell_params['synaptic']['wSC'],
                                                   delays=self.cell_params['synaptic']['dSC']),
                              target='excitatory')
                ps.Projection(retinaRight[pixel],
                              self.network[pop][0],
                              ps.FromListConnector(connListRetRBlockerR),
                              target='excitatory')
                ps.Projection(retinaRight[pixel],
                              self.network[pop][0],
                              ps.FromListConnector(connListRetRBlockerL),
                              target='inhibitory')
            pixel += 1

        # configure for the live input streaming if desired
        if not(retinae['left'].use_prerecorded_input and retinae['right'].use_prerecorded_input):
            from spynnaker_external_devices_plugin.pyNN.connections.spynnaker_live_spikes_connection import \
                SpynnakerLiveSpikesConnection

            all_retina_labels = retinaLeft.labels + retinaRight.labels
            self.live_connection_sender = SpynnakerLiveSpikesConnection(receive_labels=None, local_port=19999,
                                                                        send_labels=all_retina_labels)

            # this callback will be executed right after simulation.run() has been called. If simply a while True
            # is put there, the main thread will stuck there and will not complete the simulation.
            # One solution might be to start a thread/process which runs a "while is_running:" loop unless the main thread
            # sets the "is_running" to False.
            self.live_connection_sender.add_start_callback(all_retina_labels[0], self.start_injecting)

            import DVSReader as dvs
            # the port numbers might well be wrong
            self.dvs_stream_left = dvs.DVSReader(port=0,
                                                 label=retinaLeft.label,
                                                 live_connection=self.live_connection_sender)
            self.dvs_stream_right = dvs.DVSReader(port=1,
                                                  label=retinaRight.label,
                                                  live_connection=self.live_connection_sender)

            # start the threads, i.e. start reading from the DVS. However, nothing will be sent to the SNN.
            # See start_injecting
            self.dvs_stream_left.start()
            self.dvs_stream_right.start()

    def start_injecting(self):
        # start injecting into the SNN
        self.dvs_stream_left.start_injecting = True
        self.dvs_stream_right.start_injecting = True

    def get_network_dimensions(self):
        parameters = {'size':self.size,
                      'dim_x':self.dim_x,
                      'dim_y':self.dim_y,
                      'min_d':self.min_disparity,
                      'max_d':self.max_disparity}
        return parameters

    """ this method returns (and saves) a full list of spike times
    with the corresponding pixel location and disparities."""
    def get_spikes(self, sort_by_time=True, save_spikes=True):
        global same_disparity_indices, _retina_proj_l
        spikes_per_population = [x[1].getSpikes() for x in self.network]
        spikes = list()
        # for each column population in the network, find the x,y coordinates corresponding to the neuron
        # and the disparity. Then write them in the list and sort it by the timestamp value.
        for col_index, col in enumerate(spikes_per_population, 0):  # it is 0-indexed
            # find the disparity
            disp = self.min_disparity
            for d in range(0, self.max_disparity + 1):
                if col_index in same_disparity_indices[d]:
                    disp = d + self.min_disparity
                    break
            # for each spike in the population extract the timestamp and x,y coordinates
            for spike in col:
                x_coord = 0
                for p in range(0, self.dim_x):
                    if col_index in _retina_proj_l[p]:
                        x_coord = p
                        break
                y_coord = int(spike[0])
                spikes.append((round(spike[1], 1), x_coord+1, y_coord+1, disp))	# pixel coordinates are 1-indexed
        if sort_by_time:
            spikes.sort(key=lambda x: x[0])
        if save_spikes:
            if not os.path.exists("./spikes"):
                os.makedirs("./spikes")
            i = 0
            while os.path.exists("./spikes/{0}_{1}.dat".format(self.experiment_name, i)):
                i += 1
            with open('./spikes/{0}_{1}.dat'.format(self.experiment_name, i), 'w') as f:
                self._write_preamble(f)
                f.write("### DATA FORMAT ###\n"
                        "# Description: All spikes from the Collector Neurons are recorded. The disparity is inferred "
                        "from the Neuron ID. The disparity is calculated with the left camera as reference."
                        "The timestamp is dependent on the simulation parameters (simulation timestep).\n"
                        "# Each row contains: "
                        "Time -- x-coordinate -- y-coordinate -- disparity\n"
                        "### DATA START ###\n")
                for s in spikes:
                    f.write(str(s[0]) + " " + str(s[1]) + " " + str(s[2]) + " " + str(s[3]) + "\n")
                f.write("### DATA END ###")
        return spikes

    """ this method returns the accumulated spikes for each disparity as a list. It is not very useful except when
    the disparity sorting and formatting in the more general one get_spikes is not needed."""
    def get_accumulated_disparities(self, sort_by_disparity=True, save_spikes=True):
        if sort_by_disparity:
            global same_disparity_indices
            spikes_per_disparity_map = []
            for d in range(0, self.max_disparity - self.min_disparity + 1):
                collector_cells = [self.network[x][1] for x in same_disparity_indices[d]]
                spikes_per_disparity_map.append(sum([sum(x.get_spike_counts().values()) for x in collector_cells]))
                if save_spikes:
                    if not os.path.exists("./spikes"):
                        os.makedirs("./spikes")
                    i = 0
                    while os.path.exists("./spikes/{0}_disp_{1}.dat".format(self.experiment_name, i)):
                        i += 1
                    with open('./spikes/{0}_disp_{1}.dat'.format(self.experiment_name, i), 'w') as f:
                        self._write_preamble(f)
                        for s in spikes_per_disparity_map:
                            f.write(str(s) + "\n")
                return spikes_per_disparity_map
        else:
            # this is pretty useless. maybe it should be removed in the future
            all_spikes = sum(sum(x[1].get_spikes_count().values() for x in self.network))
            return all_spikes

    """ this method returns a list containing the membrane potential of all neural populations sorted by id."""
    def get_v(self, save_v=True):
        voltages = {"collector_v": [x[1].get_v() for x in self.network],
                    "blockers_v":[x[0].get_v() for x in self.network]}
        if save_v:
            if not os.path.exists("./membrane_potentials"):
                os.makedirs("./membrane_potentials")
            i = 0
            while os.path.exists("./membrane_potentials/{0}_{1}.dat".format(self.experiment_name, i)):
                i += 1
            with open('./membrane_potentials/{0}_{1}.dat'.format(self.experiment_name, i), 'w') as f:
                self._write_preamble(f)
                f.write("### DATA FORMAT ###\n"
                        "# Description: First all Blocker Populations are being printed. "
                        "Then all Collector populations. Both are sorted by Population ID (i.e. order of creation). "
                        "Each Blocker/Collector Population lists all neurons, sorted by Neuron ID. "
                        "There are two times more Blocker than Collector Neurons.\n"
                        "# Each row contains: "
                        "Blocker/Collector tag (b/c) -- Population ID -- Neuron ID -- Time -- Membrane Potential\n"
                        "### DATA START ###\n")
                for pop_id, pop_v in enumerate(voltages["blockers_v"]):
                    for v in pop_v:
                        f.write("b " + str(int(pop_id)) + " " + str(int(v[0])) + " " + str(v[1]) + " " + str(v[2]) + "\n")
                for pop_id, pop_v in enumerate(voltages["collector_v"]):
                    for v in pop_v:
                        f.write("c " + str(int(pop_id)) + " " + str(int(v[0])) + " " + str(v[1]) + " " + str(v[2]) + "\n")
                f.write("### DATA END ###")
        return voltages

    def _write_preamble(self, opened_file_descriptor):
        if opened_file_descriptor is not None:
            f = opened_file_descriptor
            f.write("### PREAMBLE START ###\n")
            f.write("# Experiment name: \n\t{0}\n".format(self.experiment_name))
            f.write("# Network parameters "
                    "(size in ensembles, x-dimension, y-dimension, minimum disparity, maximum disparity, "
                    "radius of excitation, radius of inhibition): "
                    "\n\t{0} {1} {2} {3} {4} {5} {6}\n".format(self.size, self.dim_x, self.dim_y,
                                                         self.min_disparity, self.max_disparity,
                                                         self.cell_params['topological']['radius_e'],
                                                         self.cell_params['topological']['radius_i']))
            f.write("# Neural parameters "
                    "(tau_excitation, tau_inhibition, tau_membrane, v_reset_blocker, v_reset_collector): "
                    "\n\t{0} {1} {2} {3} {4}\n".format(self.cell_params['neural']['tau_E'],
                                                 self.cell_params['neural']['tau_I'],
                                                 self.cell_params['neural']['tau_mem'],
                                                 self.cell_params['neural']['v_reset_blocker'],
                                                 self.cell_params['neural']['v_reset_collector']))
            f.write('# Synaptic parameters '
                    '(wBC, dBC, wSC, dSC, wSaB, dSaB, wSzB, dSzB, wCCi, dCCi, wCCe, dCCe): '
                    '\n\t{0} {1} {2} {3} {4} {5} {6} {7} {8} {9} {10} {11}\n'
                    .format(self.cell_params['synaptic']['wBC'],
                            self.cell_params['synaptic']['dBC'],
                            self.cell_params['synaptic']['wSC'],
                            self.cell_params['synaptic']['dSC'],
                            self.cell_params['synaptic']['wSaB'],
                            self.cell_params['synaptic']['dSaB'],
                            self.cell_params['synaptic']['wSzB'],
                            self.cell_params['synaptic']['dSzB'],
                            self.cell_params['synaptic']['wCCi'],
                            self.cell_params['synaptic']['dCCi'],
                            self.cell_params['synaptic']['wCCe'],
                            self.cell_params['synaptic']['dCCe']))
            f.write('# Comments: Caution: The synaptic parameters may vary according with '
                    'different simulation time steps. To understand the abbreviations for the '
                    'synaptic parameters, see the code documentation.\n')
            f.write("### PREAMBLE END ###\n")
