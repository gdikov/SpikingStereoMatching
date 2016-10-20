import spynnaker.pyNN as ps
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ps.setup(timestep=0.2, min_delay=0.2, max_delay=10*0.2)

neural_params = {'tau_E': 2.0,
                    'tau_I': 2.0,
                    'tau_mem': 2.07,
                    'v_reset_blocker': -84.0,
                    'v_reset_collector': -90.0}
synaptic_params = {'wBC': -20.5,
                      'dBC': 0.2,
                      'wSC': 20.5,
                      'dSC': 1.6,
                      'wSaB': 22.5,
                      'dSaB': 0.2,
                      'wSzB': -22.5,
                      'dSzB': 0.2,
                      'wCCi': -50.0,
                      'dCCi': 0.2,
                      'wCCe': 3.0,
                      'dCCe': 0.2}

pixel = ps.Population(1, ps.SpikeSourceArray, {'spike_times': [1]}, label="px")

# blocker = ps.Population(1 * 2, ps.IF_curr_exp, {'tau_syn_E': neural_params['tau_E'],
#                                                 'tau_syn_I': neural_params['tau_I'],
#                                                 'tau_m': neural_params['tau_mem'],
#                                                 'v_reset': neural_params['v_reset_blocker']}, label="bl")
collector = ps.Population(1, ps.IF_curr_exp, {'tau_syn_E': neural_params['tau_E'],
                                              'tau_syn_I': neural_params['tau_I'],
                                              'tau_m': neural_params['tau_mem'],
                                              'v_reset': neural_params['v_reset_collector']}, label="cl")

# blocker.record_v()
collector.record_v()

# ps.Projection(blocker, collector, ps.FromListConnector([(0, 0, -5, 0.2),(1, 0, -5, 0.2)]), target='inhibitory')
ps.Projection(pixel, collector, ps.FromListConnector([(0, 0, 20.5, 1.6)]), target='excitatory')
# ps.Projection(pixel, blocker, ps.FromListConnector([(0, 0, -5, 0.2),(1, 0, -5, 0.2)]), target='excitatory')
# ps.Projection(pixel, blocker, ps.FromListConnector([(0, 0, -5, 0.2),(1, 0, -5, 0.2)]), target='inhibitory')

ps.run(100)

voltage = collector.get_v()
print(voltage)

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(voltage[: , 1], voltage[:, 2])
plt.savefig("./figures/Test_voltage.png")

ps.end()