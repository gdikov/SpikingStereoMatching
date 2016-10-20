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

pixel_l = ps.Population(1, ps.SpikeSourceArray, {'spike_times': [1]}, label="px_l")
# pixel_r = ps.Population(1, ps.SpikeSourceArray, {'spike_times': [100]}, label="px_r")

blocker = ps.Population(1 * 2, ps.IF_curr_exp, {'tau_syn_E': neural_params['tau_E'],
                                                'tau_syn_I': neural_params['tau_I'],
                                                'tau_m': neural_params['tau_mem'],
                                                'v_reset': neural_params['v_reset_blocker']}, label="bl_lr")

collector = ps.Population(1, ps.IF_curr_exp, {'tau_syn_E': neural_params['tau_E'],
                                              'tau_syn_I': neural_params['tau_I'],
                                              'tau_m': neural_params['tau_mem'],
                                              'v_reset': neural_params['v_reset_collector']}, label="cl")

blocker.record_v()
collector.record_v()

ps.Projection(blocker, collector, ps.FromListConnector([(0, 0, -20.5, 0.2),(1, 0, -20.5, 0.2)]), target='inhibitory')

ps.Projection(pixel_l, collector, ps.FromListConnector([(0, 0, 20.5, 1.6)]), target='excitatory')
ps.Projection(pixel_l, blocker, ps.FromListConnector([(0, 0, 22.5, 0.2)]), target='excitatory')
ps.Projection(pixel_l, blocker, ps.FromListConnector([(0, 1, -22.5, 0.2)]), target='inhibitory')

# ps.Projection(pixel_r, collector, ps.FromListConnector([(0, 0, 20.5, 1.6)]), target='excitatory')
# ps.Projection(pixel_r, blocker, ps.FromListConnector([(0, 1, 22.5, 0.2)]), target='excitatory')
# ps.Projection(pixel_r, blocker, ps.FromListConnector([(0, 0, -22.5, 0.2)]), target='inhibitory')

ps.run(100)

voltage_c = collector.get_v()
voltage_b = blocker.get_v()
print(voltage_c)
print(voltage_b)



fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

ax1.plot(voltage_c[: , 1], voltage_c[:, 2])
ax2.plot(voltage_b[voltage_b[:, 0] < 0.5, 1], voltage_b[voltage_b[:, 0] < 0.5, 2])
ax3.plot(voltage_b[voltage_b[:, 0] > 0.5 , 1], voltage_b[voltage_b[:, 0] > 0.5, 2])

ax1.set_title('Collector')
ax2.set_title('Blocker left')
ax3.set_title('Blocker right')

fig.text(0.5, 0.04, 'time in ms', ha='center', va='center')
fig.text(0.06, 0.5, 'voltage in mV', ha='center', va='center', rotation='vertical')

plt.savefig("./figures/Test_voltage.png")


ps.end()