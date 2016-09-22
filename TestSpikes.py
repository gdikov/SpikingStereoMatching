import pyNN.spiNNaker as ps

ps.setup(timestep=0.2, min_delay=0.2, max_delay=10 * 0.2, threads=4)

pixel_r = ps.Population(1, ps.SpikeSourceArray, {'spike_times': [[10, 30], [1000], [1000]]}, label="test_input_r", structure=ps.Line())
pixel_l = ps.Population(1, ps.SpikeSourceArray, {'spike_times': [[10, 30], [1000], [1000]]}, label="test_input_l", structure=ps.Line())

collector = ps.Population(1, ps.IF_curr_exp, {'tau_syn_E': 2.0, 'tau_syn_I': 2.0, 'tau_m': 2.07, 'v_reset': -90.0}, label="collector")
blockers = ps.Population(1 * 2, ps.IF_curr_exp, {'tau_syn_E': 2.0, 'tau_syn_I': 2.0, 'tau_m': 2.07, 'v_reset': -84.0}, label="blockers")

collector.record()
blockers.record()

connList = [(0, 0, -20.5, 0.2), (1, 0, -20.5, 0.2)]
ps.Projection(blockers, collector, ps.FromListConnector(connList), target='inhibitory')

connListRetLBlockerL = [(0, 0, 22.5, 0.2)]
connListRetLBlockerR = [(0, 1, -22.5, 0.2)]
connListRetRBlockerL = [(0, 0, 22.5, 0.2)]
connListRetRBlockerR = [(0, 1, -22.5, 0.2)]

ps.Projection(pixel_r, collector, ps.OneToOneConnector(weights=20.5, delays=1.6), target='excitatory')
ps.Projection(pixel_r, blockers, ps.FromListConnector(connListRetRBlockerR), target='excitatory')
ps.Projection(pixel_r, blockers, ps.FromListConnector(connListRetRBlockerL), target='inhibitory')

ps.Projection(pixel_l, collector, ps.OneToOneConnector(weights=20.5, delays=1.6), target='excitatory')
ps.Projection(pixel_l, blockers, ps.FromListConnector(connListRetLBlockerL), target='excitatory')
ps.Projection(pixel_l, blockers, ps.FromListConnector(connListRetLBlockerR), target='inhibitory')

ps.run(1000)

spikes_c = collector.getSpikes()
spikes_b = blockers.getSpikes()

print(spikes_c, spikes_b)

ps.end()