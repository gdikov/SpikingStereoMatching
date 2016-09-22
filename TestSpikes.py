import spynnaker.pyNN as ps

n = 4

ps.setup(timestep=0.2, min_delay=0.2, max_delay=10 * 0.2, threads=4)

pixel_r = ps.Population(n, ps.SpikeSourceArray, {'spike_times': [[10, 30], [1000], [1000]]}, label="test_input_r", structure=ps.Line())
pixel_l = ps.Population(n, ps.SpikeSourceArray, {'spike_times': [[10, 30], [1000], [1000]]}, label="test_input_l", structure=ps.Line())

print(pixel_r[0], pixel_r[1], pixel_r[2], pixel_r[3])
print(pixel_l[0], pixel_l[1], pixel_l[2], pixel_l[3])

collector = ps.Population(n, ps.IF_curr_exp, {'tau_syn_E': 2.0, 'tau_syn_I': 2.0, 'tau_m': 2.07, 'v_reset': -90.0}, label="collector")
blockers = ps.Population(n * 2, ps.IF_curr_exp, {'tau_syn_E': 2.0, 'tau_syn_I': 2.0, 'tau_m': 2.07, 'v_reset': -84.0}, label="blockers")
print(collector)
print(blockers)

collector.record()
blockers.record()

connList = [(0, 0, -20.5, 0.2), (1, 1, -20.5, 0.2), (2, 2, -20.5, 0.2), (3, 3, -20.5, 0.2), (4, 0, -20.5, 0.2), (5, 1, -20.5, 0.2), (6, 2, -20.5, 0.2), (7, 3, -20.5, 0.2)]
ps.Projection(blockers, collector, ps.FromListConnector(connList), target='inhibitory')

connListRetLBlockerL = [(0, 0, 22.5, 0.2), (1, 1, 22.5, 0.2), (2, 2, 22.5, 0.2), (3, 3, 22.5, 0.2)]
connListRetLBlockerR = [(0, 4, -22.5, 0.2), (1, 5, -22.5, 0.2), (2, 6, -22.5, 0.2), (3, 7, -22.5, 0.2)]
connListRetRBlockerL = [(0, 0, -22.5, 0.2), (1, 1, -22.5, 0.2), (2, 2, -22.5, 0.2), (3, 3, -22.5, 0.2)]
connListRetRBlockerR = [(0, 4, 22.5, 0.2), (1, 5, 22.5, 0.2), (2, 6, 22.5, 0.2), (3, 7, 22.5, 0.2)]

ps.Projection(pixel_r, collector, ps.OneToOneConnector(weights=20.5, delays=1.6), target='excitatory')
ps.Projection(pixel_r, blockers, ps.FromListConnector(connListRetRBlockerR), target='excitatory')
ps.Projection(pixel_r, blockers, ps.FromListConnector(connListRetRBlockerL), target='inhibitory')

ps.Projection(pixel_l, collector, ps.OneToOneConnector(weights=20.5, delays=1.6), target='excitatory')
ps.Projection(pixel_l, blockers, ps.FromListConnector(connListRetLBlockerL), target='excitatory')
ps.Projection(pixel_l, blockers, ps.FromListConnector(connListRetLBlockerR), target='inhibitory')

ps.run(1000)

spikes_c = collector.getSpikes()
spikes_b = blockers.getSpikes()

print(spikes_c)
print(spikes_b)

ps.end()
