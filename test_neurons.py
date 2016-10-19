import spynnaker.pyNN as ps
import matplotlib.pyplot as plt

ps.setup(timestep=0.2, min_delay=0.2, max_delay=10*0.2)

pixel = ps.Population(1, ps.SpikeSourceArray, {'spike_times': [1]}, label="px")

# blocker = ps.Population(1 * 2, ps.IF_curr_exp, label="bl")
collector = ps.Population(1, ps.IF_curr_exp, label="cl")

# blocker.record_v()
collector.record_v()

# ps.Projection(blocker, collector, ps.FromListConnector([(0, 0, -5, 0.2),(1, 0, -5, 0.2)]), target='inhibitory')
ps.Projection(pixel, collector, ps.FromListConnector([(0, 0, -5, 0.2)]), target='excitatory')
# ps.Projection(pixel, blocker, ps.FromListConnector([(0, 0, -5, 0.2),(1, 0, -5, 0.2)]), target='excitatory')
# ps.Projection(pixel, blocker, ps.FromListConnector([(0, 0, -5, 0.2),(1, 0, -5, 0.2)]), target='inhibitory')

ps.run(100)

voltage = [x[1].get_v() for x in collector]

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.plot(voltage[: , 0], voltage[:, 1])
plt.savefig("./figures/Test_voltage.png")

ps.end()