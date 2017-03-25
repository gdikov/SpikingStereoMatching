import numpy as np

with open('../data/input/NSTlogo_disp12-8-3_1s.dat', 'r') as f:
    events = np.asarray([map(int, x.split()) for x in f.readlines()], dtype=np.int32)

# get left and right events in space and time
left_bit = 0
right_bit = 1
events = np.concatenate((events[:, :3], events[:, 4:]), axis=1)
events_left = events[events[:, 3] == left_bit, :][:, :3]
events_right = events[events[:, 3] == right_bit, :][:, :3]

# find begin and end times
start_events = min(np.min(events_left[:, 0]), np.min(events_right[:, 0]))
end_events = max(np.max(events_left[:, 0]), np.max(events_right[:, 0]))

# specify noise density along the time axis
duration = end_events - start_events    # in ms
noisiness = 0.1                         # i.e. a noisy event will appear approx. every round(1/noisiness) ms
noisy_events_count = int(duration * noisiness)

# sample uncorrelated noise for left and right retina
noise_times_left = np.random.randint(low=start_events, high=end_events, size=noisy_events_count, dtype=np.int32)
noise_coords_left = np.random.randint(low=0, high=128, size=(noisy_events_count, 2), dtype=np.int32)
noise_left = np.concatenate((noise_times_left[:, None], noise_coords_left), axis=1)

noise_times_right = np.random.randint(low=start_events, high=end_events, size=noisy_events_count, dtype=np.int32)
noise_coords_right = np.random.randint(low=0, high=128, size=(noisy_events_count, 2), dtype=np.int32)
noise_right = np.concatenate((noise_times_right[:, None], noise_coords_right), axis=1)

# merge the noise with the correct events
poluted_events_left = np.concatenate((events_left, noise_left), axis=0)
poluted_events_right = np.concatenate((events_right, noise_right), axis=0)

# sort by time
poluted_events_left = poluted_events_left[poluted_events_left[:, 0].argsort()]
poluted_events_right = poluted_events_right[poluted_events_right[:, 0].argsort()]

# add meta info (polarity, left/right bit) and merge into single array (also sorted by time).
# Notice that the polarity is 0 everywhere and it is the 4th column. Left/Right bit is 5th.
size_left = poluted_events_left.shape[0]
size_right = poluted_events_right.shape[0]

poluted_events_left = np.concatenate((poluted_events_left,
                                      np.zeros((size_left, 1), dtype=np.int32),
                                      left_bit * np.ones((size_left, 1), dtype=np.int32)), axis=1)
poluted_events_right = np.concatenate((poluted_events_right,
                                       np.zeros((size_right, 1), dtype=np.int32),
                                       right_bit * np.ones((size_right, 1), dtype=np.int32)), axis=1)
poluted_events = np.concatenate((poluted_events_left, poluted_events_right), axis=0)

poluted_events = poluted_events[poluted_events[:, 0].argsort(kind='mergesort')]

# save back into noisy version of the file
with open('../data/input/NSTlogo_disp12-8-3_noisy.dat', 'w') as f:
    for e in poluted_events:
        f.write(str(e[0]) + " " + str(e[1]) + " " +
                str(e[2]) + " " + str(e[3]) + " " +
                str(e[4]) + "\n")

