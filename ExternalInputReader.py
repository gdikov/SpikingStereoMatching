import urllib

# This class reads spikes from an external input source (url or local file) and preprocesses the spikes for the retinas
# The spikes should be stored in a file with an extension ".dat" and should be formatted as follows:
# spike_time position_x position_y polarity retina
# where spike_time is in microseconds, position_x and position_y are pixel coordinates in the range [1, dim_x(dim_y)]
# polarity is the event type (0 OFF, 1 ON) and retina is the retina ID (0 left, 1 right) (or the other way round :D)
class ExternalInputReader():
    def __init__(self, url="",
                 file_path="",
                 crop_window=False,
                 dim_x=1,
                 dim_y=1,
                 sim_time=1000,
                 is_rawdata_time_in_ms=False):
        # these are the attributes which contain will contain the sorted, filtered and formatted spikes for each pixel
        self.retinaLeft = []
        self.retinaRight = []
        
        crop_window_x = -1
        crop_window_y = -1
        if crop_window:
            crop_window_x = dim_x
            crop_window_y = dim_y

        if url is not "" and file_path is not "" or \
            url is "" and file_path is "":
            print("ERROR: Ambiguous or void input source address. Give either a URL or a local file path.")
            return

        rawdata = None
        eventList = []
        # check the url or the file path, read the data file and pass it further down for processing.
        if url is not "" and url[-4:] == ".dat":
            # connect to website and parse text data
            file = urllib.urlopen(url)
            rawdata = file.read()
            file.close()
            # read the file line by line and extract the timestamp, x and y coordinates, polarity and retina ID.
            # TODO: add a case for different files like the npz, which are read from other servers.
            for e in rawdata.split('\n'):
                if e != '':
                    eventList.append([int(x) for x in e.split()])
        elif file_path is not "" and file_path[-4:] == ".dat":
            with open(file_path, 'r') as file:
                rawdata = file.read()
                file.close()
            # read the file line by line and extract the timestamp, x and y coordinates, polarity and retina ID.
            for e in rawdata.split('\n'):
                if e != '':
                    eventList.append([int(x) for x in e.split()])
        elif file_path is not "" and file_path[-4:] == ".npz":
            from numpy import load as load_npz_file
            f = load_npz_file(file_path)
            data_left = f["left"]
            data_right = f["right"]
            for t, x, y, p in data_left:
                if t > sim_time * 1000:
                    break
                eventList.append([float(t), int(x), int(y), int(p), 1])
            for t, x, y, p in data_right:
                if t > sim_time * 1000:
                    break
                eventList.append([float(t), int(x), int(y), int(p), 0])


        # initialise the maximum time constant as the total simulation duration. This is needed to set a value
        # for pixels which don't spike at all, since the pyNN frontend requires so.
        # If they spike at the last possible time step, their firing will have no effect on the simulation.
        max_time = sim_time

        # containers for the formatted spikes
        # define as list of lists of lists so that SpikeSourceArrays don't complain
        retinaL = [[[] for y in range(dim_y)] for x in range(dim_x)]
        retinaR = [[[] for y in range(dim_y)] for x in range(dim_x)]

        # last time a spike has occured for a pixel -- used to filter event bursts
        last_tL = [[0.0]]
        last_tR = [[0.0]]
        for x in range(0, dim_x):
            for y in range(0, dim_y):
                last_tL[x].append(-1.0)
                last_tR[x].append(-1.0)
            last_tL.append([])
            last_tR.append([])

        # process each event in the event list and format the time and position. Distribute among the retinas respectively
        for evt in eventList:
            x = evt[1] - 1
            y = evt[2] - 1
            if not is_rawdata_time_in_ms:
                t = evt[0] / 1000.0  # retina event time steps are in micro seconds, so convert to milliseconds
            else:
                t = evt[0]

            # use these lower and upper bounds to get pixels from a centered window only (if the dataset contains
            # pixels which are out of current network capability)
            lowerBoundX = (128 - crop_window_x) / 2
            upperBoundX = (128 + crop_window_x) / 2

            lowerBoundY = (128 - crop_window_y) / 2
            upperBoundY = (128 + crop_window_y) / 2

            # firstly, take events only from the within of a window centered at the retina view center,
            # then sort the left and the right events.
            if crop_window_x >= 0 and crop_window_y >= 0:
                if lowerBoundX <= x < upperBoundX and lowerBoundY <= y < upperBoundY:
                    if evt[4] == 0:
                        # filter event bursts and limit the events to the maximum time for the simulation
                        if t - last_tR[x - lowerBoundX][y - lowerBoundY] >= 1.0 and t <= max_time:
                            # print "r", x, y, x-lowerBoundX, y-lowerBoundY
                            retinaR[x - lowerBoundX][y - lowerBoundY].append(t)
                            last_tR[x - lowerBoundX][y - lowerBoundY] = t
                    elif evt[4] == 1:
                        if t - last_tL[x - lowerBoundX][y - lowerBoundY] >= 1.0 and t <= max_time:
                            # 				print "l", x, y, x-lowerBoundX, y-lowerBoundY
                            retinaL[x - lowerBoundX][y - lowerBoundY].append(t)
                            last_tL[x - lowerBoundX][y - lowerBoundY] = t
            else:
                if evt[4] == 0:
                    # apply the same time filtering
                    if t - last_tR[x][y] >= 1.0 and t <= max_time:
                        retinaR[x][y].append(t)
                        last_tR[x][y] = t
                elif evt[4] == 1:
                    if t - last_tL[x][y] >= 1.0 and t <= max_time:
                        retinaL[x][y].append(t)
                        last_tL[x][y] = t

        # fill the void cells with the last time possible, which has no effect on the simulation. The SpikeSourceArray
        # requires a value for each cell.
        for y in range(0, dim_x):
            for x in range(0, dim_y):
                if retinaR[y][x] == []:
                    retinaR[y][x].append(max_time + 10)
                if retinaL[y][x] == []:
                    retinaL[y][x].append(max_time + 10)

        # store the formatted and filtered events which are to be passed to the retina constructors
        self.retinaLeft = retinaL
        self.retinaRight = retinaR
