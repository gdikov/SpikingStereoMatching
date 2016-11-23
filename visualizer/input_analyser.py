import numpy as np
import matplotlib
import matplotlib.pyplot as plt

class InputAnalyser(object):
    def __init__(self,
                 sim_time_begin=0,
                 sim_time_end=0,
                 input_file="",
                 input_file2=""):

        self.input_file = input_file
        self.input_file2 = input_file2
        self.events = []

        if input_file[-4:] == ".dat":
            with open(input_file, 'r') as file:
                rawdata = file.read()
            # read the file line by line and extract the timestamp, x and y coordinates, polarity and retina ID.
            for e in rawdata.split('\n'):
                if e != '':
                    evt = [int(float(x)) for x in e.split()]
                    if sim_time_begin*1000 <= evt[0] < sim_time_end * 1000:
                        self.events.append(evt)
                    elif evt[0] > sim_time_end * 1000:
                        break

        if input_file2[-4:] == ".dat":
            with open(input_file2, 'r') as file:
                rawdata = file.read()
            for e in rawdata.split('\n'):
                if e != '':
                    evt = [int(float(x)) for x in e.split()]
                    if sim_time_begin*1000 <= evt[0] < sim_time_end * 1000:
                        self.events.append(evt)
                    elif evt[0] > sim_time_end * 1000:
                        break
        elif input_file[-4:] == ".npz":
            from numpy import load as load_npz_file
            f = load_npz_file(input_file)
            data_left = f["left"]
            data_right = f["right"]
            for t, x, y, p in data_left:
                if sim_time_begin*1000 <= t < sim_time_end * 1000:
                    self.events.append((int(t), int(x), int(y), int(p), 1))
                elif t > sim_time_end * 1000:
                    break
            evt_count_left = len(self.events)
            print("Left num events: ", len(self.events))
            for t, x, y, p in data_right:
                if sim_time_begin*1000 <= t < sim_time_end * 1000:
                    self.events.append((int(t), int(x), int(y), int(p), 0))
                elif t > sim_time_end * 1000:
                    break
            print("Left num events: ", len(self.events)-evt_count_left)

    def input_density(self):
        self.events = np.asarray(self.events)
        # print(self.events)
        fig = plt.figure()
        ax = fig.gca()
        ax.set_xticks(np.arange(0, 128, 2))
        ax.set_yticks(np.arange(0, 128, 2))
        plt.hexbin(self.events[:, 1], self.events[:, 2])
        plt.colorbar()
        plt.grid()
        plt.show()


    def stats(self):
        self.events = np.asarray(self.events)
        print("Median in x: {0}".format(np.median(self.events[:, 1])))
        print("Median in y: {0}".format(np.median(self.events[:, 2])))

if __name__ == "__main__":
    ia = InputAnalyser(sim_time_begin=0000,
                       sim_time_end=3000,
                       input_file="./input_data/Back_On_Front_Accel_Fixed_even.npz")
    ia.stats()
    ia.input_density()


