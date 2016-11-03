import numpy as np

class InputAnalyser(object):
    def __init__(self,
                 sim_time=0,
                 input_file=""):

        self.input_file = input_file
        self.events = []

        if input_file[-4:] == ".dat":
            with open(input_file, 'r') as file:
                rawdata = file.read()
                file.close()
            # read the file line by line and extract the timestamp, x and y coordinates, polarity and retina ID.
            for e in rawdata.split('\n'):
                if e != '':
                    self.events.append([int(x) for x in e.split()])

        elif input_file[-4:] == ".npz":
            from numpy import load as load_npz_file
            f = load_npz_file(input_file)
            data_left = f["left"]
            data_right = f["right"]
            for t, x, y, p in data_left:
                if t > sim_time:
                    break
                self.events.append((int(t), int(x), int(y), int(p), 1))
            for t, x, y, p in data_right:
                if t > sim_time:
                    break
                self.events.append((int(t), int(x), int(y), int(p), 0))

    def input_density(self):
        import matplotlib
        import matplotlib.pyplot as plt

        self.events = np.asarray(self.events)
        # print(self.events)
        plt.hexbin(self.events[:, 1], self.events[:, 2])
        plt.colorbar()
        plt.show()

    def stats(self):
        self.events = np.asarray(self.events)
        print("Median in x: {0}".format(np.median(self.events[:, 1])))
        print("Median in y: {0}".format(np.median(self.events[:, 2])))

if __name__ == "__main__":
    ia = InputAnalyser(sim_time=10000000, input_file="./input_data/warning1_lr.npz")
    ia.stats()
    ia.input_density()
