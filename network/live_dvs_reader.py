from threading import Thread
import time
import serial as ser

class DVSReader(Thread):
    def __init__(self, address='/dev/ttyUSB', port=0, baudrate=4000000, buflen=64, label=None,
                 crop_window=True, dim_x=1, dim_y=1,
                 live_connection=None):
        Thread.__init__(self)

        self.dim_x = dim_x
        self.dim_y = dim_y

        if crop_window:
            self.lowerBoundX = (128 - dim_x) / 2
            self.upperBoundX = (128 + dim_x) / 2

            self.lowerBoundY = (128 - dim_y) / 2
            self.upperBoundY = (128 + dim_y) / 2
        else:
            # or should it be 1 and 128 ??
            self.lowerBoundX = 0
            self.upperBoundX = 127

            self.lowerBoundY = 0
            self.upperBoundY = 127

        self.live_connection = live_connection
        self.label = label

        self.onbuf = [(0, 0)] * buflen
        self.offbuf = [(0, 0)] * buflen
        self.tbuf = [0] * buflen
        self.buflen = buflen
        self.evind = 0
        self.port = port

        self.stop = False
        self.start_injecting = False
        self.alive = True
        self.setDaemon(True)

        self.dvsdev = ser.serial_for_url(address + str(port), baudrate, rtscts=True, dsrdtr=True, timeout=1)
        self.dvs_init()

    def dvs_init(self):
        self.dvsdev.write("R\n")
        time.sleep(0.1)
        self.dvsdev.write("1\n")  # LEDs on
        self.dvsdev.write("!E0\n")  # no timestamps
        self.dvsdev.write("E+\n")  # enable event streaming

    def dvs_close(self):
        self.dvsdev.write("0\n")  # LED off
        self.dvsdev.write("E-\n")  # event streaming off

    def run(self):
        """read and interpret data from serial port"""
        MAX_INJNEURONS_IN_POPULATION = 255
        try:
            lastx = -1
            lasty = -1

            n_pixel_cols_per_injector_pop = MAX_INJNEURONS_IN_POPULATION / self.dim_y

            while not self.stop:
                # self.bufbytes = self.dvsdev.inWaiting()
                data = bytearray(self.dvsdev.read(2))
                if (data[0] & 0x80) != 0:
                    x = data[0] & 0x7f
                    y = data[1] & 0x7f
                    p = data[1] >> 7

                    if self.lowerBoundX <= x < self.upperBoundX \
                            and self.lowerBoundY <= y < self.upperBoundY \
                            and self.start_injecting:
                        # filter naively event bursts (i.e. assume very low probability of the same pixel spiking next)
                        if x != lastx or y != lasty:
                            # normalize pixel coordinates and send spike in the corresponding population and neuron within it
                            injector_label = (x - self.lowerBoundX) / n_pixel_cols_per_injector_pop
                            injectorNeuronID = (y - self.lowerBoundY) \
                                               + ((x - self.lowerBoundX) % n_pixel_cols_per_injector_pop) * self.dim_y
                            self.live_connection.send_spike(label="{0}_{1}".format(self.label, injector_label),
                                                            neuron_id=injectorNeuronID,
                                                            send_full_keys=False)
                            lastx = x
                            lasty = y
                else:
                    self.dvsdev.read(1)
            self.dvs_close()
            self.start_injecting = False

        except self.dvsdev.SerialException, e:
            self.alive = False
            self.dvs_close()
            print "ERROR: Something went wrong at port %i" % self.port
