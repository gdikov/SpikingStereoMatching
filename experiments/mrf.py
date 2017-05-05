import time
import numpy as np
import matplotlib.pyplot as plt
import skimage.io as skio
import skimage.feature as skft

from utils.file_io import load_ground_truth


class StereoMRF(object):
    """
    Markov Random Field with loopy belief propagation (min-sum message passing). 
    """
    def __init__(self, dim, n_levels):
        self.n_levels = n_levels
        self.dimension = (n_levels,) + dim

    def _init_fields(self, image_left, image_right, prior, prior_trust_factor=0.5):
        """
        Initialise the message fields -- each hidden variable contains 5 message boxes from the 4 adjacent variables 
        (south, west, north, east) and the observed variable (data).
        """
        self.reference_image = image_left.astype('float32')
        self.secondary_image = image_right.astype('float32')
        self._message_field = {'south': np.zeros(self.dimension, dtype=np.float32),
                               'west': np.zeros(self.dimension, dtype=np.float32),
                               'north': np.zeros(self.dimension, dtype=np.float32),
                               'east': np.zeros(self.dimension, dtype=np.float32),
                               'data': np.zeros(self.dimension, dtype=np.float32)}
        nrow, ncol = self.reference_image.shape
        # crop the border from the high right in the right image as it cannot be mathed to the left
        if prior is not None:
            for l in xrange(self.n_levels):
                data_contrib = np.hstack((np.abs(self.reference_image[:, l:] - self.secondary_image[:, :ncol - l]),
                                          np.ones((nrow, l))))
                # data_contrib /= np.max(data_contrib) * 1/(np.mean(data_contrib))
                # print(data_contrib.min(), data_contrib.max(), data_contrib.mean())
                prior_mask = prior == l

                # equivalent to linear interpolating between the prior regions (having lowest value of 0) and the
                # data (weighted by the data_trust_factor = 1 - prior_trust_factor)
                self._message_field['data'][l, prior_mask] = -prior_trust_factor + (1 - prior_trust_factor) * data_contrib[prior_mask]
                self._message_field['data'][l, ~prior_mask] = data_contrib[~prior_mask]
        else:
            for l in xrange(self.n_levels):
                self._message_field['data'][l, :, :ncol - l] = np.abs(self.reference_image[:, l:]
                                                                      - self.secondary_image[:, :ncol - l])

    def _update_message_fields(self):
        """
        Pass messages from each hidden and observable variable to the corresponding adjacent hidden. 
        Since messages get summed up, normalise them to prevent overflows.
        """
        for direction in ['south', 'west', 'north', 'east']:
            message_updates = np.sum([field for d, field in self._message_field.iteritems() if d != direction], axis=0)
            if direction == 'south':
                self._message_field[direction][:, 1:, :] = message_updates[:, :-1, :]
            elif direction == 'west':
                self._message_field[direction][:, :, :-1] = message_updates[:, :, 1:]
            elif direction == 'north':
                self._message_field[direction][:, :-1, :] = message_updates[:, 1:, :]
            elif direction == 'east':
                self._message_field[direction][:, :, 1:] = message_updates[:, :, :-1]
            # add normalisation to the message values, as they grow exponentially with the number of iterations
            norm_factor = np.max(self._message_field[direction], axis=0, keepdims=True)
            norm_factor[norm_factor == 0] = 1
            self._message_field[direction] /= norm_factor

    def _update_belief_field(self):
        """
        Find an optimal (Maximum A-Posteriori) assignment for each pixel.  
        """
        energy_field = np.sum([field for d, field in self._message_field.items()], axis=0)
        self._belief_field = np.argmin(energy_field, axis=0)

    def lbp(self, image_left, image_right, prior=None, prior_trust_factor=0.5, n_iter=10):
        """
        Loopy Belief Propagation: initialise messages and pass them around iteratively. 
        Get MAP estimate after some fixed number of iterations.
        """
        self._init_fields(image_left, image_right, prior, prior_trust_factor)
        times = []
        for i in xrange(n_iter):
            start_timer = time.time()
            self._update_message_fields()
            times.append(time.time() - start_timer)
        print("LBP loop took {} sec on average".format(np.mean(times)))
        self._update_belief_field()
        return self._belief_field

if __name__ == '__main__':
    # Read left and right images as well as ground truth
    img_left = skio.imread("tsuk_L.png", as_grey=True)
    img_right = skio.imread("tsuk_R.png", as_grey=True)
    ground_truth = load_ground_truth("tsuk_gt.pgm")
    print(ground_truth.shape)

    # since the images are too large, e.g. 1980 x 2880, scale them down
    # NOTE: rescale the disparity too!
    scale_down_factor = 1.0
    if scale_down_factor != 1.0:
        from skimage.transform import rescale, resize

        img_left = rescale(img_left, 1.0 / scale_down_factor, preserve_range=True)
        img_right = rescale(img_right, 1.0 / scale_down_factor, preserve_range=True)
        ground_truth = rescale(ground_truth, 1.0 / scale_down_factor, preserve_range=True) / scale_down_factor

    ground_truth = ground_truth.astype('int16')

    # skio.imshow_collection([img_left, img_right, ground_truth])
    # plt.show()
    print("Image resolution: {}".format(img_left.shape))
    max_disp = np.max(ground_truth)
    print("Max disparity: {}".format(max_disp))

    # Initialise a MRF and calculate the some (possible sub-optimal) disparity assignemnt
    img_res = img_left.shape
    mrf = StereoMRF(img_res, n_levels=max_disp+1)
    disp_map = mrf.lbp(img_left, img_right, n_iter=10)
    # skio.imshow_collection([disp_map, ground_truth])
    # plt.show()

    # Run the LBP algorithm again with the same image pair but provide a retina-like prior,
    # sampled from the ground truth + noise
    prior_density = 0.5
    edges_mask = skft.canny(ground_truth.astype('float'), sigma=2)
    prior = ground_truth * (np.random.uniform(size=img_left.shape) <= prior_density)
    # prior = prior * edges_mask
    prior[prior == 0] = max_disp+2
    # skio.imshow(prior)
    # plt.show()

    disp_map_with_prior = mrf.lbp(img_left, img_right, prior=prior, prior_trust_factor=1, n_iter=10)
    # skio.imshow_collection([disp_map, disp_map_with_prior, prior])
    # plt.show()

    clipped_ground_truth = ground_truth[18:-18, 18:-18]
    clipped_without_prior = disp_map[18:-18, 18:-18]
    clipped_with_prior = disp_map_with_prior[18:-18, 18:-18]
    print("RMSE w/o prior: {}\nRMSE w. prior: {}".format(np.sqrt(np.mean((clipped_without_prior -
                                                                          clipped_ground_truth)**2)),
                                                         np.sqrt(np.mean((clipped_with_prior -
                                                                          clipped_ground_truth)**2))))
