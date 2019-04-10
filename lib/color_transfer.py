import numpy as np
from sklearn.preprocessing import PowerTransformer

class moment_matching_in_L_alpha_beta(object):
    """ Color Transfer between Images
        Erik Reinhard, Michael Ashikhmin, Bruce Gooch, and Peter Shirley
        University of Utah
        https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf
        
        Implementation inspired by C++ code and summary at
        https://hypjudy.github.io/2017/03/19/paperreading-color-transfer/
    """
    def __init__(self):
        """ Initialize color transform matrices and parameters """
        self.rgb_to_lms = numpy.float32([[0.3811, 0.5783, 0.0402],
                                         [0.1967, 0.7244, 0.0782],
                                         [0.0241, 0.1288, 0.8444]])
        self.lms_to_lab = numpy.float32([[1./numpy.sqrt(3.), 1./numpy.sqrt(3.), 1./numpy.sqrt(3.)],
                                         [1./numpy.sqrt(6.), 1./numpy.sqrt(6.), -2./numpy.sqrt(6.)],
                                         [1./numpy.sqrt(2.), -1./numpy.sqrt(2.), 0.]])
        self.lms_to_rgb = numpy.float32([[4.4679, -3.5873, 0.1193],
                                         [-1.2186, 2.3809, -0.1624],
                                         [0.0497, -0.2439, 1.2045]])
        self.lab_to_lms = numpy.float32([[1./numpy.sqrt(3.), 1./numpy.sqrt(6.), 1./numpy.sqrt(2.)],
                                         [1./numpy.sqrt(3.), 1./numpy.sqrt(6.), -1./numpy.sqrt(2.)],
                                         [1./numpy.sqrt(3.), 2./numpy.sqrt(6.), 0.]])
        self.distribution_fitter = PowerTransformer(method='yeo-johnson', standardize=True, copy=False)
        self.epsilon = np.finfo(float).eps

    def color_match(self, bgr_images, bgr_frames):
        """ convert to Lab, ( NOT CIELAB )
            transform images' distribution to frames' distribution,
            convert back to BGR
        """
        normal_dist=[]
        lab_transform = np.empty_like(bgr_images)
        for i, images in enumerate([bgr_frames, bgr_images]):
            lms = np.einsum('ij,...j', self.rgb_to_lms, images[...,::-1], optimize='greedy')
            log_lms = numpy.log10(lms + self.epsilon)
            lab = np.einsum('ij,...j', self.lms_to_lab, log_lms, optimize='greedy')
            for channel in range(lab.shape[-1]):
                current_dist = self.distribution_fitter.fit(lab[..., channel])
                normal_dist.append(current_dist)
                if i==1:
                    lab_normal = current_dist.transform(lab[..., channel])
                    lab_transform[..., channel] = normal_dist[channel].inverse_transform(lab_normal)

        log_lms = np.einsum('ij,...j', self.lab_to_lms, lab_transform, optimize='greedy')
        lms = 10. ** log_lms
        bgr = np.einsum('ij,...j', self.lms_to_rgb, lms, optimize='greedy')[...,::-1]

        return bgr








