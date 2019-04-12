import numpy as np
from sklearn.preprocessing import PowerTransformer

class moment_matching_in_l_alpha_beta():
    """ Color Transfer between Images
        Erik Reinhard, Michael Ashikhmin, Bruce Gooch, and Peter Shirley
        University of Utah
        https://www.cs.tau.ac.il/~turkel/imagepapers/ColorTransfer.pdf

        Implementation inspired by C++ code and summary at
        https://hypjudy.github.io/2017/03/19/paperreading-color-transfer/
    """
    def __init__(self, quick=True):
        """ Initialize color transform matrices and parameters """
        self.rgb_to_lms = np.float32([[0.3811, 0.5783, 0.0402],
                                      [0.1967, 0.7244, 0.0782],
                                      [0.0241, 0.1288, 0.8444]])
        self.lms_to_lab_one = np.float32([[1./np.sqrt(3.), 0., 0.],
                                          [0., 1./np.sqrt(6.), 0.],
                                          [0., 0., 1./np.sqrt(2.)]])
        self.lms_to_lab_two = np.float32([[1., 1., 1.],
                                          [1., 1., -2.],
                                          [1., -1., 0.]])
        self.lms_to_lab = self.lms_to_lab_one @ self.lms_to_lab_two
        self.lms_to_rgb = np.float32([[4.4679, -3.5873, 0.1193],
                                      [-1.2186, 2.3809, -0.1624],
                                      [0.0497, -0.2439, 1.2045]])
        self.lab_to_lms_one = np.float32([[1., 1., 1.],
                                          [1., 1., -1.],
                                          [1., -2., 0.]])
        self.lab_to_lms_two = np.float32([[np.sqrt(3.)/3., 0., 0.],
                                          [0., np.sqrt(6.)/6., 0.],
                                          [0., 0., np.sqrt(2.)/2.]])
        self.lab_to_lms = self.lab_to_lms_one @ self.lab_to_lms_two
        self.distribution_fitter = PowerTransformer(method='yeo-johnson',
                                                    standardize=True, copy=True)
        self.epsilon = .0000000001
        self.quick = quick

    def color_match(self, bgr_images, bgr_frames, masks):
        """ convert to Lab, ( NOT CIELAB )
            transform images' distribution to frames' distribution,
            convert back to BGR
        """
        mask_indices = np.nonzero(masks)
        if len(mask_indices[0]) == 0:
            return bgr_images

        lab_transform = np.empty_like(bgr_images)
        for i, imgs in enumerate([bgr_frames, bgr_images]):
            #frame_dist = []
            images = imgs.copy()
            lms = np.einsum('ij,...j', self.rgb_to_lms, images[..., ::-1], optimize='greedy')
            log_lms = np.log10(lms + self.epsilon)
            lab = np.einsum('ij,...j', self.lms_to_lab, log_lms, optimize='greedy')
            if self.quick:
                lab_masked = lab[mask_indices].reshape(-1, lab.shape[-1])
                lab_mean = np.mean(lab_masked, axis=0)
                lab_std = np.std(lab_masked, axis=0) + self.epsilon
                #lab_mean = np.mean(lab_masked, axis=tuple(range(lab.ndim - 1)))
                #lab_std = np.std(lab_masked, axis=tuple(range(lab.ndim - 1)))
                if i == 0:
                    frame_mean = lab_mean.copy()
                    frame_std = lab_std.copy()
                else:
                    lab_transform = ((lab - lab_mean) * (frame_std/lab_std)) + frame_mean
            else:
                # TODO as opposed to quick which works on single or batched images
                # this more accurate function only works on single images

                #shaping = lab.shape
                reshaped = lab[mask_indices].reshape(-1, lab.shape[-1])
                if i == 0:
                    print('\n  ', self.distribution_fitter.fit(reshaped).get_params())
                    #frame_dist.append(temp_dist)
                else:
                    #print('\n  ',len(frame_dist))
                    """
                    image_dist = self.distribution_fitter.fit(reshaped)
                    lab_normal = image_dist.transform(lab.reshape(-1, lab.shape[-1]))
                    lab_transform_masked = frame_dist[0].inverse_transform(lab_normal)
                    lab_transform = lab_transform_masked.reshape(shaping)
                    """
                    lab_transform = lab

        log_lms = np.einsum('ij,...j', self.lab_to_lms, lab_transform, optimize='greedy')
        lms = 10. ** log_lms
        bgr = np.einsum('ij,...j', self.lms_to_rgb, lms, optimize='greedy')[..., ::-1]
        print('\n', np.mean(bgr[mask_indices].reshape(-1, lab.shape[-1]), axis=0)," ---- ", np.mean(bgr_images[mask_indices].reshape(-1, lab.shape[-1]), axis=0)," ---- ", np.mean(bgr_frames[mask_indices].reshape(-1, lab.shape[-1]), axis=0))

        return bgr



class CIELAB():
    """ fill """
    def __init__(self):
        """ fill """
        self.whitepoint = np.array([1., 1., 1.])
        self.epsilon = 216. / 24389.
        self.epsilon_cube_root = np.cbrt(self.epsilon)
        self.kappa = 24389. / 27.
        #invert
        self.s_rgb_to_xyz_d65 = np.array([[0.4124564, 0.3575761, 0.1804375],
                                          [0.2126729, 0.7151522, 0.0721750],
                                          [0.0193339, 0.1191920, 0.9503041]], dtype='float32')
        self.d65_to_e = np.array([[1.0502616, 0.0270757, -0.0232523],
                                  [0.0390650, 0.9729502, -0.0092579],
                                  [-0.0024047, 0.0026446, 0.9180873]], dtype='float32')
        self.e_to_d65 = np.array([[0.9531874, -0.0265906, 0.0238731],
                                  [-0.0382467, 1.0288406, 0.0094060],
                                  [0.0026068, -0.0030332, 1.0892565]], dtype='float32')
        self.xyz_d65_to_s_rgb = np.array([[3.2404542, -1.5371385, -0.4985314],
                                          [-0.9692660, 1.8760108, 0.0415560],
                                          [0.0556434, -0.2040259, 1.0572252]], dtype='float32')
                                        #l      a      b
        self.fxfyfz_to_lab = np.array([[  0.,  500.,    0.],  # fx
                                       [116., -500.,  200.],  # fy
                                       [  0.,    0., -200.]]) # fz
                                        # fx      fy         fz
        self.lab_to_fxfyfz = np.array([[1./116., 1./116.,  1./116.],  # l
                                       [1./500.,      0.,       0.],  # a
                                       [    0.,       0., -1./200.]]) # b

    def from_srgb_one_to_xyz(self, srgb):
        """ fill """
        is_smaller = srgb <= 0.040449936
        is_not_smaller = np.invert(is_smaller)
        srgb[is_smaller] /= 12.92
        srgb[is_not_smaller] = ((srgb[is_not_smaller] + 0.055) / 1.055) ** 2.4

        xyz = srgb @ self.s_rgb_to_xyz_d65
        return xyz

    def from_xyz_to_srgb_one(self, xyz):
        """ fill """
        srgb = xyz @ self.xyz_d65_to_s_rgb

        is_smaller = srgb <= 0.0031308
        is_not_smaller = np.invert(is_smaller)
        srgb[is_smaller] *= 12.92
        srgb[is_not_smaller] = 1.055 * srgb[is_not_smaller] ** (1. / 2.4) - 0.055
        return srgb

    def from_xyz_to_lab(self, xyz):
        """ fill """
        lab = (xyz @ self.d65_to_e) / self.whitepoint

        is_greater = lab > self.epsilon
        is_not_greater = np.invert(is_greater)
        lab[is_greater] = np.cbrt(lab[is_greater])
        lab[is_not_greater] = (self.kappa * lab[is_not_greater] + 16.) / 116.

        lab[..., 0] = (116. * lab[..., 1]) - 16.
        lab[..., 1] = 500. * (lab[..., 0] - lab[..., 1])
        lab[..., 2] = 200. * (lab[..., 1] - lab[..., 2])

        #lab = lab @ self.fxfyfz_to_lab + np.array([-16., 0., 0.])
        return lab

    def from_lab_to_xyz(self, lab):
        """ fill """
        lab[..., 1] = (lab[..., 0] + 16.) / 116.
        lab[..., 0] = (lab[..., 0] / 500.) + lab[..., 1]
        lab[..., 2] = lab[..., 1] - (lab[..., 2] / 200.)

        #lab = (lab + np.array([16.0, 0.0, 0.0])) @ self.lab_to_fxfyfz

        is_greater = lab > self.epsilon_cube_root
        is_not_greater = np.invert(is_greater)
        lab[is_greater] = lab[is_greater] ** 3.
        lab[is_not_greater] = (116. * lab[is_not_greater] - 16.) / self.kappa

        xyz_d65 = (lab * self.whitepoint) @ self.e_to_d65
        return xyz_d65
