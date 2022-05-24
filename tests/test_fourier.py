"""
@brief  Unit tests to check the FDA Python package.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   24 May 2022.
"""
import unittest
import numpy as np

# My imports
import fda

class TestFourierMethods(unittest.TestCase):

    def test_fft(self, eps=1e-9):
        # Generate a random BGR image 
        h = 1080
        w = 1920
        c = 3
        im = np.round(np.random.rand(h, w, c) * 255).astype(np.uint8)

        # DFT
        blue_amp, blue_phase = fda.fft_amp_phase(im[:, :, 0])
        green_amp, green_phase = fda.fft_amp_phase(im[:, :, 1])
        red_amp, red_phase = fda.fft_amp_phase(im[:, :, 2])
        
        # Inverse DFT 
        blue_recon = fda.ifft_amp_phase(blue_amp, blue_phase)
        green_recon = fda.ifft_amp_phase(green_amp, green_phase)
        red_recon = fda.ifft_amp_phase(red_amp, red_phase)
        im_recon = np.dstack((blue_recon, green_recon, red_recon))
        
        # Check that the reconstructed image is the same as the original 
        self.assertEqual(np.sum(im_recon - im), 0)


if __name__ == '__main__':
    unittest.main()
