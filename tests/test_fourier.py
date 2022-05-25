"""
@brief  Unit tests to check the FDA Python package.
@author Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
@date   24 May 2022.
"""
import unittest
import numpy as np
import cv2

# My imports
import fda


class TestFourierMethods(unittest.TestCase):

    def test_dft(self, eps=np.finfo(np.float32).eps):
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

    def test_hermitian_property(self, eps=1e-1):
        rows = 1080
        cols = 1920

        # Generate a random grayscale image and compute the DFT
        im = np.round(np.random.rand(rows, cols) * 255).astype(np.uint8)
        amp, phase = fda.fft_amp_phase(im)

        # Generate another random grayscale image and compute the DFT
        im2 = np.round(np.random.rand(rows, cols) * 255).astype(np.uint8)
        amp2, phase2 = fda.fft_amp_phase(im2)

        # Replace a chunk of the DFT magnitude of one image with the DFT 
        # magnitude of the other image
        crow, ccol = rows // 2, cols // 2
        win_h = int(np.random.rand() * 256)
        win_w = int(np.random.rand() * 256)
        replacement = amp2[crow - win_h:crow + win_h, 
                           ccol - win_w:ccol + win_w].copy()
        amp[crow - win_h:crow + win_h, 
            ccol - win_w:ccol + win_w] = replacement 

        # Combine amplitude and phase to form the frequency spectrum 
        dft_shift = np.multiply(amp, np.exp(1j * phase))

        # Put back the DC component in the (0, 0) coordinate
        dft = np.fft.ifftshift(dft_shift)

        # Separate complex number into two channels, as OpenCV iDFT expects
        dft = np.array([dft.real, dft.imag]).transpose(1, 2, 0)

        # Inverse DFT
        idft = cv2.idft(dft, flags=cv2.DFT_SCALE)
        # idft_real = idft[:, :, 0]
        idft_imaginary = idft[:, :, 1]

        # As both images are real, the Fourier transforms are Hermitian, 
        # therefore, the mix and match of frequencies in the spectra is also
        # Hermitian, and the inverse DFT must be real
        self.assertTrue(np.all(idft_imaginary < eps))


if __name__ == '__main__':
    unittest.main()
