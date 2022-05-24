#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# @brief   Simple Fourier-based domain adaptation.
# @author  Luis Carlos Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date    4 June 2020.

import numpy as np
import cv2


def fft_amp_phase(im: np.ndarray):
    """
    @brief    Retrieve magnitude and phase images using the Discrete Fourier 
              Transform from a BGR image.

    @param[in]  im  OpenCV single-channel image (H, W) in the [0, 255] 
                    range (np.uint8).

    @returns the Fourier amplitude and phase images (amplitude, phase).
             As would be expected, size of the magnitude and phase images 
             returned is the same of the input image.
    """
    if im.dtype != np.uint8:
        raise ValueError('[ERROR] fft_amp_phase() expects a np.ndarray of '
                         'type np.uint8.')

    # FFT
    dft = cv2.dft(im.astype(np.float64) / 255., flags = cv2.DFT_COMPLEX_OUTPUT)

    # Join the two channels (real and imag) into a single number, 
    # as numpy expects
    dft = dft[:, :, 0] + 1j * dft[:, :, 1]

    # Shift the DC component to the centre of the image
    dft_shift = np.fft.fftshift(dft)
    
    # Compute Fourier amplitude and angle from the frequency spectrum 
    amp = np.abs(dft_shift)
    phase = np.angle(dft_shift)

    # Make sure that the output images have the same dimensions of the 
    # input image
    assert(im.shape[0] == amp.shape[0])
    assert(amp.shape[0] == phase.shape[0])
    assert(im.shape[1] == amp.shape[1])
    assert(amp.shape[1] == phase.shape[1])

    return amp, phase


def ifft_amp_phase(amp: np.ndarray, phase: np.ndarray):
    """
    @brief  Inverse Discrete Fourier Transform. 

    @param[in]  amp    Fourier amplitude image.
    @param[in]  phase  Fourier phase image.

    @returns BGR reconstructed image.
    """

    # Combine amplitude and phase to form the frequency spectrum 
    dft_shift = np.multiply(amp, np.exp(1j * phase))

    # Put back the DC component in the (0, 0) coordinate
    dft = np.fft.ifftshift(dft_shift)

    # Separate complex number into two channels, as OpenCV iFFT expects
    dft = np.array([dft.real, dft.imag]).transpose(1, 2, 0)

    # Inverse FFT
    im = cv2.idft(dft, flags=cv2.DFT_SCALE)
    im = np.abs(im[:, :, 0] + 1j * im[:, :, 1])

    # Min-max normalisation to put back the image in range [0.0, 1.0]
    im = (im - np.min(im)) / (np.max(im) - np.min(im))

    # Convert image to unsigned int in range [0, 255] as OpenCV expects
    im = np.round(im * 255.).astype(np.uint8)

    return im


def fda(source_im: np.ndarray, target_im: np.ndarray, 
        beta: float = 0.001) -> np.ndarray:
    """
    @brief    Fourier Domain Adaptation, as proposed in:
           
              "FDA: Fourier Domain Adaptation for Semantic Segmentation" by 
              Yanchao Yang and Stefano Soatto, CVPR 2020.

    @details  The domain adaptation consists of replacing the high frequency
              component of the Fourier amplitude of the source image with that
              of the target domain image. 

              To do so, we replace a small rectangle around the centre of the 
              source amplitude image with a rectangle of the same size from
              the target amplitude image. What you regulate with the beta 
              parameter is the size of the rectangle (h * beta, w * beta).

    @param[in]  source_im  OpenCV/Numpy BGR image (H, W), range [0, 255] 
                           (np.uint8).
    @param[in]  target_im  OpenCV/Numpy BGR image (H, W), range [0, 255] 
                           (np.uint8).
    @param[in]  beta       Parameter to regulate the size of the rectangle
                           that captures the high frequencies of the target
                           image, default value is beta = 0.001. Higher values
                           of beta will get lower frequencies of the 
                           target domain. If you push it, artifacts will start
                           to appear.

    @returns the source image adapted to the target domain.
    """
    # Compute stats of the source image
    rows, cols, _ = source_im.shape
    crow, ccol = rows // 2, cols // 2
    
    # Compute the size of the Fourier centre crop
    win_h = int(round(beta * rows)) 
    win_w = int(round(beta * cols))

    if win_h < 1 or win_w < 1:
        beta = min(1. / cols, 1. / rows)
        print('[WARN] The window size is to small because either your images '
              + 'are too small or your beta is too low. '
              + 'The beta has been modified to ' + ('%.4f' % beta) + ' so that '
              + 'the window is at least 1x1 pixel.')
        win_h = int(round(beta * rows)) 
        win_w = int(round(beta * cols))

    # Resize target image to the size of the source image
    target_im = cv2.resize(target_im, (cols, rows))

    adapted_im = np.empty_like(source_im)
    for k in range(3):
        # FFT
        amp_s, phase_s = fft_amp_phase(source_im[:, :, k])
        amp_t, phase_t = fft_amp_phase(target_im[:, :, k])

        # Perform domain adaptation by transferring the FFT amplitude from target to source
        replacement = amp_t[crow - win_h:crow + win_h, 
                            ccol - win_w:ccol + win_w].copy()
        amp_s[crow - win_h:crow + win_h, 
              ccol - win_w:ccol + win_w] = replacement 
        adapted_im[:, :, k] = ifft_amp_phase(amp_s, phase_s)
        
    return adapted_im


if __name__ == '__main__':
    raise RuntimeError('[ERROR] This module (fda) is not meant to be run '
                       + 'as a script.')
