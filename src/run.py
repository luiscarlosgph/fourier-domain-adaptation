##
# @brief  This script performs Fourier Domain Adaptation.
#
# @author Luis Carlos Garcia-Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   1 Jun 2020.

import argparse
import numpy as np
import cv2 
import os
import time

# My imports
import fda


def parse_command_line_parameters(parser):
    """
    @brief   Parses the command line parameters provided by the user and 
             makes sure that mandatory parameters are present.
    
    @param[in] parser argparse.ArgumentParser
    
    @returns an object with the parsed arguments. 
    """

    parser.add_argument('--source', required = True, 
                        help = 'Path to the source image.')
    parser.add_argument('--target', required = True,
                        help = 'Path to the target image.')
    parser.add_argument('--output', required = True,
                        help = 'Path to the output image.')
    parser.add_argument('--beta', required = True, default = 1e-2, 
                        type = float,
                        help = 'Factor to regulate the degree of adaptation.')
    
    # Parse command line
    args = parser.parse_args()

    return args


def validate_cmd_param(args):
    if not os.path.isfile(args.source):
        raise ValueError('[ERROR] The input file ' + args.source + ' does not exist.')

    if not os.path.isfile(args.target):
        raise ValueError('[ERROR] The input file ' + args.target + ' does not exist.')

 
def main(): 
    # Process command line parameters
    parser = argparse.ArgumentParser() 
    args = parse_command_line_parameters(parser)
    validate_cmd_param(args)
    
    # Read images
    source_im = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
    target_im = cv2.imread(args.target, cv2.IMREAD_UNCHANGED)
    
    # Domain adaptation
    tic = time.time()
    adapted_im = fda.fda(source_im, target_im, args.beta)
    toc = time.time()
    print('Domain adaptation performed in ' + str(toc - tic) + ' seconds.')
    
    # Save output image
    cv2.imwrite(args.output, adapted_im)
        

if __name__ == '__main__':
    main()
