
'''
This script applies various image augmentations using the Albumentations library.
It supports color jitter, hue-saturation-value adjustments, brightness alterations, and gamma adjustments.

Usage:
    python augment.py <aug_type> <input_dir> <output_dir>


'''

import albumentations as A
import argparse
import os
import shutil
import cv2

class Augmentation:
    def __init__(self):
        pass


    @staticmethod
    def jitter_colour(image_path):
        '''
            Jitters the colour of the image
        '''
        image = cv2.imread(image_path)
        transform = A.ColorJitter(p=1.0)
        jittered = transform(image=image)
        return jittered['image']
    
    @staticmethod
    def alter_hsv(image_path):
        '''
            Alters the hue, saturation and value of the image
        '''
        image = cv2.imread(image_path)
        transform = A.HueSaturationValue(p=1.0)
        altered = transform(image=image)
        return altered['image']
    
    @staticmethod
    def alter_brightness(image_path):
        '''
            Alters the brightness of the image
        '''
        image = cv2.imread(image_path)
        transform = A.RandomBrightnessContrast(p=1.0)
        altered = transform(image=image) 
        return altered['image']
    
    @staticmethod
    def random_gamma(image_path):
        '''
            Alters the gamma of the image
        '''
        image = cv2.imread(image_path)
        transform = A.RandomGamma(p=1.0)
        altered = transform(image=image)
        return altered['image']
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('aug_type', type=str, help='Type of augmentation to apply, can be colour_jitter, hsv')
    parser.add_argument('input_dir', type=str, help='Path to the input directory')
    parser.add_argument('output_dir', type=str, help='Path to the output directory')
    args = parser.parse_args()

    output_dir = os.path.join(args.output_dir, args.aug_type)

    print('creating output directory: ' + output_dir)
    os.makedirs(output_dir, exist_ok=True)

    print(f'finding images from {args.input_dir}...')
    image_paths = os.listdir(args.input_dir)
    image_paths = [os.path.join(args.input_dir, image_path) for image_path in image_paths]
    print(f'found {len(image_paths)} images')

    print('transforming images...')
    for image_path in image_paths:

        if args.aug_type == 'jitter':
            transformed_image = Augmentation.jitter_colour(image_path)
        elif args.aug_type == 'hsv':
            transformed_image = Augmentation.alter_hsv(image_path)
        elif args.aug_type == 'brightness':
            transformed_image = Augmentation.alter_brightness(image_path)
        elif args.aug_type == 'gamma':
            transformed_image = Augmentation.random_gamma(image_path)

        output_path = os.path.join(output_dir, args.aug_type + '_' + os.path.basename(image_path))
        cv2.imwrite(output_path, transformed_image)
    
    print('transformed images saved to: ' + output_dir)
        
