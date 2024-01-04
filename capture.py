"""Script to capture images from object detection"""

import argparse
from utils import video
from detection import deeplearning, traditional


def parse_argument():
    """Parse arguments for command line"""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--method',
        help='Two computer vision method: traditional or deeplearning',
        required=False,
        type=str,
        default='deeplearning')
    parser.add_argument(
        '--detectionType',
        help='Type of detection for traditional method: color OR shape',
        required=False,
        default='color')
    parser.add_argument(
        '--multipleObject',
        help='Whether to detect single or multiple types of objects.',
        required=False,
        default=False)
    parser.add_argument(
        '--trueClass',
        help='Name of the object class to be detected if it is a single type.',
        required=False,
        type=str,
        default='yellow_duck')
    parser.add_argument(
        '--collectAll',
        help='Whether to collect all images or only the True Positive one.',
        required=False,
        default=False)
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='./model/frogducky2.tflite')
    parser.add_argument(
        '--frameWidth',
        help='Width of frame to capture from camera.',
        required=False,
        type=int,
        default=640)
    parser.add_argument(
        '--frameHeight',
        help='Height of frame to capture from camera.',
        required=False,
        type=int,
        default=480)
    parser.add_argument(
        '--numThreads',
        help='Number of CPU threads to run the model.',
        required=False,
        type=int,
        default=4)
    parser.add_argument(
        '--enableEdgeTPU',
        help='Whether to run the model on EdgeTPU.',
        action='store_true',
        required=False,
        default=False)
    parser.add_argument(
        '--videoFilename',
        help='Name of the output video file with .mp4 extension',
        required=False,
        default='Deteksi_Objek.mp4')
    return parser.parse_args()


def main():
    """Main function to run detection"""
    args = parse_argument()
    
    if args.method == 'deeplearning':
        detector = deeplearning.DeepDetector(bool(args.multipleObject), args.model,
                                                 args.frameWidth, args.frameHeight,
                                                 int(args.numThreads),bool(args.enableEdgeTPU))
        detector.capture(str(args.trueClass), bool(args.collectAll), str(args.videoFilename))
    
    elif args.method == 'traditional':   
        if str(args.detectionType) == 'category':
            args.detectionType = terminal.prompt_type()
        detector = traditional.TraditionalDetector(bool(args.multipleObject), args.frameWidth,
                                                   args.frameHeight)
        detector.capture(str(args.detectionType), str(args.trueClass), 
                         bool(args.collectAll), str(args.videoFilename))
        
    video.move_video(str(args.videoFilename))


if __name__ == '__main__':
    main()