"""Main script to run the object detection routine."""

import argparse
from utils import array, plot, terminal
from detection.deeplearning import DeepDetector
from detection.traditional import TraditionalDetector

def parse_arguments():
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
        help='Type of object detection: color, shape, or category',
        required=False,
        default='color')
    parser.add_argument(
        '--multipleObject',
        help='Whether to detect single or multiple types of objects.',
        required=False,
        default=False)
    parser.add_argument(
        '--trueLabel',
        help="""If it is color detection, input (Blue, Green, Red) of the object.
                Else if shape, input (Height, Width, Size). Else, input class_name
                of the object. In traditional, the label is also class_name""",
        required=False,
        type=str,
        default='(55, 232, 254)')
    parser.add_argument(
        '--model',
        help='Path of the object detection model.',
        required=False,
        default='./model/color_detector2.tflite')
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
        '--PLCAddress',
        help='IP Address of the PLC to be sent detection data to',
        required=False,
        default='10.42.26.129')
    parser.add_argument(
        '--csvFilename',
        help='Name or path of the output CSV file.',
        required=False,
        default='./csv_data/Deteksi.csv')
    parser.add_argument(
        '--showMean',
        help='Whether to show the mean on detection graph.',
        required=False,
        default=False)
    return parser.parse_args()


def main():
    """Main function to run detection"""
    args = parse_arguments()
    
    if args.method == 'deeplearning':
        detector = DeepDetector(bool(args.multipleObject), args.model,
                                    args.frameWidth, args.frameHeight,
                                    int(args.numThreads),bool(args.enableEdgeTPU))
    
    elif args.method == 'traditional':
        if str(args.detectionType) == 'category':
            args.detectionType = terminal.prompt_type()    
        detector = TraditionalDetector(bool(args.multipleObject), args.frameWidth,
                                       args.frameHeight)
        
    
    result = detector.detect(str(args.detectionType), str(args.trueLabel),
                             str(args.PLCAddress))
    record = array.stack_array(result[0][0:], result[1][0:],
                               result[2][0:], result[3][0:],
                               result[4][0:], result[5][0:],
                               result[6][0:], result[7][0:])
    array.create_csv(record, args.method, str(args.detectionType), str(args.csvFilename))    
    dly_arr, dt_count,_ = array.collect_data(result[0])
    fps_arr, fps_count, _ = array.collect_data(result[1])
    _, _, dtct_ratio = array.collect_data(result[4])
    plot.plot_delay(dly_arr, dt_count, show_mean=bool(args.showMean))
    plot.plot_fps(fps_arr, fps_count, show_mean=bool(args.showMean))
    plot.plot_detection(dtct_ratio)


if __name__ == '__main__':
    main()