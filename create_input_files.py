from utils import create_input_files
import argparse

parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption for SVG')

parser.add_argument('--dataset', type=str, default = "coco", help='dataset type, coco, chart')
parser.add_argument('--karpathy_json_path', type=str, default = '../data/karpathy/dataset_coco.json', help='')
parser.add_argument('--image_folder', type=str, default = '../data/coco_2014/', help='')
parser.add_argument('--captions_per_image', type=int, default = 5, help='')
parser.add_argument('--min_word_freq', type=int, default = 5, help='')
parser.add_argument('--output_folder', type=str, default = '/home/can.liu/caption/data/karpathy_output/', help='')
parser.add_argument('--max_len', default=50, type=int, help='max length of the input text')
parser.add_argument('--image_type', type=str, default = 'pixel', help='image type as input')
parser.add_argument('--need_text', action='store_true', help="decide whether need text")
parser.add_argument('--max_element_number', default=40, type=int, help="decide whether need text")

max_element_number

args = parser.parse_args()


if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset=args.dataset,
                       karpathy_json_path = args.karpathy_json_path,
                       image_folder = args.image_folder,
                       captions_per_image = args.captions_per_image,
                       min_word_freq = args.min_word_freq,
                       output_folder= args.output_folder,
                       max_len = args.max_len,
                       image_type = args.image_type,
                       need_text = args.need_text,
                       max_element_number = args.max_element_number)
