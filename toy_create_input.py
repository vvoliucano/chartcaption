from utils import create_input_files

# import argparse
# parser = argparse.ArgumentParser(description='Show, Attend, and Tell - Tutorial - Generate Caption for SVG')

# parser.add_argument('--karpathy_json_path', type=str, default = '/Users/tsunmac/vis/projects/autocaption/data/template.json', help='folder with data files saved by create_input_files.py')
# parser.add_argument('--image_folder', type=str, default = 'coco_5_cap_per_img_5_min_word_freq', help='base name shared by data files')
# parser.add_argument('--beam_size', '-b', default=5, type=int, help='beam size for beam search')
# parser.add_argument('--dont_smooth', dest='smooth', action='store_false', help='do not smooth alpha overlay')

# args = parser.parse_args()



if __name__ == '__main__':
    # Create input files (along with word map)

    create_input_files(dataset='chart',
                       karpathy_json_path='../data/svg_try/template.json',
                       image_folder='../data/svg_try/try_image/',
                       captions_per_image=5,
                       min_word_freq=0,
                       output_folder='../data/svg_try/try_output',
                       max_len=50,
                       image_type = "svg")
