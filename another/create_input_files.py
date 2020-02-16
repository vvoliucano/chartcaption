from utils import create_input_files

if __name__ == '__main__':
    # Create input files (along with word map)
    create_input_files(dataset='coco',
                       karpathy_json_path='/home/can.liu/caption/data/annotations/captions_train2014.json',
                       image_folder='/home/can.liu/caption/data/train2014/',
                       captions_per_image=5,
                       min_word_freq=5,
                       output_folder='/home/can.liu/caption/data/',
                       max_len=50)
