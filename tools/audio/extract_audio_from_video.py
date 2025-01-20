import moviepy.editor as mp
import os
from moviepy.editor import VideoFileClip
import argparse
import json
import ipdb


def extract_audio_with_mapping(mapping, 
                               source_folder, 
                               dest_folder, 
                               split_size=None, 
                               split_idx=None, 
                               reverse_dataset=False,
                               audio_format='mp3'):
    # load the mapping
    if mapping is not None:
        all_mapping_content = json.load(open(mapping))
        all_mapping_key = list(all_mapping_content.keys())
        all_mapping_key.sort()
    else:
        all_video = os.listdir(source_folder)
        all_mapping_key = all_video
        all_mapping_content = {ele:ele for ele in all_mapping_key}      

    # sort the mapping key
    if split_size is not None:
        assert split_idx is not None
        curr_split_start = split_size * split_idx
        curr_split_end = split_size * (split_idx + 1)
        all_mapping_key = all_mapping_key[curr_split_start:curr_split_end]
    
    print('the size of the curr split is:', len(all_mapping_key))
    if reverse_dataset:
        all_mapping_key.reverse()
        
    # handle each file
    for count_i, file_key in enumerate(all_mapping_key):
        # the source path
        source_file = all_mapping_content[file_key]
        # the dest path
        full_source_file_path = os.path.join(source_folder, source_file)
        
        dest_file = source_file.split('.')[0] + '.' + audio_format
        full_dest_file_path = os.path.join(dest_folder, dest_file)
        # ipdb.set_trace() # check the dest file
        # the dest folder
        full_dest_file_folder = '/'.join(full_dest_file_path.split('/')[:-1])
        
        # make the dest folder
        if not os.path.exists(full_dest_file_folder):
            os.makedirs(full_dest_file_folder)
            
        # if target file exist skip
        if os.path.exists(full_dest_file_path):
            print(full_dest_file_path, ' exist, skipped.')
            continue
        
        # extract the audio
        try:
            print(f"Extracting audio from: {full_source_file_path}")
            video_clip = VideoFileClip(full_source_file_path)
            video_clip.audio.write_audiofile(full_dest_file_path)
            video_clip.close()
            print(f"Audio saved to: {full_dest_file_path}")
        except Exception as e:
            print(f"Failed to extract audio from {full_source_file_path}: {e}")
        
        print('total:', len(all_mapping_key), 'now:', count_i)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='extract image feature from image backebon')
    parser.add_argument('--source-folder', dest='source_folder', type=str, default=None)
    parser.add_argument('--dest-folder', dest='dest_folder', type=str, default=None)
    parser.add_argument('--mapping-file', dest='mapping_file', type=str, default=None)
    parser.add_argument('--split-size', dest='split_size', type=int, default=None)
    parser.add_argument('--split-idx', dest='split_idx', type=int, default=None)
    parser.add_argument('--reverse-dataset', dest='reverse_dataset', action='store_true')
    
    # parser.add_argument('--dest-folder', dest='feature_saving_root', type=str, default=None) 
    # parser.add_argument('--num-workers', dest='num_workers', type=int, default=0)    
    # parser.add_argument('--mapping-file', dest='video_mapping_path', type=str, default=None)    
    # parser.add_argument('--target-frame-num', dest='target_frame_num', type=int, default=32)    
    # parser.add_argument('--video-backend', dest='video_backend', type=str, default='av')
    # parser.add_argument('--split-loading', dest='split_loading', type=int, default=None)
    
    
    args = parser.parse_args()
    extract_audio_with_mapping(args.mapping_file, 
                               args.source_folder, 
                               args.dest_folder, 
                               args.split_size,
                               args.split_idx,
                               args.reverse_dataset,
                               audio_format='mp3')

# sample usage:
# python tools/Audio/extract_audio_from_video.py \
# --source-folder /path_to_your_video \
# --dest-folder /path_to_where_to_save_the_audio \
# --mapping-file /a_mapping_where_map_the_video_name_to_video_path \
    
    