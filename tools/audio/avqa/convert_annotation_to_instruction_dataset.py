# This script is used to convert the raw annotation to the instrcution dataset

import json
import os

origin_json_file = './data/video_instruction_tuning/avqa/train_qa.json'
save_json_file = './data/video_instruction_tuning/avqa/train_qa_instruct.json'
save_mapping_json_file = './data/video_instruction_tuning/avqa/from_vid_to_video_name.json'
save_feat_mapping_json_file = './data/video_instruction_tuning/avqa/from_vid_to_feat_name.json'

origin_file_content = json.load(open(origin_json_file))


start_prompt = 'Select the best answer to the following multiple-choice question based on the video and the subtitles. Respond with only the letter (A, B, C, or D) of the correct option.\n'
end_prompt = "Answer with the option's letter from the given choices directly."
option_mapping = {0:'A', 1:'B', 2:'C', 3:'D'}

# loop over all the videos
# loop over the annotation and create the instruct file
all_result = []
mapping = {}
feat_mapping = {}
for ele in origin_file_content:

    vid = ele['video_name']
    video_name = vid + '.mp4'
    # fill in the mapping
    mapping[vid] = video_name
    feat_mapping[vid] = vid + '.pt'
    
    # prepare the questio nand the prompt
    curr_question = start_prompt + ele['question_text'] + '\n'
    
    # prepare the options
    curr_options = ele['multi_choice'] 
    refined_options = "A." + curr_options[0] + '\n' + "B." + curr_options[1] + '\n' + "C." + curr_options[2] + '\n' + "D." + curr_options[3] + '\n'
    curr_question = curr_question + refined_options + end_prompt
    
    curr_answer = option_mapping[ele['answer']]
    
    
    curr_id = ele['id']
    curr_final_conversation = []
    # add the question
    curr_final_conversation.append({
        'from': 'human',
        'value': '<image>\n' + curr_question
    })
    # add the answer 
    curr_final_conversation.append({
        'from': 'gpt',
        'value': curr_answer
    })            
    # add the special token to the first question
    # curr_final_conversation[0]['value'] = '<image>\n' + curr_final_conversation[0]['value'] 
    curr_dict = {'id': curr_id,
                'conversations': curr_final_conversation,
                'video': vid}
    all_result.append(curr_dict)
        

file = open(save_json_file, 'w')
file.write(json.dumps(all_result))
file.close()

file = open(save_mapping_json_file, 'w')
file.write(json.dumps(mapping))
file.close()

file = open(save_feat_mapping_json_file, 'w')
file.write(json.dumps(feat_mapping))
file.close()
