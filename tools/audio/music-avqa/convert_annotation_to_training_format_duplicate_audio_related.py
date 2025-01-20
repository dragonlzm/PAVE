# This script aims to convert the origin json file to the format which used in the training json file
import json
import os


origin_json_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/music_avqa/updated_avqa-train.json'
save_json_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/music_avqa/music_avqa_train_instruct_duplicate_audio.json'

origin_file_content = json.load(open(origin_json_file))

# the format we need 
# {'id': 'v_nw8MpBwL508', 'conversations': [{'from': 'human', 'value': '<image>\nWhere does the video take place?'}, 
#                                           {'from': 'gpt', 'value': 'The video takes place in a large indoor arena filled with spectators.'}, 
#                                           {'from': 'human', 'value': 'What is the gymnast wearing?'}, 
#                                           {'from': 'gpt', 'value': 'The gymnast is wearing a purple leotard with white sleeves.'}, 
#                                           {'from': 'human', 'value': 'What event is being showcased in the video?'}, 
#                                           {'from': 'gpt', 'value': "The event being showcased is the '2014 P&G CHAMPIONSHIPS'."}, 
#                                           {'from': 'human', 'value': 'What are some of the movements the gymnast performs on the balance beam?'}, 
#                                           {'from': 'gpt', 'value': 'The gymnast performs a split leap, a handstand, a back walkover, a high leap with one leg extended, a back handspring, and various controlled movements including steps and poses.'}, 
#                                           {'from': 'human', 'value': "How does the audience react to the gymnast's performance?"}, 
#                                           {'from': 'gpt', 'value': "The audience watches intently and remains captivated by the gymnast's performance."}], 
#  'data_source': '1_2_m_academic_v0_1', 
#  'video': 'academic_source/activitynet/v_nw8MpBwL508.mp4'}

# loop over all the videos
# loop over the annotation and create the instruct file
all_result = []
for ele in origin_file_content:
    question = ele['question_content']
    answer = ele['anser']
    vid = ele['video_id']
    video_sub_folder = video_subfolder = 'MUSIC-AVQA-videos-Real' if vid[0].isdigit() else 'MUCIS-AVQA-videos-Synthetic'
    video_name = vid + '.mp4'
    curr_id = ele['question_id']
    curr_final_conversation = []
    # add the question
    curr_final_conversation.append({
        'from': 'human',
        'value': '<image>\n' + question
    })
    # add the answer 
    curr_final_conversation.append({
        'from': 'gpt',
        'value': answer
    })            
    # add the special token to the first question
    # curr_final_conversation[0]['value'] = '<image>\n' + curr_final_conversation[0]['value'] 
    curr_dict = {'id': curr_id,
                'conversations': curr_final_conversation,
                'video': vid}
    all_result.append(curr_dict)
    # duplicate the anno if the anno is audio related
    if 'Audio' in ele['type']:
        all_result.append(curr_dict)
        

file = open(save_json_file, 'w')
file.write(json.dumps(all_result))
file.close()
