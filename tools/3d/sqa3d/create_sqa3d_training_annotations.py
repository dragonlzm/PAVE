# this script aims to convert the origin format of the sqa3d annotation to the format of instructional tuning

import json

# load the raw annotation
origin_anno_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/scannet/sqa3q_ScanQA_format/SQA_train.json'
save_json_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/scannet/sqa3q_ScanQA_format/SQA_train_instruct.json'
original_anno_content = json.load(open(origin_anno_file))

# loop over the annotation and create the instruct file
all_result = []
for ele in original_anno_content:
    question = ele['situation'] + ' ' + ele['question']
    answer = ele['answers'][0]
    video_folder = ele['scene_id']
    curr_id = str(ele['question_id'])
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
                'video': video_folder}
    all_result.append(curr_dict)
        

file = open(save_json_file, 'w')
file.write(json.dumps(all_result))
file.close()
