# This script aims to convert the raw annotation of the ego-exo to the insturction format.

import json

# prompt
question = "In this video, a human is performing an activity. Your goal is to evaluate the human's skill level based on their technique, consistency, and overall execution of the activity. You will classify the performance into one of the following levels: 1. Novice: Beginner-level performance with noticeable gaps in basic skills, understanding, or execution. 2. Early Expert: Competent performance showing solid foundational skills, but with room for refinement. 3. Intermediate Expert: Advanced performance demonstrating strong technical abilities and strategic understanding, but not yet at mastery. 4. Late Expert: Near-masterful or masterful performance, showcasing exceptional skill, precision, and expertise. Example Output: Early Expert"

# load the raw annotation
origin_anno_file = 'data/video_instruction_tuning/egoexo_origin/annotations/proficiency_demonstrator_train.json'
save_json_file = 'data/video_instruction_tuning/egoexo_origin/annotations/proficiency_demonstrator_train_instruct_short_prompt.json'
original_anno_content = json.load(open(origin_anno_file))

# loop over the annotation and create the instruct file
all_result = []
for ele in original_anno_content['annotations']:
    answer = ele['proficiency_score']
    curr_id = ele['take_uid']
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
    curr_dict = {'id': curr_id,
                'conversations': curr_final_conversation,
                'video': curr_id}
    all_result.append(curr_dict)
        

file = open(save_json_file, 'w')
file.write(json.dumps(all_result))
file.close()
