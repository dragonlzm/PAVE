# This script aims to convert the raw test answer annotation to the file format can be used in the llava 3d version
# it will generate two file 'llava3d_sqa3d_test_question.json' and 'llava3d_sqa3d_test_answer.json'


import json

# It need to contains
# 'question_id'
# 'text'
# 'type'

# the raw format
# {'answers': ['piano'], 'object_ids': [], 'object_names': [], 
# 'question': 'What instrument in front of me is ebony and ivory?', 
# 'situation': 'I am standing by the ottoman on my right facing a couple of toolboxes.', 
# 'question_id': 220602000002, 'scene_id': 'scene0050_00', 
# 'position': [0.7110268899979686, -0.03219739162793617, 0, 0, 0, -0.9995736030415032, -0.02919952230128897]}

raw_file_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/scannet/sqa3q_ScanQA_format/SQA_test.json'
raw_file_content = json.load(open(raw_file_path))

# target format
# {'question_id': '220602000002', 'scene_id': 'scene0050_00', 'text': 'piano' , 'video': 'scene0050_00', 'type':}

# create the type mapping
type_mapping = {'What':0, 'Is':1, 'How':2, 'Can':3, 'Which':4, 'Other':5}

# loop over the result and get the new annotation
new_eval_annotation = []
for ele in raw_file_content:
    question_first_word = ele['question'].split(' ')[0]
    if question_first_word in type_mapping:
        question_type = type_mapping[question_first_word]
    else:
        question_type = 5
    
    curr_anno = {
        'question_id': str(ele['question_id']),
        'scene_id':ele['scene_id'],
        'text': ele['answers'][0],
        'video': ele['scene_id'],
        'type': question_type,
    }
    new_eval_annotation.append(curr_anno)
    

# dump the results
file = open('llava3d_sqa3d_test_answer.json', 'w')
file.write(json.dumps(new_eval_annotation))
file.close()



################## handle the question file
# target format is 
#{'question_id': '000001', 'scene_id': 'scene0011_00', 'text': 'What color is the chair in the kitchen?', 'answers': ['dark brown', 'brown'], 'video': 'scene0011_00'}


# loop over the result and get the new annotation
new_eval_annotation = []
for ele in raw_file_content:
    question_first_word = ele['question'].split(' ')[0]
    if question_first_word in type_mapping:
        question_type = type_mapping[question_first_word]
    else:
        question_type = 5
    
    curr_anno = {
        'question_id': str(ele['question_id']),
        'scene_id':ele['scene_id'],
        'text': ele['situation'] + ' ' + ele['question'],
        'answers': ele['answers'],
        'video': ele['scene_id'],
        'type': question_type,
    }
    new_eval_annotation.append(curr_anno)
    

# dump the results
file = open('llava3d_sqa3d_test_question.json', 'w')
file.write(json.dumps(new_eval_annotation))
file.close()
