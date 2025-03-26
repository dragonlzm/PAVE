# This script aims to create a annotation subset for the llava-video 178k dataset
# it will select 'sample_per_video' number of QA pair 

import json

# original_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_oe_v0_1_qa_processed.json'
# save_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_academic_oe_v0_1_qa_processed_2pv.json' 

# original_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_youtube_oe_v0_1_qa_processed.json'
# save_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/1_2_m_youtube_oe_v0_1_qa_processed_2pv.json' 

# original_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_oe_v0_1_qa_processed.json'
# save_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_academic_oe_v0_1_qa_processed_2pv.json' 

original_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_youtube_oe_v0_1_qa_processed.json'
save_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/LLaVA_Video_178K/2_3_m_youtube_oe_v0_1_qa_processed_2pv.json' 

sample_per_video = 2


annotation_content = json.load(open(original_file))

from_vid_to_annotation = {}
for ele in annotation_content:
    curr_id = ele['id']
    if curr_id not in from_vid_to_annotation:
        from_vid_to_annotation[curr_id] = []
    if len(from_vid_to_annotation[curr_id]) < sample_per_video:
        from_vid_to_annotation[curr_id].append(ele)

# aggregate all the annotation back to list
all_annos = []
for key in from_vid_to_annotation:
    all_annos += from_vid_to_annotation[key]

# dump the annotation 
file = open(save_file, 'w')
file.write(json.dumps(all_annos))
file.close()


