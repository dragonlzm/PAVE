# This script aims to convert the annotation into the format that could be loaded by COCo

# dict_keys(['info', 'images', 'licenses', 'type', 'annotations'])
# >>> test['info']
# {'description': 'This is stable 1.0 version of the 2014 MS COCO dataset.', 'url': 'http://mscoco.org', 'version': '1.0', 'year': 2014, 'contributor': 'Microsoft COCO group', 'date_created': '2015-01-27 09:11:52.357475'}
# >>> test['images'][0]
# {'license': 3, 'url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg', 'file_name': 'COCO_val2014_000000391895.jpg', 'id': 391895, 'width': 640, 'date_captured': '2013-11-14 11:18:45', 'height': 360}
# >>> test['licenses']
# [{'url': 'http://creativecommons.org/licenses/by-nc-sa/2.0/', 'id': 1, 'name': 'Attribution-NonCommercial-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nc/2.0/', 'id': 2, 'name': 'Attribution-NonCommercial License'}, {'url': 'http://creativecommons.org/licenses/by-nc-nd/2.0/', 'id': 3, 'name': 'Attribution-NonCommercial-NoDerivs License'}, {'url': 'http://creativecommons.org/licenses/by/2.0/', 'id': 4, 'name': 'Attribution License'}, {'url': 'http://creativecommons.org/licenses/by-sa/2.0/', 'id': 5, 'name': 'Attribution-ShareAlike License'}, {'url': 'http://creativecommons.org/licenses/by-nd/2.0/', 'id': 6, 'name': 'Attribution-NoDerivs License'}, {'url': 'http://flickr.com/commons/usage/', 'id': 7, 'name': 'No known copyright restrictions'}, {'url': 'http://www.usa.gov/copyright.shtml', 'id': 8, 'name': 'United States Government Work'}]
# >>> test['type']
# 'captions'
# >>> test['annotations'][0]
# {'image_id': 203564, 'id': 37, 'caption': 'A bicycle replica with a clock as the front wheel.'} # this 'id' is annotation id
# >>> test['annotations'][1]
# {'image_id': 179765, 'id': 38, 'caption': 'A black Honda motorcycle parked in front of a garage.'}
# >>> test['annotations'][2]
# {'image_id': 322141, 'id': 49, 'caption': 'A room with blue walls and a white sink and door.'}                                                                            

import json

# original file 
original_file_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/avsd/mock_test_set4DSTC10-AVSD_from_DSTC7_singref.json'
# sample gt file
sample_gt_file_path = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/pycocoevalcap/example/captions_val2014.json'

original_content = json.load(open(original_file_path))
sample_gt_content = json.load(open(sample_gt_file_path))

new_dict = {}
# directly copy the 'info', 'licenses', 'type'
new_dict['info'] = sample_gt_content['info']
new_dict['licenses'] = sample_gt_content['licenses']
new_dict['type'] = sample_gt_content['type']

# fill in the 'image' and the 'annotation'
all_fake_image_id = []
all_annotation = []

# loop over all the annotation and collect the gt answer
count = 0
for ele in original_content['dialogs']:
    curr_id = ele['image_id']
    curr_answer = ele['dialog'][-1]['answer']
    curr_dict = {'image_id': curr_id, 
                    'id': count, 
                    'caption': curr_answer}
    all_annotation.append(curr_dict)
    all_fake_image_id.append(curr_id)
    count += 1


new_dict['annotations'] = all_annotation

# loop over all the fake image id 
all_fake_image_info = []
for curr_fake_image_id in all_fake_image_id:
    curr_dict = {'license': 3, 
                 'url': 'http://farm9.staticflickr.com/8186/8119368305_4e622c8349_z.jpg', 
                 'file_name': 'COCO_val2014_000000391895.jpg', 
                 'id': curr_fake_image_id, 
                 'width': 640, 
                 'date_captured': '2013-11-14 11:18:45', 
                 'height': 360}
    all_fake_image_info.append(curr_dict)

new_dict['images'] = all_fake_image_info


file = open('coco_version_test_gt.json', 'w')
file.write(json.dumps(new_dict))
file.close()
