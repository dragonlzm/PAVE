# This script aims to convert the origin json file to the format which used in the training json file
import json


# For each video, it has 10 question, we will group this 10 question into 5 groups 2 round q-a
conversation_round = 5

origin_json_file = './avsd_train.json'
save_json_file = './avsd_train_instruct.json'

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
all_result = []
for vid in origin_file_content:
    curr_vid_content = origin_file_content[vid]
    curr_conversation_len = len(curr_vid_content['data'])
    split_num = curr_conversation_len // conversation_round
    for i in range(split_num):
        curr_start = i * conversation_round
        curr_end = (i+1) * conversation_round
        curr_origin_conversation = curr_vid_content['data'][curr_start: curr_end]
        # create a sample 
        curr_id = vid + '_' + str(i)
        # filter the refine the samples 
        curr_final_conversation = []
        # refine the conversation 
        for ele in curr_origin_conversation:
            curr_question = ele['question']
            curr_answer = ele['answer']
            # add the question first
            curr_final_conversation.append({
                'from': 'human',
                'value': curr_question
            })
            # add the answer 
            curr_final_conversation.append({
                'from': 'gpt',
                'value': curr_answer
            })            
        # add the special token to the first question
        curr_final_conversation[0]['value'] = '<image>\n' + curr_final_conversation[0]['value'] 
        curr_dict = {'id': curr_id,
            'conversations': curr_final_conversation,
            'video': vid + '.mp4'}
        all_result.append(curr_dict)
        

file = open(save_json_file, 'w')
file.write(json.dumps(all_result))
file.close()

    
