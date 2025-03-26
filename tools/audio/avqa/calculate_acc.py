# This script load the model prediction for the avqa and calculate the acc

import json
import argparse
import ast
import ipdb


def test(args):
    prediction_path = args.prediction_path
    prediction_content = json.load(open(prediction_path))
    
    count = 0
    for ele in prediction_content:
        curr_gt = ele['answer']
        curr_pred = ele['caption']
        if curr_gt in curr_pred:
            count += 1
        else:
            print('curr_gt:', curr_gt, 'curr_pred:', curr_pred)
            
    print('count:', count, 'len(prediction_content):', len(prediction_content), 'ACC: ', count / len(prediction_content))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-path", dest='prediction_path', type=str, default=None)
    args = parser.parse_args()
    test(args)
