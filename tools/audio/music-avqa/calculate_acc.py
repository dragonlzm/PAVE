import json
import argparse
import ast
import ipdb


def test(args):
    # model.eval()
    # load the prediction
    prediction = json.load(open(args.prediction_path, 'r'))
    
    # load the annotation
    samples = json.load(open(args.annotation_path, 'r'))
    assert len(prediction) == len(samples)
    
    total = 0
    correct = 0
    A_count = []
    A_cmp = []
    V_count = []
    V_loc = []
    AV_ext = []
    AV_count = []
    AV_loc = []
    AV_cmp = []
    AV_temp = []

    for prediction_content, anontation_content in zip(prediction, samples):
        # audio,visual_posi,visual_nega, target, question = sample['audio'].to('cuda'), sample['visual_posi'].to('cuda'),sample['visual_nega'].to('cuda'), sample['label'].to('cuda'), sample['question'].to('cuda')

        # preds_qa,out_match_posi,out_match_nega = model(audio, visual_posi,visual_nega, question)
        # preds = preds_qa
        # _, predicted = torch.max(preds.data, 1)
        # ipdb.set_trace() # check the forward
        predicted = prediction_content['pred'].lower()
        target = prediction_content['answer'].lower()

        # check the correct
        correct_mark = 1 if target in predicted else 0

        total += 1
        correct += correct_mark

        x = anontation_content
        type =ast.literal_eval(x['type'])
        if type[0] == 'Audio':
            if type[1] == 'Counting':
                A_count.append(correct_mark)
            elif type[1] == 'Comparative':
                A_cmp.append(correct_mark)
        elif type[0] == 'Visual':
            if type[1] == 'Counting':
                V_count.append(correct_mark)
            elif type[1] == 'Location':
                V_loc.append(correct_mark)
        elif type[0] == 'Audio-Visual':
            if type[1] == 'Existential':
                AV_ext.append(correct_mark)
            elif type[1] == 'Counting':
                AV_count.append(correct_mark)
            elif type[1] == 'Location':
                AV_loc.append(correct_mark)
            elif type[1] == 'Comparative':
                AV_cmp.append(correct_mark)
            elif type[1] == 'Temporal':
                AV_temp.append(correct_mark)

    print('Audio Counting Accuracy: %.2f %%' % (
            100 * sum(A_count)/len(A_count)))
    print('Audio Cmp Accuracy: %.2f %%' % (
            100 * sum(A_cmp) / len(A_cmp)))
    print('Audio Accuracy: %.2f %%' % (
            100 * (sum(A_count) + sum(A_cmp)) / (len(A_count) + len(A_cmp))))
    print('Visual Counting Accuracy: %.2f %%' % (
            100 * sum(V_count) / len(V_count)))
    print('Visual Loc Accuracy: %.2f %%' % (
            100 * sum(V_loc) / len(V_loc)))
    print('Visual Accuracy: %.2f %%' % (
            100 * (sum(V_count) + sum(V_loc)) / (len(V_count) + len(V_loc))))
    print('AV Ext Accuracy: %.2f %%' % (
            100 * sum(AV_ext) / len(AV_ext)))
    print('AV counting Accuracy: %.2f %%' % (
            100 * sum(AV_count) / len(AV_count)))
    print('AV Loc Accuracy: %.2f %%' % (
            100 * sum(AV_loc) / len(AV_loc)))
    print('AV Cmp Accuracy: %.2f %%' % (
            100 * sum(AV_cmp) / len(AV_cmp)))
    print('AV Temporal Accuracy: %.2f %%' % (
            100 * sum(AV_temp) / len(AV_temp)))

    print('AV Accuracy: %.2f %%' % (
            100 * (sum(AV_count) + sum(AV_loc)+sum(AV_ext)+sum(AV_temp)
                   +sum(AV_cmp)) / (len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp))))

    print('Overall Accuracy: %.2f %%' % (
            100 * correct / total))
    
    print('total:', total, 'Audio:', len(A_count) + len(A_cmp), 'AV:', len(AV_count) + len(AV_loc)+len(AV_ext)+len(AV_temp)+len(AV_cmp), 'V:', (len(V_count) + len(V_loc)))
    
#     print('visual subset:', (
#             100 * (sum(AV_count) + sum(AV_loc)+sum(AV_cmp) + sum(V_count) + sum(V_loc)) / (len(AV_count) + len(AV_loc)+len(AV_cmp))))

    return 100 * correct / total

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-path", dest='prediction_path', type=str, default=None)
    parser.add_argument("--annotation-path", dest='annotation_path', type=str, default=None)

    args = parser.parse_args()

    test(args)



# sample usage:
# python tools/prepare_audio/MUSIC_AVQA/calculate_acc.py --prediction-path /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/prediction/vidit_v5_1_3_lora_music_avqa.json \
# --annotation-path /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/music_avqa/updated_avqa-test.json
