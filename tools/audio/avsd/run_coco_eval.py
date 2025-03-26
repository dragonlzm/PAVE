import argparse
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap

# annotation_file = 'captions_val2014.json'
# results_file = 'captions_val2014_fakecap_results.json'

# annotation_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/avsd/coco_version_val_gt.json'
# results_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/prediction/llava_onevision_asvd_zero_shot.json'

# gt_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/avsd/coco_version_test_gt.json'
# # results_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/prediction/llava_onevision_asvd_zero_shot.json'

# results_file = '/depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/prediction/llava_onevision_asvd_zero_shot_7B.json'



# create coco object and coco_result object
def coco_eval(arg):

    coco = COCO(arg.gt_file)
    coco_result = coco.loadRes(arg.results_file)

    # create coco_eval object by taking coco and coco_result
    coco_eval = COCOEvalCap(coco, coco_result)

    # evaluate on a subset of images by setting
    # coco_eval.params['image_id'] = coco_result.getImgIds()
    # please remove this line when evaluating the full validation set
    coco_eval.params['image_id'] = coco_result.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f'{metric}: {score:.3f}')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", dest='gt_file', type=str, default=None)
    parser.add_argument("--results-file", dest='results_file', type=str, default=None)

    args = parser.parse_args()

    coco_eval(args)
    
# sample usage:
# python tools/prepare_audio/AVSD/run_coco_eval.py \
# --gt-file /depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/avsd/coco_version_test_gt.json \
# --results-file /depot/schaterj/data/3d/work_dir/zhuoming_temp/storage/data/video_instruction_tuning/prediction/llava_onevision_asvd_zero_shot_7B.json