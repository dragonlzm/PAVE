import json

with open("/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/scannet/ScanQA_v1.0_val.json", "r") as file:
    data = json.load(file)

formatted_questions = []

question_id = 1
for entry in data:

    formatted_entry = {
        "question_id":  f"{question_id:06d}",
        "scene_id": entry.get("scene_id", ""),
        "text": entry.get("question", ""),
        "answers": entry.get("answers", []),
        "video": entry.get("scene_id", "")
    }
    formatted_questions.append(formatted_entry)
    question_id += 1

with open("/depot/schaterj/data/3d/work_dir/zhuoming_temp/run_llama/data/video_instruction_tuning/scannet/llava-3d-scanqa_val_question.json", "w") as outfile:
    json.dump(formatted_questions, outfile, indent=4)


