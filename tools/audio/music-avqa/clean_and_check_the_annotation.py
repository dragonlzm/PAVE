# This scrip is for checking the annotation of the model
# refine the question by replace the template with the question.


import re
import json

def replace_templates(json_file):
    """
    Replace templates in the 'question_content' of each dictionary with corresponding values from 'templ_values'.

    Args:
        json_file (str): Path to the JSON file containing a list of dictionaries.

    Returns:
        list: Updated list of dictionaries with replaced templates in 'question_content'.
    """
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    for entry in data:
        question_content = entry.get('question_content', '')
        templ_values = entry.get('templ_values', '[]')
        
        # Convert 'templ_values' string into a list
        values = json.loads(templ_values)
        
        # Extract templates from the 'question_content'
        templates = re.findall(r'<(.*?)>', question_content)
        
        # Ensure the number of templates matches the number of values
        if len(templates) != len(values):
            raise ValueError(f"Mismatch in number of templates and values in entry: {entry}")
        
        # Replace each template with the corresponding value
        for template, value in zip(templates, values):
            question_content = question_content.replace(f"<{template}>", value, 1)
        
        # Update the entry with the modified 'question_content'
        entry['question_content'] = question_content

    return data


# json_file = 'avqa-test.json'  # Path to your JSON file
json_file = 'avqa-train.json'

updated_data = replace_templates(json_file)

# Optionally save the updated data to a new JSON file
# with open('updated_avqa-test.json', 'w') as outfile:
#     json.dump(updated_data, outfile, indent=4)
    
with open('updated_avqa-train.json', 'w') as outfile:
    json.dump(updated_data, outfile, indent=4)

print("Templates replaced and saved to 'updated_questions.json'")