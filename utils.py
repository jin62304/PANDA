import os
import re
import string
import json
from minutes_writer import *


def read_json(file_path):
    with open(file_path, "r", encoding='utf-8') as f:
        data = json.load(f)

    print("================================")
    print(f"Loading data from {file_path}.")
    print("================================")

    return data


def save_json(obj, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print("================================")
    print(f"Saved at {file_path}.")


def path_finder(task, model_type, prompt, input_type, hist_num=None):
    if hist_num is not None:
        output_dir = f"./results/{task}/{input_type}/{hist_num}/"
    else:
        output_dir = f"./results/{task}/{input_type}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if 'gen' in task:
        input_path = "data/panda_sample.json"
        output_file = f"{model_type}_{prompt}_panda_sample.json"
    elif 'ptag' in task:
        if hist_num is not None:
            input_path = f"./results/gen/{input_type}/{hist_num}/{model_type}_{prompt}_panda_sample.json"
        else:
            input_path = f"./results/gen/{input_type}/{model_type}_{prompt}_panda_sample.json"
        output_file = f"ptag_{model_type}_{prompt}_panda_sample.json"
    elif 'ttag' in task:
        if "utt" in model_type:
            input_path = f"./results/ptag/{input_type}/ptag_{model_type}_panda_sample.json"
        else:  # model responses
            if hist_num is not None:
                input_path = f"./results/ptag/{input_type}/{hist_num}/ptag_{model_type}_{prompt}_panda_sample.json"
            else:
                input_path = f"./results/ptag/{input_type}/ptag_{model_type}_{prompt}_panda_sample.json"
        output_file = f"ttag_{model_type}_{prompt}_panda_sample.json"
    else:  # eval
        if hist_num is not None:
            input_path = f"./results/ttag/{input_type}/{hist_num}/ttag_{model_type}_{prompt}_panda_sample.json"
        else:
            input_path = f"./results/ttag/{input_type}/ttag_{model_type}_{prompt}_panda_sample.json"
        output_file = f"eval_{model_type}_{prompt}_panda_sample.json"

    output_path = os.path.join(output_dir, output_file)

    return input_path, output_path


def api_setup(model):
    if "gpt" in model:
        with open("./openai_api_key.txt", 'r') as f:
            api_key = f.readline().strip()
    elif "gemini" in model:
        with open("./gemini_api_key.txt", 'r') as f:
            api_key = f.readline().strip()
    else:  # llama, mistral ...
        raise ValueError("Refined code for this method will be added further.")

    return api_key


def normalize_text(text):
    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

