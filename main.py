import numpy as np
import json
import os
import copy
import sys
import argparse
from minutes_writer import MinuteWriter
from setproctitle import setproctitle
from utils import *
from tasks import *
from prompt_template import PromptTemplate
from datetime import datetime
from tqdm import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", default="gen", type=str, required=True)
    parser.add_argument("--model_type", default="chatgpt", type=str, required=True)
    parser.add_argument("--input_path", default="", type=str)
    parser.add_argument("--output_path", default="", type=str)
    parser.add_argument("--prompt", default="vanilla", type=str)
    parser.add_argument("--api_key", default="", type=str)
    parser.add_argument("--input_type", default="h+u", type=str)
    parser.add_argument("--start_idx", default=0, type=int)
    parser.add_argument("--end_idx", default=0, type=int)
    parser.add_argument("--hist_num", default=None, type=int)

    args = parser.parse_args()
    task = args.task

    if args.api_key == "":
        api_key = api_setup(args.model_type)
    else:
        api_key = args.api_key

    # to set in/out paths
    if args.input_path == "":
        input_path, output_path = path_finder(task, args.model_type, args.prompt, args.input_type, args.hist_num)
    else:
        input_path = args.input_path
        output_dir, output_file = os.path.join(args.output_path.split("/")[:-1]), args.output_path.split("/")[-1]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        output_path = os.path.join(output_dir, output_file)

    # data load
    data = read_json(input_path)

    # >>> For debugging an example
    if args.start_idx != 0 and args.end_idx != 0:
        data = data[args.start_idx:args.end_idx]
    else:
        if args.start_idx != 0:
            data = data[args.start_idx:]
        if args.end_idx != 0:
            data = data[:args.end_idx]
    # <<<

    # model + prompting setup
    if 'gen' in task:
        # writer (generator) setup
        Writer = MinuteWriter(api_key, args.model_type)
        prompt_template = PromptTemplate(args.prompt)
        pred_model = Generator(Writer, prompt_template, args.prompt)

        new_data = []
        for datum in tqdm(data):
            contents, reference = pred_model.parse_data(datum, args.input_type, args.hist_num)
            if "vanilla" in args.prompt:
                response = pred_model.r_generate(contents, args.model_type)  # response -> list
                datum["pred"] = response[0]
            else:
                response, reasoning_path = pred_model.r_generate(contents, args.model_type)
                datum["rsn_path"] = reasoning_path
                datum["pred"] = response[0]

            print("================================")
            print(response)

            new_data.append(datum)

        save_json(new_data, output_path)

    # Dialogue labeling using GPT-4o
    elif 'ptag' in task:
        model_type = "gpt-4o-2024-05-13"
        prompt_type = 'ptag'
        Writer = MinuteWriter(api_key, model_type)
        prompt_template = PromptTemplate(prompt_type)

        if "utt" in output_path:
            target = "utt"
        else:
            target = "response"
        tagger = PersonaTagger(Writer, prompt_template, target)

        new_data = []
        for datum in tqdm(data):
            # pre-process
            contents = tagger.parse_data(datum)
            ptag = tagger.tag(contents)
            print("================================")
            print(ptag)

            # new key add
            datum["ptags"] = ptag

            new_data.append(datum)

        save_json(new_data, output_path)

    # persona-topic mapping
    elif 'ttag' in task:
        mapping_file = "data/panda_sample.json"
        tagger = TopicTagger(mapping_file)

        new_data = []
        for idx, datum in enumerate(tqdm(data)):
            persona_list = datum["self_text"] + datum["partner_text"]  # list
            ptags = datum["ptags"]  # list
            mapped_topics = tagger.p2t_mapping(persona_list, ptags)  # {persona : topics} -> topics list [0, 1]

            # print(mapped_topics)

            datum["ttags"] = mapped_topics
            new_data.append(datum)

        save_json(new_data, output_path)

    else:  # eval
        eval_types = ["chrf", "rouge", "f1", "ovs"]
        evaluator = Evaluator(eval_types)

        chrf, rouge, f1, ovs = 0, 0, 0, 0
        norm_chrf, norm_rouge = 0, 0

        # ttags : the set of all topics included in pred (or u)
        # ex) list -> [0, 0, 1, 2]
        u_file = "./results/ttag/h+u/ttag_utt_vanilla_panda_sample.json"
        print("Partner utterance information: ")
        u_data = read_json(u_file)

        # debug
        if args.end_idx != 0:
            u_data = u_data[:args.end_idx]

        assert len(u_data) == len(data)

        new_data = []
        for datum, u_datum in tqdm(zip(data, u_data)):
            idv_scores = dict()

            tagged_u = u_datum["ttags"]
            tagged_pred = datum["ttags"]
            _ovs, ovs_type = evaluator.ovs_eval(tagged_u, tagged_pred)

            ref = datum["utterances"][-1].strip()
            pred = datum["pred"]

            utt = datum["utterances"][-2].strip()
            datum["utterance"] = utt
            datum["utt_ttags"] = tagged_u

            datum["ovs_type"] = ovs_type

            if "f1" not in eval_types:
                scores = evaluator.other_eval(ref, pred)
            else:
                persona_list = datum["self_text"] + datum["partner_text"]
                scores = evaluator.other_eval(ref, pred, persona_list)
                f1 += scores["f1"]

            chrf += scores["chrf"]
            rouge += scores["rouge"]
            ovs += _ovs

            norm_chrf += scores["norm_chrf"]
            norm_rouge += scores["norm_rouge"]

            idv_scores["chrf"] = scores["chrf"]
            idv_scores["rouge"] = scores["rouge"]
            idv_scores["f1"] = scores["f1"]
            idv_scores["ovs"] = _ovs

            datum["scores"] = idv_scores

            new_data.append(datum)

        chrf = round(chrf/len(data), 2)
        rouge = round(rouge/len(data), 2)
        ovs = round(ovs/len(data), 2)
        f1 = round(f1/len(data), 2)

        norm_chrf = round(norm_chrf / len(data), 2)
        norm_rouge = round(norm_rouge / len(data), 2)

        print("###################Reuslts###########################")
        print(f"chrf: {chrf}, rouge: {rouge}, ovs: {ovs}, f1:{f1} | norm_chrf: {norm_chrf}, norm_rouge: {norm_rouge}")
        print("###################Reuslts###########################")

        save_json(new_data, output_path)

    print("================================")
