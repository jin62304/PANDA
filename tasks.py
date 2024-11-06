import os
import math
from torchmetrics.text import CHRFScore
from torchmetrics.text.rouge import ROUGEScore
from collections import Counter

from utils import *


class Generator:
    def __init__(self, writer, prompt_template, prompt_type):
        self.writer = writer
        print(self.writer.model)
        self.prompt = prompt_template
        self.prompt_type = prompt_type

    def parse_data(self, datum, input_type, hist_num=None):
        self_persona = datum["self_text"]  # list
        partner_persona = datum["partner_text"]
        dial = datum["utterances"]

        if hist_num is not None:
            dial = dial[:(hist_num * 2)+2]

        history, utterance, reference = dial[:-2], dial[-2].strip(), dial[-1].strip()

        self_persona = "\n".join(self_persona)
        partner_persona = "\n".join(partner_persona)

        if "u_only" in input_type:
            new_history = ""
        else:
            new_history = []
            for idx, utt in enumerate(history):
                if idx % 2 == 0:  # partner
                    new_utt = "Partner: " + utt.strip()
                else:
                    new_utt = "You: " + utt.strip()
                new_history.append(new_utt)

            new_history = "\n".join(new_history)

        system_prompt = self.prompt.prompting(**{'self_persona': self_persona,
                                                 'partner': partner_persona,
                                                 'history': new_history})

        contents = dict()
        contents["system"] = system_prompt
        contents["user"] = "Partner: " + utterance.strip()

        return contents, reference  # dict, str

    def r_generate(self, contents, model_type):
        messages = [
            {"role": "system",
             "content": contents["system"]},
            {"role": "user",
             "content": contents["user"]}
        ]

        if self.prompt_type != 'vanilla':
            if "gpt" in model_type:
                first_response = self.writer.write(messages)
                cot_prompt = contents["system"].split("---")[0] + "\n\n" + self.prompt.cot_prompting(**{'reasoning_path': first_response[0],})

                messages = [
                    {"role": "system",
                     "content": cot_prompt},
                    {"role": "user",
                     "content": contents["user"]}
                ]

                response = self.writer.write(messages, typ=self.prompt_type)

                if "refine" in self.prompt_type:
                    try:
                        response, rsn_path = [response[0].replace("\n", "").split(": ")[-1]], response[0].replace("\n", "").split(": ")[1].split("- ")[0]
                    except:
                        print("Self-refine initial response: ")
                        print(response)
                        print()
                        response, rsn_path = [response[0].split("\n")[0]], response[0].split("\n")[-1]
                else:
                    rsn_path = first_response[0]

                # print("Response: ")
                # print(response)
                # print("Reasoning path: ")
                # print(rsn_path)
                #
                # exit(0)

            elif "llama" in model_type or "mistral" in model_type or "gemma" in model_type:
                # multi-step logic
                # generate reasoning paths
                first_response = self.writer.llama_write(messages)

                cot_prompt = contents["system"].split("---")[0] + "\n\n" + self.prompt.cot_prompting(**{'reasoning_path': first_response[0],
                                                                                                        })
                messages = [
                    {"role": "system",
                     "content": cot_prompt},
                    {"role": "user",
                     "content": contents["user"]}
                ]

                response = self.writer.llama_write(messages, typ=self.prompt_type)


                if "refine" in self.prompt_type:
                    if "gemma" in model_type:
                        try:
                            response, rsn_path = [response[0].replace("\n", "").split("Refined response:")[-1]], response[0].replace("\n", "").split("Refined response:")[0].strip().split("Feedback:")[-1]
                        except:
                            response, rsn_path = [response[0].split("\n\n")[-1]], response[0].split("\n\n")[1]
                    elif "llama" in model_type or "mistral" in model_type:
                        response, rsn_path = [response[0]], response[0]
                    else:  # gpt
                        try:
                            response, rsn_path = [response[0].replace("\n", "").split(": ")[-1]], response[0].replace("\n", "").split(": ")[1].split("- ")[0]
                        except:
                            response, rsn_path = [response[0].split("\n")[-1]], response[0].split("\n")[1]
                else:
                    rsn_path = first_response[0]
            else:  # gemini
                response = ""
        else:  # vanilla
            if "gpt" in model_type:
                response = self.writer.write(messages)
            elif "llama" in model_type or "mistral" in model_type or "gemma" in model_type:
                response = self.writer.llama_write(messages)
            else:  # gemini
                response = ""

        if "vanilla" in self.prompt_type:
            return response
        else:
            return response, rsn_path


class PersonaTagger:
    def __init__(self, writer, prompt_template, target):
        self.writer = writer
        print(self.writer.model)
        self.prompt = prompt_template
        self.tagging_obj = target

    def parse_data(self, datum):
        self_persona = datum["self_text"]  # list
        partner_persona = datum["partner_text"]
        # numbering
        self_persona = [f"{idx}: " + sp for idx, sp in enumerate(self_persona)]
        partner_persona = [f"{idx+len(self_persona)}: " + pp for idx, pp in enumerate(partner_persona)]
        self_persona = "\n".join(self_persona)
        partner_persona = "\n".join(partner_persona)

        if "utt" in self.tagging_obj:
            utterance = datum["utterances"][-2].strip()  # str
        else:  # model response
            utterance = datum["pred"].strip()

        system_prompt = self.prompt.prompting(**{'self_persona': self_persona,
                                                 'partner': partner_persona, })
        contents = dict()
        contents["system"] = system_prompt
        contents["user"] = utterance

        return contents

    def tag(self, contents):
        messages = [
            {"role": "system",
             "content": contents["system"]},
            {"role": "user",
             "content": contents["user"]}
        ]

        response = self.writer.write(messages)

        return response


class TopicTagger:
    def __init__(self, mapping_file_path):
        all_mapping_dict = dict()
        data = read_json(mapping_file_path)
        for datum in data:
            # self
            for st, sl in zip(datum['self_text'], datum['self_labels']):
                st = st.strip()
                if st not in all_mapping_dict.keys():
                    all_mapping_dict[st] = sl  #
            # partner
            for pt, pl in zip(datum['partner_text'], datum['partner_labels']):
                pt = pt.strip()
                if pt not in all_mapping_dict.keys():
                    all_mapping_dict[pt] = pl

        self.all_mapping_dict = all_mapping_dict

    def p2t_mapping(self, persona_list, ptags):
        if len(ptags) > 1:
            print("################")
            print("Noooooooooooo")
            raise ValueError("ptags are more than 1 !!!!!!!!!!!!!!!!!")
        else:
            ptags = [i.strip() for i in ptags[0].split(",")]
        new_list = []
        for idx, p in enumerate(persona_list):
            if str(idx) in ptags:
                new_list.append(p.strip())

        mapped_topics = []
        for persona in new_list:  # persona : int
            if persona in self.all_mapping_dict.keys():
                if type(self.all_mapping_dict[persona]) == int:
                    mapped_topics.append(self.all_mapping_dict[persona])
                else:
                    mapped_topics.extend(self.all_mapping_dict[persona])
            else:
                print("#####################")
                print(f"Unseen persona attribute:{persona}. Check the mapper.")

        return mapped_topics


class Evaluator:
    def __init__(self, eval_types):
        print("eval types: ", eval_types)

        if "chrf" in eval_types:
            self.chrf_metric = CHRFScore()
        if "rouge" in eval_types:
            self.rouge_metric = ROUGEScore()

    def ovs_eval(self, tagged_u, tagged_pred):
        eps = 0.1
        hyper_k = 1.0

        u_count = Counter(tagged_u)
        pred_count = Counter(tagged_pred)

        ovs_list = []
        ovs_type_list = []
        for pc in pred_count:
            if pc in u_count.keys():
                len_u = u_count[pc]
                len_pred = pred_count[pc]
            else:
                len_u = 0
                len_pred = pred_count[pc]

            penalty_w, w_type = self._ovs_penalty(len_u, len_pred)

            ovs = (len_pred / (eps + len_u)) * penalty_w
            ovs_list.append(ovs)
            ovs_type_list.append(w_type)

        # both penalty
        if "off_topic" in ovs_type_list and "qty_exc" in ovs_type_list:
            k = hyper_k
        else:
            k = 1

        if len(ovs_list) == 0:
            OVS = 0.0
        else:
            OVS = self._sigmoid(k * math.log(sum(ovs_list)/len(ovs_list)))

        # print(f"ovs: {OVS}")

        return OVS, ovs_type_list

    def _ovs_penalty(self, len_u, len_pred):
        diff = (len_pred - len_u) + 1
        # off-topic
        if len_u == 0 and len_u < len_pred:
            w = math.exp(diff)
            w = math.log(w)
            typ = "off_topic"
        # excessive-quantity
        elif len_u != 0 and len_u < len_pred:
            w = diff * math.exp(1)
            w = math.log(w)
            typ = "qty_exc"
        else:  # not overuse
            w = math.exp(0)
            typ = "accordant"

        return w, typ

    def _sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def other_eval(self, ref, pred, persona_list=None):
        scores = dict()
        norm_pred = normalize_text(pred)
        norm_ref = normalize_text(ref)

        # fluency
        chrf = self.chrf_metric([pred], [[ref]]).item()
        rouge = self.rouge_metric([pred], [[ref]])['rougeLsum_fmeasure'].item()

        if persona_list is not None:
            f1 = 0
            for p in persona_list:
                att = normalize_text(p.strip())
                _f1 = self.compute_f1(norm_pred, att)
                f1 += _f1
            f1 = f1/len(persona_list)

            scores["f1"] = f1

        norm_chrf = self.chrf_metric([norm_pred], [[norm_ref]]).item()
        norm_rouge = self.rouge_metric([norm_pred], [[norm_ref]])['rougeLsum_fmeasure'].item()

        scores["norm_chrf"] = norm_chrf
        scores["norm_rouge"] = norm_rouge
        scores["chrf"] = chrf
        scores["rouge"] = rouge

        return scores

    def compute_f1(self, prediction, attribute):
        pred_tokens = prediction.split()
        truth_tokens = attribute.split()

        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return int(pred_tokens == truth_tokens)

        common_tokens = set(pred_tokens) & set(truth_tokens)

        # if there are no common tokens then f1 = 0
        if len(common_tokens) == 0:
            return 0

        prec = len(common_tokens) / len(pred_tokens)
        rec = len(common_tokens) / len(truth_tokens)

        return 2 * (prec * rec) / (prec + rec)
