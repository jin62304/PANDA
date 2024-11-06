import json
import time
import openai
from os import getenv
from prompt_template import PromptTemplate


class MinuteWriter:
    def __init__(self, api_key, model_type):
        self.model_type = model_type
        if 'chatgpt' in model_type:
            openai.api_key = api_key
            self.model = "gpt-3.5-turbo-1106"
        elif 'gpt-4' in model_type:  # for dialogue labeling task
            openai.api_key = api_key
            self.model = model_type
        elif 'gemini' in model_type:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            safety_settings = [
                {
                    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HATE_SPEECH",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_HARASSMENT",
                    "threshold": "BLOCK_NONE"
                },
                {
                    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                    "threshold": "BLOCK_NONE"
                },
            ]

            self.model = genai.GenerativeModel(model_name='gemini-pro',
                                               safety_settings=safety_settings,
                                               )
            # self.model = genai.GenerativeModel('gemini-pro')
        elif 'hf' in model_type:
            from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
            import torch

            model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=True)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            raise ValueError("Select an appropriate model type.")

    def gemini_write(self, content, pt:PromptTemplate, reasoning_path=None, typ=None):
        raise ValueError("Refined code for this method will be added further.")

    def llama_write(self, messages, reasoning_path=None, typ=None):
        raise ValueError("Refined code for this method will be added further.")

    # OpenAI models (ChatGPT ...)
    def write(self, messages, reasoning_path=None, typ=None):
        # multi-step prompts
        if typ is not None:
            print("================================")
            print(f"Prompt type: {typ}")
            print()
        else:
            print("================================")
            print("Prompt type: Vanilla")
            print()

        print("================================")
        print("Msg for API: ")
        print(messages)
        print()

        results = []
        retries = 0
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=messages
                )
                result = response['choices'][0]['message']['content']
                # messages.append(dict(response['choices'][0]['message']))

                results.append(result)

                return results
            except:
                retries += 1
                print(f"Request failed. Retrying ({retries} …")
                time.sleep(2 ** retries)
