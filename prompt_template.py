import os
from jinja2 import Template


class PromptTemplate:
    def __init__(self, prompt_type):
        if "vanilla" in prompt_type:
            template_path = "./prompts/default.jinja"
        elif "cot" in prompt_type:
            template_path = "./prompts/cot.jinja"
        elif "refine" in prompt_type:
            template_path = "./prompts/refine.jinja"
        elif "decom" in prompt_type:
            template_path = "./prompts/decom.jinja"
        elif "ptag" in prompt_type:
            template_path = "./prompts/ptag.jinja"
        elif "ttag" in prompt_type:
            template_path = "./prompts/ttag.jinja"
        else:
            raise ValueError("prompt type is not valid.")

        with open(template_path, 'r') as fp:
            self.template = Template(fp.read())
        self.prompt = self.template.blocks['prompt']
        # empty_context = self.template.new_context()
    
    def prompting(self, **kwargs):
        context = self.template.new_context(kwargs)
        return ''.join(self.prompt(context)).strip()

    def cot_prompting(self, **kwargs):
        cot_prompt = self.template.blocks['cot']
        context = self.template.new_context(kwargs)
        return ''.join(cot_prompt(context)).strip()


if __name__ == '__main__':
    # template_path = './prompts/default.jinja'
    prompt_type = "vanilla"
