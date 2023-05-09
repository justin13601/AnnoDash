import os
import time
import openai
import json
from src.prompts import *


class RankGPT:
    def __init__(self):
        # Load your API key from an environment variable or secret management service
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = 'gpt-3.5-turbo'
        self.system_prompt = ''
        self.user_prompt = ''
        self.response = None

    def prepare_prompt(self, target, choices, metadata):
        keys_to_keep = ['id', 'CODE', 'LABEL']  # , 'SYSTEM', 'SCALE_TYP', 'METHOD_TYP', 'CLASS']
        filtered_choices = [{key: d[key] for key in keys_to_keep} for d in choices]

        code_text = ""
        for each_code in filtered_choices:
            code_text += f"{each_code['id']},{each_code['CODE']},{each_code['LABEL'].strip()}\n"
        self.user_prompt = user_prompt_template.format(
            target=target,
            choices=code_text,
            examples=', '.join(metadata['examples'])
        )
        return

    def execute_rank(self):
        openai.api_key = self.api_key
        self.response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{
                "role": "system",
                "content": self.system_prompt,
            },
                {
                    "role": "user",
                    "content": self.user_prompt
                }],
            temperature=0,
            max_tokens=2000,
        )
        return
