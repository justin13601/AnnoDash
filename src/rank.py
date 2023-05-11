import os
import time
import openai
import cohere
import json
from src.prompts import *
from functools import lru_cache


@lru_cache(maxsize=None)
def get_response(model, system_prompt, user_message):
    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": user_message
            }
        ],
        temperature=0,
        max_tokens=2000,
    )
    return response


class RankGPT:
    def __init__(self):
        # Load your API key from an environment variable or secret management service
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.response = None
        self.model = 'gpt-3.5-turbo'
        self.system_prompt = ''
        self.user_prompt = ''

    def prepare_prompt(self, target, choices, metadata):
        keys_to_keep = ['id', 'CODE', 'LABEL']  # , 'SYSTEM', 'SCALE_TYP', 'METHOD_TYP', 'CLASS']
        filtered_choices = [{key: d[key] for key in keys_to_keep} for d in choices]

        code_text = ""
        for each_code in filtered_choices:
            code_text += f"{each_code['id']},{each_code['CODE']},{each_code['LABEL'].strip()}\n"
        self.system_prompt = system_prompt_template
        self.user_prompt = user_prompt_template.format(
            target=target,
            choices=code_text,
            examples=', '.join(metadata['examples'])
        )
        return

    def execute_rank(self):
        openai.api_key = self.api_key
        self.response = get_response(self.model, self.system_prompt, self.user_prompt)
        return


class RankCohere:
    def __init__(self):
        # Load your API key from an environment variable or secret management service
        self.api_key = os.getenv("COHERE_API_KEY")
        self.model = "rerank-multilingual-v2.0"
        self.query = ''
        self.documents = []
        self.result = []

    def prepare_ranker(self, target, choices):
        self.query = target
        self.documents = [d['LABEL'] for d in choices]
        return

    def execute_rank(self):
        co = cohere.Client(self.api_key)
        rerank_hits = co.rerank(query=self.query, documents=self.documents, top_n=len(self.documents), model=self.model)

        for hit in rerank_hits:
            self.result.append(self.documents[hit.index])
        return
