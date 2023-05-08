import os
import openai
import json
from prompts import *


def main():
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = 'gpt-3.5-turbo'

    response = openai.ChatCompletion.create(
        model=model,
        messages=[{
            "role": "system",
            "content": system_prompt_template,
        },
            {
                "role": "user",
                "content": user_prompt_template
            }],
        temperature=0,
        max_tokens=4000,
    )

    print(response)


if __name__ == '__main__':
    main()
