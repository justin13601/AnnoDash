import os
import time
import openai


def main():
    start_time = time.time()
    # Load your API key from an environment variable or secret management service
    openai.api_key = os.getenv("OPENAI_API_KEY")
    model = 'gpt-3.5-turbo'

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": 'Please list 50 SNOMED CT codes'
            }
        ]
    )

    print(response)
    elapsed_time = time.time() - start_time
    print('--------GPT Ranking Time:', elapsed_time, 'seconds--------')
    return


if __name__ == "__main__":
    main()
