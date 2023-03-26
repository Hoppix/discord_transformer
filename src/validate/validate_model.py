from transformers import pipeline

import os

print("cwd is:")
print(os.getcwd())
TRAINED_MODEL_PATH = f"./gpt2-discord_chat"
discord_generator = pipeline('text-generation', model=TRAINED_MODEL_PATH, tokenizer='anonymous-german-nlp/german-gpt2')


def run():
    result = discord_generator('hallo ich bin')
    result_text = result[0]['generated_text']
    print(f"Result : {result_text}")

if __name__ == "__main__":
    raise SystemExit(run())