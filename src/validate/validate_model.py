from transformers import pipeline, AutoTokenizer


TRAINED_MODEL_PATH = f"./gpt2-discord_chat"
MODEL_NAME = "anonymous-german-nlp/german-gpt2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
discord_generator = pipeline('text-generation', model=TRAINED_MODEL_PATH, tokenizer=tokenizer, max_new_tokens=20)


def run():
    while True:
        chat_promp()

def chat_promp():
    print("Chat: ")
    parameter = str(input())
    response = discord_generator(parameter)
    response = parse_response(response)
    print("Chat:" + response)

def parse_response(response):
    result_text = response[0]['generated_text']
    result_text = result_text.split("\n")
    result_text.pop(0)
    return ", ".join(result_text)

if __name__ == "__main__":
    raise SystemExit(run())