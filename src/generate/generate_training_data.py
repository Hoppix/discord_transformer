import os
import logging

import pandas as pd

from sklearn.model_selection import train_test_split


logging.getLogger().setLevel("DEBUG")

TRAINING_DATA_PATH = f"{os.getcwd()}/data/training_data.csv"
VALIDATION_DATA_PATH = f"{os.getcwd()}/data/validation_data.csv"
USER = "Hoppix#6723"

def run():
    chat_df = load_data()
    logging.info(chat_df.head)

    # chat_df = chat_df.loc[chat_df["Author"] == USER]
    chat_df = chat_df.drop(["Attachments", "Reactions"], axis=1)
    chat_df = chat_df.dropna()
    chat_df = chat_df[chat_df["Content"].str.contains("https://") == False]
    chat_df = chat_df[chat_df["Content"].str.contains("http://") == False]
    chat_df
    sentences = chat_df["Content"].tolist()
    sentences = [ str(sentence) for sentence in sentences]

    logging.info(sentences)

    training_sentences, validation_sentences = train_test_split(sentences, test_size=0.2)

    write_sentence_file(training_sentences, TRAINING_DATA_PATH)
    write_sentence_file(validation_sentences, VALIDATION_DATA_PATH)

    

def write_sentence_file(sentences, path):
    with open(path, "w+", encoding="utf8") as file:
        for sentence in sentences:
            file.write(sentence + "\n")

def load_data():
    chat_df = pd.DataFrame()
    with open(f"{os.getcwd()}/data/names.txt", "r+") as file:
        for name in file:
            name = name.strip()
            df = pd.read_csv(f"{os.getcwd()}/data/{name}_messages.csv")
            chat_df = pd.concat([chat_df, df])
    chat_df.to_csv(f"{os.getcwd()}/data/merged_messages.csv")
    return chat_df



if __name__ == "__main__":
    run()