# Training a GPT-2 Model with Discord Chat Data

This repository contains code and instructions for training a [GPT-2](https://github.com/openai/gpt-2) model using Discord chat data. The model can be used to generate new text based on the patterns and styles found in the chat data.

## Requirements

- Python 3.6 or higher
- Some export tool for discord
- [PyTorch](https://pytorch.org/) library
- [transformers](https://huggingface.co/transformers/) library

## Data

To train the model, you'll need a Discord chat log in csv format. This can be obtained by exporting the chat log from Discord in .csv fromat. Store this data in a root directory ``data``
