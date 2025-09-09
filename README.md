# pets
Fork of FastAI pets, updated to run in 2025 + switched to ONNX model instead of Pickle.

Modified to run version of code from here:
https://huggingface.co/spaces/jph00/pets/tree/main

This is a part of the FastAI examples, which no longer runs on Hugging Face due to deprecated features.
Modified to use ONNX instead of PKL file to be more secure.

Runs locally, but model size is too large to upload to GITHUB/Hugging Face.
Can be run locally, generating the model with **train.ipynb**, then running **appy.py**
