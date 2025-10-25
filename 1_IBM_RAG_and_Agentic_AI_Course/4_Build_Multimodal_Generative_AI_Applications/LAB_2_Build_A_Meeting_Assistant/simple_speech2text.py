import os

import requests
import torch
from transformers import pipeline

def download_if_needed(url, filename):
    """
    Downloads a file from a URL only if it doesn't already exist in the current path.

    Args:
        url (str): The URL of the file to download.
        filename (str): The name to save the file as.
    """
    # Check if the file already exists
    if os.path.exists(filename):
        print(f"File '{filename}' already exists. Skipping download. 👍")
    else:
        print(f"File '{filename}' not found. Downloading... 📥")
        try:
            # Send a GET request to the URL
            response = requests.get(url)
            
            # Raise an exception for bad status codes (like 404 Not Found)
            response.raise_for_status()

            # Write the content to the file in binary mode
            with open(filename, "wb") as file:
                file.write(response.content)
            print(f"File '{filename}' downloaded successfully.")

        except requests.exceptions.RequestException as e:
            # Handle any exceptions that occurred during the request
            print(f"Failed to download the file: {e}")

# URL of the audio file and the desired local file name
file_url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/hTqGqoC-LrW6S79HjuJUkg/trimmed-02.wav"
audio_file_path = "sample-meeting.wav"

# Call the function
download_if_needed(file_url, audio_file_path)

pipe = pipeline(
  "automatic-speech-recognition",
  model="openai/whisper-tiny.en",
  chunk_length_s=30,
)
# Define the path to the audio file that needs to be transcribed
sample = audio_file_path

# Perform speech recognition on the audio file
# The `batch_size=8` parameter indicates how many chunks are processed at a time
# The result is stored in `prediction` with the key "text" containing the transcribed text
prediction = pipe(sample, batch_size=8)["text"]
# Print the transcribed text to the console
print(prediction)