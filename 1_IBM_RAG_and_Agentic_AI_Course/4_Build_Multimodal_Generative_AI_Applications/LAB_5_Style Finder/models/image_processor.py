# models/image_processor.py
"""
Module for image processing, encoding, and similarity matching.
This module handles the conversion of images to vectors and finding
similar images in the dataset.
"""

import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50
from PIL import Image
import requests
import base64
from io import BytesIO
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ImageProcessor:
    """
    Handles image processing, encoding, and similarity comparisons.
    """
    
    def __init__(self, image_size=(224, 224), 
                 norm_mean=[0.485, 0.456, 0.406], 
                 norm_std=[0.229, 0.224, 0.225]):
        """
        Initialize the image processor with a pre-trained ResNet50 model.
        
        Args:
            image_size (tuple): Target size for input images
            norm_mean (list): Normalization mean values for RGB channels
            norm_std (list): Normalization standard deviation values for RGB channels
        """
        # TODO: Initialize the device (CPU or GPU)
        # Hint: Use torch.device to determine if CUDA is available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # TODO: Load the pre-trained ResNet50 model and set it to evaluation mode
        # Hint: Use resnet50(pretrained=True) and move it to the device
        self.model = resnet50(pretrained=True).to(self.device)
        self.model.eval()
        
        # TODO: Create the preprocessing pipeline using transforms.Compose
        # The pipeline should resize, convert to tensor, and normalize the image
        self.preprocess = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean, std=norm_std),
        ])
    
    def encode_image(self, image_input, is_url=True):
        """
        Encode an image and extract its feature vector.
        
        Args:
            image_input: URL or local path to the image
            is_url: Whether the input is a URL (True) or a local file path (False)
            
        Returns:
            dict: Contains 'base64' string and 'vector' (feature embedding)
        """
        try:
            # TODO: Load the image based on the input type (URL or file path)
            # Hint: Use requests.get for URLs and Image.open for file paths
            # Don't forget to convert to RGB format
            if is_url:
                # Fetch the image from URL
                response = requests.get(image_input)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                # Load the image from a local file
                image = Image.open(image_input).convert("RGB")
            
            # TODO: Convert image to Base64
            # Hint: Use BytesIO and base64.b64encode
            # This avoids file system I/O and ensures the image is in a compatible format (JPEG).
            buffered = BytesIO() # Create an in-memory buffer
            """
            Why Save First: Understanding Image Conversion to Base64

            When converting an image to Base64, you can't directly encode an image object (like a PIL.Image) because it’s not in raw byte data format. Base64 encoding requires binary data, not a high-level image object.

            The Process:

            Image as Object: When you load an image using PIL, it's an image object, not raw byte data.

            Save to BytesIO: The BytesIO buffer is used to temporarily save the image in a specific format (e.g., JPEG), converting it into the necessary raw byte data.

            Base64 Encoding: Once you have the raw byte data, you can encode it into Base64, which turns the image data into a text-based string.
            Why BytesIO?

            Raw Data: BytesIO acts as an in-memory buffer, allowing the image to be serialized into a byte format (e.g., JPEG).

            Base64 Needs Bytes: Base64 encoding works with raw byte data, not objects or images in memory.

            Without this step, you wouldn't have the raw binary data to pass to the Base64 encoder.

            In short, saving to BytesIO first is essential for converting the image into the proper binary format for Base64 encoding.
            """
            image.save(buffered, format="JPEG") # Save the image in JPEG format to the buffer
            # Encode the image from the buffer to a Base64 string
            # The Base64 encoding allows the image to be represented as a string, suitable for text-based formats (e.g., JSON).
            base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
            
            # TODO: Preprocess the image for ResNet50
            # Hint: Use the preprocess pipeline and add a batch dimension with unsqueeze(0)
            input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            
            # TODO: Extract features using ResNet50 and convert to numpy array
            # Hint: Use torch.no_grad() to disable gradient calculation
            with torch.no_grad():
                features = self.model(input_tensor)

            # Convert features to a NumPy array
            feature_vector = features.cpu().numpy().flatten()
            
            return {"base64": base64_string, "vector": feature_vector}
        except Exception as e:
            print(f"Error encoding image: {e}")
            return {"base64": None, "vector": None}

    def find_closest_match(self, user_vector, dataset):
        """
        Find the closest match in the dataset based on cosine similarity.
        
        Args:
            user_vector: Feature vector of the user-uploaded image
            dataset: DataFrame containing precomputed feature vectors
            
        Returns:
            tuple: (Closest matching row, similarity score)
        """
        try:
            # TODO: Extract all embedding vectors from the dataset
            # Hint: Use np.vstack on the 'Embedding' column
            dataset_vectors = np.vstack(dataset['Embedding'].dropna().values)
            
            # TODO: Calculate cosine similarity between user vector and all dataset vectors
            # Hint: Use cosine_similarity from sklearn and reshape user_vector to (1, -1)
            similarities = cosine_similarity(user_vector.reshape(1, -1), dataset_vectors)
            
            # TODO: Find the index of the most similar vector and its similarity score
            # Hint: Use np.argmax to find the index with the highest similarity
            closest_index = np.argmax(similarities)
            similarity_score = similarities[0][closest_index]
            
            # Retrieve the closest matching row
            closest_row = dataset.iloc[closest_index]
            return closest_row, similarity_score
        except Exception as e:
            print(f"Error finding closest match: {e}")
            return None, None