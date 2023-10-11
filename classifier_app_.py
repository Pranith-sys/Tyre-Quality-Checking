import streamlit as st
import numpy as np
from PIL import Image
import pickle
import pandas as pd


# Load the dumped model
loaded_model = pickle.load(open(r'D:\python\AI\computer vision\image binary classification for tyre\classifier model\classifier_ml_model.sav', 'rb'))

# Create a function for image classification
def image_classification(image_array):
    # Perform classification using the loaded model
    prediction = loaded_model.predict(image_array.reshape(1, -1))
    
    if int(prediction[0]) == 0:
        return 'THIS IS GOOD TYRE'
    else:
        return 'THIS IS DEFECTIVE TYRE' 

def preprocess_image(image):

    # Resize the image to 80x80 pixels
    resized_image = image.resize((80, 80))
    
    # Convert the image to a NumPy array and normalize it
    image_array = np.array(resized_image) / 255.0
    
    # Flatten the image array
    flattened_image = image_array.flatten()  # Make sure this results in 19200 features
    
    return flattened_image


def main():
    # Set the page title
    st.title('Image Classification')
    try:
    # Add a file uploader widget to allow users to upload an image
        uploaded_image = st.file_uploader("Upload an image")
    
        if uploaded_image is not None:
            # Print the name of the uploaded image file to the Python terminal
            st.write("Uploaded image file name:", uploaded_image.name)
        
            # Open the image using Pillow (PIL)
            image = Image.open(uploaded_image)

            preprocessed_image = preprocess_image(image)
        
            # Classify the image
            result = image_classification(preprocessed_image)
           
            # Display the uploaded image
            st.image(image, caption='Uploaded Image', use_column_width=True)
        
            # Display the prediction result
            st.write("RESULT : ", result) #'Please provide either jpg or jpeg file format image....'

    except Exception as e:
        st.error(f"An error occurred: {'Please provide either jpg or jpeg file format image....'}")

if __name__ == '__main__':
    main()
