#############
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import numpy as np
import cv2
from PIL import Image

# Define the model architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')
])

# Load model weights
model.load_weights('model_weights.h5')

# Define classes
classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']

# Define pesticides recommendation
pesticides_recommendation = {
    'leaf curl': "Imidachloprid @ 1ml per litre water",
    'yellowish': "Urea @ 1g per litre water",
    'whitefly': "Propenophos or chloropyriphos @ 1 ml/litre water",
    'leaf spot': "Mancozeb @ 0.5 gm/litre water",
    'healthy': "No specific pesticide recommendation needed"
}

def preprocess_image(image):
    # Resize image to match model input shape
    image = cv2.resize(image, (150, 150))
    # Convert image to numpy array
    image = np.array(image)
    # Normalize image
    image = image / 255.0
    return image

def predict(image):
    # Preprocess image
    image = preprocess_image(image)
    # Make prediction
    prediction = model.predict(np.expand_dims(image, axis=0))
    # Get predicted class
    predicted_class = classes[np.argmax(prediction)]
    return predicted_class

def main():
    st.title('Chili Disease Prediction')

    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        prediction = predict(np.array(image))
        st.write(f'Prediction: {prediction}')

        if prediction != 'healthy':
            recommendation = pesticides_recommendation[prediction]
            st.write(f'Recommended Pesticide: {recommendation}')

if __name__ == '__main__':
    main()



####################


# import streamlit as st
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# import numpy as np
# import cv2
# from PIL import Image
# import os
# import google.generativeai as genai

# # Load environment variables
# from dotenv import load_dotenv
# load_dotenv()

# # Configure the GenerativeAI API key
# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# # Define the model architecture
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dropout(0.5),
#     Dense(512, activation='relu'),
#     Dense(5, activation='softmax')
# ])

# # Load model weights
# model.load_weights('model_weights.h5')

# # Define classes
# classes = ['healthy', 'leaf curl', 'leaf spot', 'whitefly', 'yellowish']

# # Define pesticides recommendation
# pesticides_recommendation = {
#     'leaf curl': "Imidachloprid @ 1ml per litre water",
#     'yellowish': "Urea @ 1g per litre water",
#     'whitefly': "Propenophos or chloropyriphos @ 1 ml/litre water",
#     'leaf spot': "Mancozeb @ 0.5 gm/litre water",
#     'healthy': "No specific pesticide recommendation needed"
# }

# def preprocess_image(image):
#     # Resize image to match model input shape
#     image = cv2.resize(image, (150, 150))
#     # Convert image to numpy array
#     image = np.array(image)
#     # Normalize image
#     image = image / 255.0
#     return image

# def predict(image):
#     # Preprocess image
#     image = preprocess_image(image)
#     # Make prediction
#     prediction = model.predict(np.expand_dims(image, axis=0))
#     # Get predicted class
#     predicted_class = classes[np.argmax(prediction)]
#     return predicted_class

# def get_gemini_response(question):
#     model = genai.GenerativeModel("gemini-pro")
#     chat = model.start_chat(history=[])
#     response = chat.send_message(question, stream=True)
#     return response

# def main():
#     st.title('Chili Disease Prediction and Q&A')

#     # Q&A Section
#     st.subheader("Q&A Section")
#     input_qa = st.text_input("Input (Q&A):", key="input_qa")
#     submit_qa = st.button("Ask the Question")

#     if submit_qa and input_qa:
#         response_qa = get_gemini_response(input_qa)
#         st.subheader("Q&A Response:")
#         for chunk_qa in response_qa:
#             st.write(chunk_qa.text)

#     # Image Section
#     st.subheader("Image Section")
#     uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

#     if uploaded_file is not None:
#         image = Image.open(uploaded_file)
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         prediction = predict(np.array(image))
#         st.write(f'Prediction: {prediction}')

#         if prediction != 'healthy':
#             recommendation = pesticides_recommendation[prediction]
#             st.write(f'Recommended Pesticide: {recommendation}')

# if __name__ == '__main__':
#     main()
