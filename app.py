import streamlit as st
import openai
import base64
import os
from PIL import Image
import io

class OpenAIVisionAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def query_vision_model(self, image, text_query):
        base64_image = self.encode_image(image)
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_query},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
	max_tokens=900
        )
        return response.choices[0].message['content']

# Streamlit app
def main():
    st.title("Image and Notes Submission to GPT-4 Vision")

    api_key = os.environ.get('OPENAI_API_KEY')

    vision_api = OpenAIVisionAPI(api_key)

    image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    text_query = st.text_area("Enter your notes")

    if st.button("Submit"):
        if image_file is not None and text_query:
            image = Image.open(image_file)
            try:
                result = vision_api.query_vision_model(image, text_query)
                st.success("Response from GPT-4 Vision:")
                st.write(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload an image and enter some notes.")

if __name__ == "__main__":
    main()

