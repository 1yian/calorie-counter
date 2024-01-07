import streamlit as st
from backend import OpenAIVisionAPI
import os
from PIL import Image

# Streamlit app
def main():
    st.title("Yian and Milton's Calorie Counter :D")

    api_key = os.environ.get('OPENAI_API_KEY')
    vision_api = OpenAIVisionAPI(api_key)

    image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    text_query = st.text_area("Enter your notes")

    if st.button("Submit"):
        if image_file is not None:
            image = Image.open(image_file)
            try:
                cals, fats, proteins, carbs = vision_api.query_vision_model(image, text_query)
                st.write(f"Calories: {cals}\nFats: {fats}\nProteins: {proteins}\n Carbohydrates: {carbs}")
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.error("Please upload an image and enter some notes.")

if __name__ == "__main__":
    main()
