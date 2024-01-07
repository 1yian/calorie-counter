import streamlit as st
import io
from backend import OpenAIVisionAPI
import os
from PIL import Image

# Streamlit app
def main():
    st.title("Yian and Milton's Calorie Counter :D")

    api_key = os.environ.get('OPENAI_API_KEY')
    vision_api = OpenAIVisionAPI(api_key)


    upload_image_option = st.radio("Choose an option", ("Take a Picture", "Upload Image",))

    if upload_image_option == "Upload Image":
        image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            text_query = st.text_area("Enter your notes")
            if st.button("Submit"):
                image = Image.open(image_file)
                try:
                    result = vision_api.query_vision_model(image, text_query)
                    cals, proteins, fats, carbs = vision_api.query_vision_model(image, text_query)
                    st.markdown("""
                        **Nutritional Information:**
                        - Calories: {:.2f}
                        - Fats: {:.2f} g
                        - Proteins: {:.2f} g
                        - Carbohydrates: {:.2f} g
                        """.format(cals, fats, proteins, carbs))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please wait for image upload and notes entry.")

    elif upload_image_option == "Take a Picture":
        st.write("Click below to take a picture:")
        picture = st.camera_input(label="User Uploaded Image")
        if picture is not None:
            text_query = st.text_area("Enter your notes")
            if st.button("Submit"):
                picture_bytes = picture.read()
                image = Image.open(io.BytesIO(picture_bytes))
                try:
                    cals, proteins, fats, carbs = vision_api.query_vision_model(image, text_query)
                    st.markdown("""
                        **Nutritional Information:**
                        - Calories: {:.2f}
                        - Fats: {:.2f} g
                        - Proteins: {:.2f} g
                        - Carbohydrates: {:.2f} g
                        """.format(cals, fats, proteins, carbs))
                except Exception as e:
                    st.error(f"An error occurred: {e}")
            else:
                st.warning("Please wait for picture capture and notes entry.")


if __name__ == "__main__":
    main()
