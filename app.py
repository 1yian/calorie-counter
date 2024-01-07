import streamlit as st
import io
from backend import OpenAIVisionAPI
import os
from PIL import Image

# Streamlit app
def main():
    st.title("Yian and Milton's Calorie Counter :D")

    api_key = os.environ.get('OPENAI_API_KEY')
    print(api_key)
    vision_api = OpenAIVisionAPI(api_key)
    

    upload_image_option = st.radio("Choose an option", ("Upload Image", "Take a Picture"))

    if upload_image_option == "Upload Image":
        image_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
        if image_file is not None:
            text_query = st.text_area("Enter your notes")
            if st.button("Submit"):
                if text_query:
                    image = Image.open(image_file)
                    try:
                        result = vision_api.query_vision_model(image, text_query)
                        st.success("Response from GPT-4 Vision:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error("Please enter some notes.")
            else:
                st.warning("Please wait for image upload and notes entry.")

    elif upload_image_option == "Take a Picture":
        st.write("Click below to take a picture:")
        picture = st.camera_input(label="User Uploaded Image")
        if picture is not None:
            text_query = st.text_area("Enter your notes")
            if st.button("Submit"):
                if text_query:
                    picture_bytes = picture.read() 
                    image = Image.open(io.BytesIO(picture_bytes))
                    try:
                        result = vision_api.query_vision_model(image, text_query)
                        st.success("Response from GPT-4 Vision:")
                        st.write(result)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                else:
                    st.error("Please enter some notes.")
            else:
                st.warning("Please wait for picture capture and notes entry.")


if __name__ == "__main__":
    main()
