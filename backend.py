import openai
import base64
from PIL import Image
import io
import re

class OpenAIVisionAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        openai.api_key = self.api_key

    def encode_image(self, image):
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def find_function_call(self, text):
        # Regex pattern to match the function call with four numbers
        # \s* around the commas allows for optional spaces
        pattern = r"estimate\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)"

        # Find all matches in the text
        matches = re.findall(pattern, text)

        # Process matches
        function_calls = []
        for match in matches:
            # Convert string numbers to integers
            numbers = [int(num) for num in match]
            function_calls.append(numbers)

        return function_calls

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
            max_tokens=900,
            temperature=0.3,
        )
        answer = response.choices[0].message['content']

        calories, fats, proteins, carbs = self.find_function_call(answer)[0]

        return calories, fats, proteins, carbs
