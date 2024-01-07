import openai
import base64
from PIL import Image
import io
import re

DEFAULT_PROMPT = "To the best of your ability, come up with a maximum " + \
"likelihood estimate of the macro and calorie count of this meal based on how " + \
"it looks alone. When you come up with your estimate, call the estimate " + \
"function by saying 'estimate(x, p, f, c)' where x is your best guess of " + \
"the calories, p is protein, f is fats, and c is carbs."

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

    def create_prompt(self, notes: str):
        if notes:
            return DEFAULT_PROMPT + f" Here are some notes about the meal: {notes}"
        else:
            return DEFAULT_PROMPT


    def query_vision_model(self, image, notes):
        text_input = self.create_prompt(notes)
        base64_image = self.encode_image(image)
        response = openai.ChatCompletion.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_input},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=900,
            temperature=0.3,
        )
        answer = response.choices[0].message['content']

        macros_found = self.find_function_call(answer)
        if len(macros_found) > 0:
            calories, fats, proteins, carbs = macros_found[0]
        else:
            assert False, f"Could not find anything {answer}"
        return calories, fats, proteins, carbs
