import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import OpenAI, ChatCompletion
from dotenv import load_dotenv
import json

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)

# Initialize the OpenAI client with the API key from environment variables
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

def process_openai_response(response):
    # Extract relevant information from the response
    chat_id = response.id
    model = response.model
    created = response.created
    usage = {
        "completion_tokens": response.usage.completion_tokens,
        "prompt_tokens": response.usage.prompt_tokens,
        "total_tokens": response.usage.total_tokens
    }
    choices = []

    for choice in response.choices:
        choice_data = {
            "finish_reason": choice.finish_reason,
            "index": choice.index,
            "logprobs": choice.logprobs,
            "message": {
                "role": choice.message.role,
                "content": choice.message.content,
                "function_call": choice.message.function_call,
                "tool_calls": choice.message.tool_calls
            }
        }
        choices.append(choice_data)

    # Create a dictionary to represent the response in JSON format
    response_dict = {
        "id": chat_id,
        "model": model,
        "created": created,
        "usage": usage,
        "choices": choices
    }

    # Convert the dictionary to a JSON string
    response_json = json.dumps(response_dict, indent=2)
    return response_json



def generate_study_plan(subject, available_time, weekpoints):
    # Create the prompt for ChatGPT
    prompt = (f"Create a study plan for the subject: {subject}. "
              f"The student has {available_time} hours available per week, "
              f"with these weak points: {weekpoints}. "
              "Provide a detailed plan with recommended time and frequency to study.")

    # Call the ChatGPT API
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        model="gpt-3.5-turbo",
    )

    # Log the raw response
    #print("Raw API response:", response)


    return process_openai_response(response)

@app.route('/genPlan', methods=['POST'])
def gen_plan():
    # Get the JSON data from the request
    data = request.json
    subject = data.get('subject')
    available_time = data.get('available_time')
    weekpoints = data.get('weekpoints')

    # Generate the study plan using the GPT API
    raw_response = generate_study_plan(subject, available_time, weekpoints)
    
    # Return the raw response as JSON
    return raw_response #jsonify({'raw_response': raw_response})

if __name__ == '__main__':
    app.run(debug=True)
