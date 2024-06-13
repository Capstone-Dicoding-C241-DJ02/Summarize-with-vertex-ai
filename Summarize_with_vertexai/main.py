from flask import Flask, request, render_template, send_from_directory
import os
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration
from pre import convert_pdf_to_images, extract_text_with_pytesseract, extract_text_with_langchain_pdf, clean_text
import vertexai
from vertexai.language_models import TextGenerationModel

# Initialize Vertex AI
project_id = "submissionmlgc-alansugito"
temperature = 0.2
vertexai.init(project=project_id, location="us-central1", service_account="key.json")
parameters = {
    "temperature": temperature,
    "max_output_tokens": 2048,
    "top_p": 0.8,
    "top_k": 40
}
model = TextGenerationModel.from_pretrained("text-bison@002")

# Define the template for the structured JSON
template = """
{
  "personal_info": {
    "name": "Unknown",
    "email": "Unknown",
    "phone": "Unknown",
    "linkedin": "Unknown",
    "website": "Unknown",
    "location": "Unknown"
  },
  "work_experience": [],
  "projects": [],
  "education": [],
  "volunteer": [],
  "skills": [],
  "tools": [],
  "language": [],
  "certifications": []
}
"""

# Function to start labeling a single resume
def start_labeling(cv):
    response = model.predict(
        f"""
Extract the following information from the resume and format it as JSON:
{template}
Resume Text:
{cv}
If you don't know just type 'Unknown', don't make up an answer. For personal info, ensure the name is in upper-lower case, the phone starts with +62.
""",
        **parameters,
    )
    
    # Get the raw text response from the model
    response_text = response.text

    try:
        # Attempt to find the JSON object within the output
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        json_str = response_text[json_start:json_end]

        # Try to parse the cleaned JSON string
        json_data = json.loads(json_str)
        return json_data
    except json.JSONDecodeError as e:
        print("Failed to parse JSON:", e)
        print("Raw output:", response_text)
        return None

# Load the T5 model and tokenizer
t5_model = T5ForConditionalGeneration.from_pretrained('model')
t5_tokenizer = T5Tokenizer.from_pretrained('model/tokenizer')

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    original_text = None
    summary = None
    json_data = None
    formatted_data = None
    error = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file:
            filename = os.path.join(UPLOAD_FOLDER, 'original.pdf')
            file.save(filename)
            try:
                text_langchain_pdf = extract_text_with_langchain_pdf(filename)
                if len(text_langchain_pdf) > 500:
                    cleaned_text = clean_text(text_langchain_pdf)
                    original_text = cleaned_text
                else:
                    raise ValueError("Insufficient text extracted with LangChain")
            except Exception as e:
                try:
                    images = convert_pdf_to_images(filename)
                    text_pytesseract = extract_text_with_pytesseract(images)
                    cleaned_text = clean_text(text_pytesseract)
                    original_text = cleaned_text
                except Exception as e:
                    return render_template('index.html', error=f"Error processing with PyTesseract: {e}")

            # Summarize the text
            inputs = t5_tokenizer.encode("summarize: " + cleaned_text, return_tensors='pt', max_length=512, truncation=True)
            summary_ids = t5_model.generate(inputs, max_length=600, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = t5_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            
            # Extract JSON data from resume
            json_data = start_labeling(cleaned_text)

            if json_data:
                formatted_data = {
                    "personal_info": json_data.get("personal_info", {}),
                    "work_experience": json_data.get("work_experience", []),
                    "projects": json_data.get("projects", []),
                    "education": json_data.get("education", []),
                    "volunteer": json_data.get("volunteer", []),
                    "skills": json_data.get("skills", []),
                    "tools": json_data.get("tools", []),
                    "certifications": json_data.get("certifications", [])
                }

    return render_template('index.html', original_text=original_text, summary=summary, formatted_data=formatted_data, error=error)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
