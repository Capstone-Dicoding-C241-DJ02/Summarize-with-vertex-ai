# Resume Parser and Summarizer

A small Flask app that takes a CV as a PDF and returns two things: a short written summary and the CV's contents as structured fields. It was built in June 2024 as part of the Bangkit Academy capstone for Dicoding Jobs (team C241-DJ02).

The other parts of the capstone are in the same organization: the cloud functions in [`cv-summarize-func`](https://github.com/Capstone-Dicoding-C241-DJ02/cv-summarize-func) and [`cv-scoring-func`](https://github.com/Capstone-Dicoding-C241-DJ02/cv-scoring-func), the [`backend`](https://github.com/Capstone-Dicoding-C241-DJ02/backend) and the [`web-client`](https://github.com/Capstone-Dicoding-C241-DJ02/web-client).

## What it does

1. **Reads the PDF.** Text is extracted with LangChain's `UnstructuredFileLoader`. If that returns fewer than 500 characters, which is what a scanned CV gives, the pages are rendered to images at 300 dpi with `pypdfium2` and read with Tesseract OCR instead.
2. **Cleans the text.** Indonesian phone numbers are normalized to the `62` country code, common OCR mistakes are corrected (`|` and a lone `l` become `I`), and stray symbols are removed.
3. **Summarizes it.** A T5 model writes the summary, using beam search with four beams.
4. **Parses it.** The cleaned text is sent to Vertex AI (`text-bison@002`) with a JSON template. The model fills in personal info, work experience, projects, education, volunteer work, skills, tools, languages and certifications, and is told to answer `Unknown` rather than guess. The JSON object is cut out of the response and parsed.
5. **Shows the result** on one page: the summary and each parsed section.

## Stack

Python, Flask, Hugging Face Transformers (T5), PyTorch, Vertex AI, LangChain, Unstructured, pypdfium2, Tesseract.

## Structure

All code is in `Summarize_with_vertexai/`.

- `main.py` - the Flask app, the T5 summary and the Vertex AI parsing prompt
- `pre.py` - PDF text extraction, the OCR fallback and text cleaning
- `templates/index.html` - upload form and result page
- `requirements.txt` - pinned dependencies

## Running locally

```
cd Summarize_with_vertexai
pip install -r requirements.txt
python main.py
```

The app listens on port 5000. Three things it needs are not in this repository:

- `model/` and `model/tokenizer/` - the T5 weights and tokenizer the app loads at start
- `key.json` - a Google Cloud service account with access to Vertex AI. The project id is set near the top of `main.py` and has to be changed to your own.
- Tesseract, installed on the machine, for the OCR fallback
