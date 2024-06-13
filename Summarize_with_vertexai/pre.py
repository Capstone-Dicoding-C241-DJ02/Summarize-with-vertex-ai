import os
import re
import pytesseract
from PIL import Image
from io import BytesIO
import pypdfium2 as pdfium
from langchain_community.document_loaders import UnstructuredFileLoader

# # Configure Tesseract path
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path

# # Configure Poppler path
# poppler_path = "C:\\path\\to\\poppler\\bin"  # Modify this path to the directory containing Poppler binaries
# os.environ["PATH"] += os.pathsep + poppler_path

# Increase the maximum image size limit
Image.MAX_IMAGE_PIXELS = None

def convert_pdf_to_images(file_path, scale=300/72):
    try:
        pdf_file = pdfium.PdfDocument(file_path)
        page_indices = [i for i in range(len(pdf_file))]

        renderer = pdf_file.render(
            pdfium.PdfBitmap.to_pil,
            page_indices=page_indices,
            scale=scale,
        )

        list_final_images = []

        for i, image in zip(page_indices, renderer):
            image_byte_array = BytesIO()
            image.save(image_byte_array, format='jpeg', optimize=True)
            image_byte_array = image_byte_array.getvalue()
            list_final_images.append({i: image_byte_array})

        return list_final_images
    except Exception as e:
        raise ValueError(f"Failed to convert PDF to images: {e}")

def extract_text_with_pytesseract(list_dict_final_images):
    image_list = [list(data.values())[0] for data in list_dict_final_images]
    image_content = []

    for image_bytes in image_list:
        image = Image.open(BytesIO(image_bytes))
        raw_text = pytesseract.image_to_string(image)
        # Convert "|" to "I"
        raw_text = raw_text.replace("|", "I")
        image_content.append(raw_text)

    return "\n".join(image_content)

def extract_text_with_langchain_pdf(pdf_file):
    try:
        loader = UnstructuredFileLoader(pdf_file)
        documents = loader.load()
        pdf_pages_content = '\n'.join(doc.page_content for doc in documents)
        return pdf_pages_content
    except Exception as e:
        raise ValueError(f"Error processing PDF with LangChain: {e}")

def clean_text(text):
    """Cleans text by removing unwanted characters, correcting Indonesian phone numbers, and adding spaces after them."""

    isd_code = "62"

    def correct_id_phone_number(phone_number):
        """Helper function to clean and correct individual phone numbers."""
        # Remove all non-digit and non '+' characters
        cleaned_number = re.sub(r'[^\d+]', '', phone_number)

        # If number starts with '(+62)', remove the parentheses and '+'
        if cleaned_number.startswith("(+62)"):
            cleaned_number = cleaned_number[1:-1]  # Remove first and last characters (parentheses)
        # If number starts with '+62', remove leading '+'
        elif cleaned_number.startswith("+62"):
            cleaned_number = cleaned_number[1:]
        # If number starts with '62', it is already correct
        elif cleaned_number.startswith("62"):
            pass
        # If number starts with '0', replace leading '0' with '62'
        elif cleaned_number.startswith("0"):
            cleaned_number = isd_code + cleaned_number[1:]
        # If number starts with neither '62' nor '0', add '62' at the beginning
        else:
            cleaned_number = isd_code + cleaned_number

        return cleaned_number + " "

    # Apply the phone number correction
    text = re.sub(r'\b(\(?\+?62\)?[\d\s-]+)\b', lambda match: correct_id_phone_number(match.group(1)), text)
    text = re.sub(r'\b(0\d[\d\s-]+)\b', lambda match: correct_id_phone_number(match.group(1)), text)
    text = re.sub(r'cid:\d+', 'i', text)  # Replace 'cid:220' with 'i'

    # Replace single lowercase 'l' with uppercase 'I'
    text = re.sub(r'\bl\b', 'I', text)
    
    # Remove any extra spaces introduced by the corrections
    text = re.sub(r'\s+', ' ', text).strip()

    # Additional text cleaning (retain necessary characters)
    text = re.sub(r'[^\w\s.,:;?!/@-]', '', text)

    return text
