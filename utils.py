import base64
import io
from PIL import Image

def decode_base64_image(image_base64: str) -> Image.Image:
    try:
        image_data = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        raise ValueError(f"Failed to decode image: {e}")

def extract_text_from_image(image: Image.Image) -> str:
    import pytesseract
    return pytesseract.image_to_string(image)
