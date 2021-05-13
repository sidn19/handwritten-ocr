from bottle import get, post, run, view, request, BaseRequest
from PIL import Image
from io import BytesIO
import pytesseract
import base64

BaseRequest.MEMFILE_MAX = 10 * 1024 * 1024 # 10 MB

@get('/')
@view('index')
def index():
    pass

@post('/')
def get_text():
    data = request.json['image'].replace('data:image/png;base64,', '')
    data = base64.b64decode(data)
    return pytesseract.image_to_string(Image.open(BytesIO(data)))

run(reloader=True)