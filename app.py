from bottle import get, post, run, view, request, BaseRequest
from PIL import Image
from io import BytesIO
import pytesseract
import base64
from segmentation import image_to_text

BaseRequest.MEMFILE_MAX = 10 * 1024 * 1024 # 10 MB

@get('/')
@view('index')
def index():
    pass

@post('/upload')
def upload():
    data = request.json['image'].replace('data:image/png;base64,', '')
    data = base64.b64decode(data)
    img = Image.open(BytesIO(data))
    return image_to_text(img)

@post('/capture')
def capture():
    data = request.json['image'].replace('data:image/png;base64,', '')
    data = base64.b64decode(data)
    img = Image.open(BytesIO(data))
    img = img.transpose(Image.FLIP_LEFT_RIGHT)
    return image_to_text(img)

run(reloader=True)