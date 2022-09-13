from PIL import Image

# Pillow is deprecating the top-level resampling attributes (e.g., Image.BILINEAR) in
# favor of the Image.Resampling enum. The top-level resampling attributes will be
# removed in Pillow 10.
if hasattr(Image, "Resampling"):
    _pil_interpolation_to_str = {
        Image.Resampling.NEAREST: 'nearest',
        Image.Resampling.BILINEAR: 'bilinear',
        Image.Resampling.BICUBIC: 'bicubic',
        Image.Resampling.BOX: 'box',
        Image.Resampling.HAMMING: 'hamming',
        Image.Resampling.LANCZOS: 'lanczos',
    }
else:
    _pil_interpolation_to_str = {
        Image.NEAREST: 'nearest',
        Image.BILINEAR: 'bilinear',
        Image.BICUBIC: 'bicubic',
        Image.BOX: 'box',
        Image.HAMMING: 'hamming',
        Image.LANCZOS: 'lanczos',
    }

_str_to_pil_interpolation = {b: a for a, b in _pil_interpolation_to_str.items()}

class PIL_IMAGE_RESAMPLING:
    NEAREST = _str_to_pil_interpolation["nearest"]
    BILINEAR = _str_to_pil_interpolation["bilinear"]
    BICUBIC = _str_to_pil_interpolation["bicubic"]
    BOX = _str_to_pil_interpolation["box"]
    HAMMING = _str_to_pil_interpolation["hamming"]
    LANCZOS = _str_to_pil_interpolation["lanczos"]


def fetch_bytes(url_or_path):
    if str(url_or_path).startswith('http://') or str(url_or_path).startswith('https://'):
        from urllib.request import urlopen 
        return urlopen(url_or_path) 
    return open(url_or_path, 'r')


def fetch_huggingface_key():
    """ Fetch some huggingface token.
    Extracted from some colab notebook
    """
    try:
        with fetch_bytes('https://raw.githubusercontent.com/WASasquatch/easydiffusion/main/key.txt') as f:
            key = f.read().decode('utf-8').split(':')
    except OSError as e:
        print(e)
    huggingface_username = key[0].strip()
    huggingface_token = key[1].strip()
    return huggingface_token
