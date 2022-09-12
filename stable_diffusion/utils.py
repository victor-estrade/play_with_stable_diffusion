
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
