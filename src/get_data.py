"""
Module for downloading the maestro dataset from web.
"""

import requests
from tqdm import tqdm
import zipfile

url = "https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0-midi.zip"

r = requests.get(url, stream=True)
content_length = int(r.headers["Content-length"])

# download data
chunk_size = 10_485_760
with open("maestro-v3.0.0-midi.zip", "wb") as f:
    for chunk in tqdm(
        r.iter_content(chunk_size=chunk_size), total=content_length / chunk_size
    ):
        if chunk:
            f.write(chunk)

# extract data
with zipfile.ZipFile("maestro-v3.0.0-midi.zip", "r") as zip_ref:
    zip_ref.extractall("data")
