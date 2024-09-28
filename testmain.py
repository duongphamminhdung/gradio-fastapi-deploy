from typing import Union
import numpy as np
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import base64
from face2stk_rfb320 import detect
import os 

app = FastAPI()
TOKEN = 'hehehe'

# class Image_file(BaseModel):

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        # with open(os.path.join('image', 'a.jpg'), 'wb') as f:
        #     f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()

    return {"message": f"Successfully uploaded {file.filename}"}

# @app.put("/items/{item_id}")
# def update_item(item_id: int, item: Item):
#     return {"item_name": item.name, "item_id": item_id}