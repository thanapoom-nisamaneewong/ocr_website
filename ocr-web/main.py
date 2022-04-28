from fastapi import FastAPI, Request, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
import pybase64
import json
import pandas as pd
import modelIDcard
import modelAlien
import modelPassport
import modelIDcardV
import modelAlienV
import modelPassportV
from ocrform import ocr_form_mink

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

class ImageType(BaseModel):
    url: str

@app.get("/")
def main(request: Request):
    return templates.TemplateResponse('home.html', context={'request': request})

@app.get("/page1")
def main(request: Request):
    return templates.TemplateResponse('index.html', context={'request': request})


@app.post("/page1")
async def readFile(request: Request, image_upload: UploadFile = Form(...), cardType: str = Form(...)):
    # file upload
    data = await image_upload.read()
    upload_filename = 'static/' + image_upload.filename

    with open(upload_filename, 'wb') as f:
        f.write(data)

    # call model
    if cardType == 'thaiid':
        template_filename = 'static/idcardTemplate.jpg'
        pre_result = modelIDcard.getMain(upload_filename, template_filename)

    elif cardType == 'alien':
        template_filename = 'static/aliencardTemplate.png'
        pre_result = modelAlien.getMain(upload_filename, template_filename)

    elif cardType == 'passport':
        template_filename = 'static/passportTemplete.png'
        pre_result = modelPassport.getMain(upload_filename, template_filename)

    os.remove(upload_filename)

    return pre_result

@app.get("/page2")
def main(request: Request):
    return templates.TemplateResponse('page2.html', context={'request': request})

@app.post("/page2")
async def readFile(request: Request, image_upload: UploadFile = Form(...), cardType: str = Form(...)):
    # file upload
    data = await image_upload.read()
    upload_filename = 'static/' + image_upload.filename

    with open(upload_filename, 'wb') as f:
        f.write(data)

    # call model
    if cardType == 'thaiid':
        template_filename = 'static/idcardTemplate.jpg'
        pre_result = modelIDcardV.getMain(upload_filename, template_filename)
        df = pd.DataFrame([pre_result], columns=pre_result.keys())

    elif cardType == 'alien':
        template_filename = 'static/aliencardTemplate.png'
        pre_result = modelAlienV.getMain(upload_filename, template_filename)
        df = pd.DataFrame([pre_result], columns=pre_result.keys())
    elif cardType == 'passport':
        template_filename = 'static/passportTemplete.png'
        pre_result = modelPassportV.getMain(upload_filename, template_filename)
        df = pd.DataFrame([pre_result], columns=pre_result.keys())
    os.remove(upload_filename)

    return templates.TemplateResponse("page2.html",context={'request':request,'result':df.T.to_html(header=None)})

@app.get("/page3")
def main(request: Request):
    return templates.TemplateResponse('page3.html', context={'request': request})

@app.post("/page3")
async def readFile(request: Request, image_upload: UploadFile = Form(...)):
    # file upload
    data = await image_upload.read()
    upload_filename = 'static/' + image_upload.filename

    with open(upload_filename, 'wb') as f:
        f.write(data)

    # call model
    pre_result = ocr_form_mink.getMain(upload_filename)
    df = pd.DataFrame([pre_result], columns=pre_result.keys())

    os.remove(upload_filename)

    return templates.TemplateResponse("page3.html",context={'request':request,'result':df.T.to_html(header=None)})

if __name__ == '__main__':
    uvicorn.run(app, debug=True)
