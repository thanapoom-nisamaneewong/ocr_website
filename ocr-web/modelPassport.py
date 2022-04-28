import cv2
import numpy as np
import pytesseract
import pybase64
import json

def cleanData(x):
    x = x.replace('\n', '')
    x = x.replace('\x0c', '')
    x = x.replace('~', '')
    x = x.replace('‘', '')
    x = x.replace(',', '.')
    return x

def readText(x, lang):
    #pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
    #pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
    pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'
    #pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.1.0/bin/tesseract'

    TH_config = ('-l tha --psm 6')
    EN_config = ('-l eng --psm 6')
    Num_config = ('--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789')
    TH_EN_config = ('-l tha+eng -c preserve_interword_spaces=1 --oem 1 --psm 6')

    if lang == 'eng':
        x = pytesseract.image_to_string(x, lang=lang, config=EN_config)
    if lang == 'tha':
        x = pytesseract.image_to_string(x, lang=lang, config=TH_config)
    if lang == '':
        x = pytesseract.image_to_string(x)
    if lang == 'num':
        x = pytesseract.image_to_string(x, lang='eng', config=Num_config)
    if lang == 'tha+eng':
        x = pytesseract.image_to_string(x, lang=lang, config=TH_EN_config)

    x = cleanData(x)
    return x

def getMain(upload_path, temp_path):
    per = 25
    pixelThreshold = 500

    roiN = [[(42, 132), (266, 442), ' image', 'photo'],
            [(272, 116), (536, 142), 'text-en', 'lastname_en'],
            [(272, 156), (510, 188), 'text-en', 'firstname_en'],
            [(272, 206), (498, 242), 'text-th', 'name_th'],
            [(268, 258), (360, 282), 'text-en', 'nationality'],
            [(410, 260), (550, 288), 'text-en', 'hbd'],
            [(564, 260), (746, 282), 'text-en', 'idcard'],
            [(270, 308), (314, 334), 'text-en', 'gender'],
            [(408, 310), (496, 336), 'text-en', 'height'],
            [(564, 304), (826, 322), 'text-en', 'place_bd'],
            [(270, 352), (452, 378), 'text-en', 'date_issue'],
            [(560, 352), (900, 380), 'text-en', 'issue_authen'],
            [(268, 396), (438, 430), 'text-en', 'date_expire']]

    orb = cv2.ORB_create(10000)

    imgQ = cv2.imread(temp_path)
    kp1, des1 = orb.detectAndCompute(imgQ, None)

    img = cv2.imread(upload_path)
    kp2, des2 = orb.detectAndCompute(img, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des2, des1)
    matches.sort(key=lambda x: x.distance)
    good = matches[:int(len(matches) * (per / 100))]

    srcPoints = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dstPoints = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    h, w, c = imgQ.shape
    M, _ = cv2.findHomography(srcPoints, dstPoints, cv2.RANSAC, 5.0)
    imgWrap = cv2.warpPerspective(img, M, (w, h))
    imgScan = imgWrap.copy()

    #cropped_image = imgScan[132:42, 442:266]
    #fileFace = 'static/CroppedFace.jpg'
    #cv2.imwrite(fileFace, cropped_image)

    dictResult = dict()

    for x, r in enumerate(roiN):
        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        imgFace = imgWrap[r[0][1]:r[1][1], r[0][0]:r[1][0]]

        if r[2] == 'number':
            x = readText(imgCrop, 'num')
            dictResult[r[3]] = x

        if r[2] == 'text':
            x = readText(imgCrop, '')
            dictResult[r[3]] = x

        if r[2] == 'text-th':
            x = readText(imgCrop, 'tha')
            dictResult[r[3]] = x

        if r[2] == 'text-en':
            x = readText(imgCrop, 'eng')
            dictResult[r[3]] = x

        if r[2] == 'text-th-en':
            x = readText(imgCrop, 'tha+eng')
            dictResult[r[3]] = x


    myData = json.dumps(dictResult, indent=4, ensure_ascii=False)

    base64Encoded = pybase64.standard_b64encode(imgScan)
    jsonData = json.loads(myData)
    #jsonData['image'] = base64Encoded.decode('utf-8')

    #faceImg = cv2.imread(fileFace)
    #photoEncoded = pybase64.standard_b64encode(faceImg)
    #jsonData['faceImage'] = photoEncoded.decode('utf-8')

    return jsonData
