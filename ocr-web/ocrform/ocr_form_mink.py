import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
from pdf2image import convert_from_path
import json

pytesseract.pytesseract.tesseract_cmd = '/app/.apt/usr/bin/tesseract'
#pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.1.0/bin/tesseract'

def cleanData(x):
    x = x.replace('\n', '')
    x = x.replace('\x0c', '')
    x = x.replace('~', '')
    x = x.replace('‘', '')
    x = x.replace(',', ' ')
    x = x.replace('.', ' ')
    return x

def cleanData_num(x):
    x = x.replace('\n', '')
    x = x.replace('\x0c', '')
    x = x.replace('~', '')
    x = x.replace('‘', '')
    #x = x.replace(',', '.')
    #x = x.replace('.', ' ')
    return x

def readText(x, lang):


    TH_config = ('--psm 3')
    #TH_config = ('-l tha --psm 6')
    EN_config = ('--psm 6')
    #EN_config = ('-l eng --psm 6')
    Num_config = ('--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789.,')
    TH_EN_config = ('-l tha+eng -c preserve_interword_spaces=1 --oem 1 --psm 6')

    if lang == 'eng':
        #x = pytesseract.image_to_string(x, lang='DilleniaUPC_eng', config=EN_config)
        x = pytesseract.image_to_string(x, lang='eng', config=EN_config)

    if lang == 'tha':
        #x = pytesseract.image_to_string(x, lang='DilleniaUPC_tha', config=TH_config)
        x = pytesseract.image_to_string(x, lang='tha', config=TH_config)

    if lang == '':
        x = pytesseract.image_to_string(x)
    if lang == 'num':
        x = pytesseract.image_to_string(x, lang='eng', config=Num_config)
    if lang == 'tha+eng':
        x = pytesseract.image_to_string(x, lang=lang, config=TH_EN_config)


    x = cleanData(x)
    return x

def readText_num(x):

    Num_config = ('--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789/')
    x = pytesseract.image_to_string(x, lang='eng', config=Num_config)
    x = cleanData_num(x)
    return x

def pdfToIMG(filepath):
    PDF_file = filepath
    pages = convert_from_path(PDF_file, 350)

    image_counter = 1

    for page in pages:
        filename = 'ocrform/form_test/page' + str(image_counter) + '.png'
        page.save(filename, 'PNG')
        image_counter = image_counter + 1

# START
def getMain(filepath):

    pdfToIMG(filepath)
    per = 25
    pixelThreshold = 500

    roi_1 = [[(712, 62), (822, 120), 'text', 'district'],
            [(136, 216), (360, 268), 'text', 'name'],
            [(434, 218), (616, 262), 'number', 'id'],
            [(656, 222), (706, 262), 'number', 'age'],
            [(812, 220), (862, 260), 'text', 'house_no'],
            [(892, 226), (924, 260), 'number', 'village_no'],
            [(974, 218), (1128, 266), 'text', 'subdistrict'],
            [(192, 270), (462, 308), 'text', 'caretaker'],
            [(592, 268), (770, 304), 'text', 'status'],
            [(894, 270), (1088, 302), 'number', 'tel'],
            [(186, 310), (428, 342), 'text', 'UD'],
            [(610, 308), (692, 338), 'number', 'pressure'],
            [(950, 306), (994, 336), 'number', 'sugar'],
            [(138, 346), (206, 374), 'number', 'weight'],
            [(356, 348), (432, 374), 'number', 'height'],
            [(592, 346), (650, 372), 'number', 'waist'],
            [(790, 346), (854, 372), 'number', 'bmi'],
            [(158, 438), (170, 454), 'box', 'group_1_social'],
            [(156, 478), (170, 492), 'box', 'group_2_home'],
            [(158, 516), (168, 532), 'box', 'group_3_bed'],
            [(330, 556), (342, 570), 'box', 'club_Y'],
            [(412, 552), (422, 566), 'box', 'club_N'],
            [(262, 580), (440, 614), 'text', 'occupation'],
            [(476, 594), (486, 602), 'box', 'salary_Y'],
            [(630, 566), (744, 604), 'number', 'salary'],
            [(812, 594), (822, 602), 'box', 'salary_N'],
            [(178, 630), (186, 642), 'box', 'live_1_alone'],
            [(384, 628), (394, 640), 'box', 'live_2_two'],
            [(684, 630), (692, 640), 'box', 'live_3_taker'],
            [(350, 706), (360, 716), 'box', 'exercise_1_often'],
            [(450, 706), (458, 718), 'box', 'exercise_2_sometimes'],
            [(564, 710), (576, 718), 'box', 'exercise_3_never'],
            [(340, 782), (352, 790), 'box', 'eat_1_often'],
            [(440, 782), (450, 790), 'box', 'eat_2_sometimes'],
            [(558, 782), (566, 792), 'box', 'eat_3_never'],
            [(534, 898), (544, 910), 'box', 'cough_Y'],
            [(596, 898), (606, 908), 'box', 'cough_N'],
            [(232, 984), (244, 996), 'box', 'brush_1_no'],
            [(524, 984), (536, 994), 'box', 'brush_2_onetime'],
            [(802, 984), (814, 998), 'box', 'brush_3_two'],
            [(238, 1022), (250, 1032), 'box', 'brush_4_more'],
            [(528, 1022), (542, 1032), 'box', 'brush_5_other'],
            [(630, 1004), (790, 1056), 'text', 'brush_6_note'],
            [(230, 1062), (240, 1070), 'box', 'teethpatse_Y'],
            [(306, 1062), (316, 1070), 'box', 'teethpatse_N'],
            [(498, 1098), (508, 1110), 'box', 'brushteeth_N'],
            [(594, 1098), (604, 1106), 'box', 'brushteeth_Y'],
            [(764, 1064), (972, 1120), 'text', 'brushteeth_tools'],
            [(386, 1138), (396, 1146), 'box', 'risk_1_smoke'],
            [(636, 1138), (648, 1148), 'box', 'risk_2_betel'],
            [(804, 1138), (816, 1148), 'box', 'risk_3_alcohol'],
            [(382, 1160), (472, 1188), 'number', 'num_teeth_usage'],
            [(710, 1156), (780, 1190), 'number', 'num_teeth_other'],
            [(884, 1262), (950, 1290), 'box', 'eye1_N'],
            [(960, 1262), (1024, 1296), 'box', 'eye1_Y'],
            [(1036, 1260), (1190, 1296), 'text', 'eye1_note'],
            [(884, 1300), (944, 1328), 'box', 'eye2_N'],
            [(960, 1302), (1018, 1328), 'box', 'eye2_Y'],
            [(1036, 1302), (1190, 1326), 'text', 'eye2_note'],
            [(886, 1342), (948, 1370), 'box', 'eye3_N'],
            [(960, 1342), (1020, 1370), 'box', 'eye3_Y'],
            [(1042, 1344), (1052, 1352), 'box', 'eye3_L'],
            [(1114, 1346), (1124, 1356), 'box', 'eye3_R'],
            [(884, 1382), (942, 1442), 'box', 'eye4_N'],
            [(956, 1380), (1022, 1448), 'box', 'eye4_Y'],
            [(1042, 1382), (1052, 1394), 'box', 'eye4_L'],
            [(1114, 1384), (1126, 1394), 'box', 'eye4_R'],
            [(890, 1462), (938, 1504), 'box', 'eye5_N'],
            [(960, 1458), (1018, 1516), 'box', 'eye5_Y'],
            [(1042, 1462), (1052, 1472), 'box', 'eye5_L'],
            [(1112, 1462), (1124, 1468), 'box', 'eye5_R'],
            [(336, 1524), (400, 1546), 'text', 'eye_result'],
            [(102, 1572), (112, 1584), 'box', 'NoProblem'],
            [(102, 1612), (114, 1618), 'box', 'Problem']]

    roi_2 = [[(796, 134), (840, 158), 'box', 'depressed_1_N'],
             [(858, 134), (900, 160), 'box', 'depressed_1_Y'],
             [(794, 174), (850, 222), 'box', 'depressed_2_N'],
             [(856, 178), (902, 220), 'box', 'depressed_2_Y'],
             [(926, 100), (932, 110), 'box', 'depressed_normal'],
             [(924, 138), (936, 150), 'box', 'depressed_risk'],
             [(100, 274), (110, 282), 'box', 'knee_normal'],
             [(102, 308), (110, 320), 'box', 'knee_risk'],
             [(532, 386), (582, 420), 'number', 'fall_min'],
             [(628, 392), (666, 422), 'number', 'fall_sec'],
             [(98, 446), (108, 452), 'box', 'fall_1_cant'],
             [(242, 446), (252, 452), 'box', 'fall_2_normal'],
             [(548, 440), (562, 448), 'box', 'fall_3_fall'],
             [(98, 482), (110, 492), 'box', 'fall_4_over'],
             [(694, 690), (812, 714), 'box', 'brain1_1'],
             [(830, 688), (946, 716), 'box', 'brain1_2'],
             [(956, 686), (1080, 716), 'box', 'brain1_3'],
             [(1092, 686), (1190, 712), 'box', 'brain1_4'],
             [(698, 726), (814, 752), 'box', 'brain2_1'],
             [(824, 724), (948, 756), 'box', 'brain2_2'],
             [(958, 724), (1076, 750), 'box', 'brain2_3'],
             [(1090, 726), (1190, 752), 'box', 'brain2_4'],
             [(696, 766), (818, 788), 'box', 'brain3_1'],
             [(828, 766), (948, 792), 'box', 'brain3_2'],
             [(958, 766), (1076, 788), 'box', 'brain3_3'],
             [(1088, 764), (1188, 790), 'box', 'brain3_4'],
             [(694, 804), (818, 832), 'box', 'brain4_1'],
             [(828, 806), (948, 834), 'box', 'brain4_2'],
             [(958, 804), (1080, 832), 'box', 'brain4_3'],
             [(1094, 806), (1188, 826), 'box', 'brain4_4'],
             [(694, 840), (816, 868), 'box', 'brain5_1'],
             [(830, 842), (944, 866), 'box', 'brain5_2'],
             [(958, 840), (1076, 868), 'box', 'brain5_3'],
             [(1092, 842), (1184, 866), 'box', 'brain5_4'],
             [(696, 882), (818, 910), 'box', 'brain6_1'],
             [(826, 882), (946, 908), 'box', 'brain6_2'],
             [(960, 882), (1078, 910), 'box', 'brain6_3'],
             [(1090, 882), (1190, 908), 'box', 'brain6_4'],
             [(694, 922), (816, 946), 'box', 'brain7_1'],
             [(830, 922), (946, 946), 'box', 'brain7_2'],
             [(958, 920), (1082, 950), 'box', 'brain7_3'],
             [(1094, 922), (1190, 946), 'box', 'brain7_4'],
             [(696, 960), (812, 984), 'box', 'brain8_1'],
             [(830, 962), (946, 986), 'box', 'brain8_2'],
             [(956, 960), (1078, 986), 'box', 'brain8_3'],
             [(1092, 962), (1188, 982), 'box', 'brain8_4'],
             [(696, 998), (816, 1028), 'box', 'brain9_1'],
             [(828, 1000), (946, 1026), 'box', 'brain9_2'],
             [(960, 1000), (1078, 1026), 'box', 'brain9_3'],
             [(1092, 1002), (1186, 1022), 'box', 'brain9_4'],
             [(696, 1042), (818, 1070), 'box', 'brain10_1'],
             [(830, 1042), (946, 1066), 'box', 'brain10_2'],
             [(958, 1040), (1076, 1064), 'box', 'brain10_3'],
             [(1092, 1042), (1188, 1062), 'box', 'brain10_4'],
             [(696, 1080), (810, 1142), 'box', 'brain11_1'],
             [(828, 1082), (944, 1134), 'box', 'brain11_2'],
             [(962, 1080), (1074, 1142), 'box', 'brain11_3'],
             [(1090, 1082), (1188, 1140), 'box', 'brain11_4'],
             [(696, 1162), (812, 1218), 'box', 'brain12_1'],
             [(828, 1156), (948, 1216), 'box', 'brain12_2'],
             [(960, 1158), (1078, 1216), 'box', 'brain12_3'],
             [(1092, 1158), (1186, 1218), 'box', 'brain12_4'],
             [(696, 1234), (814, 1328), 'box', 'brain13_1'],
             [(828, 1236), (942, 1330), 'box', 'brain13_2'],
             [(956, 1234), (1082, 1332), 'box', 'brain13_3'],
             [(1090, 1234), (1188, 1330), 'box', 'brain13_4'],
             [(694, 1348), (818, 1378), 'box', 'brain14_1'],
             [(830, 1350), (944, 1372), 'box', 'brain14_2'],
             [(958, 1350), (1074, 1372), 'box', 'brain14_3'],
             [(1090, 1350), (1188, 1374), 'box', 'brain14_4'],
             [(108, 1414), (164, 1436), 'text', 'alzheimers_score'],
             [(344, 1394), (354, 1402), 'box', 'alzheimers_normal'],
             [(578, 1392), (584, 1400), 'box', 'alzheimers_doubt'],
             [(576, 1428), (586, 1436), 'box', 'alzheimers_risk']]

    roi_3 = [[(444, 60), (848, 120), 'text', 'time'],
             [(452, 132), (828, 168), 'text', 'where'],
             [(968, 244), (980, 258), 'box', 'food1_Y'],
             [(1102, 244), (1112, 258), 'box', 'food1_N'],
             [(970, 290), (982, 300), 'box', 'food2_Y'],
             [(1100, 288), (1112, 300), 'box', 'food2_N'],
             [(364, 428), (376, 438), 'box', 'hear1_L'],
             [(560, 430), (578, 440), 'box', 'hear2_R'],
             [(716, 424), (730, 438), 'box', 'hear3_no'],
             [(1014, 430), (1026, 438), 'box', 'hear4_normal'],
             [(924, 518), (942, 530), 'box', 'urine_Y'],
             [(1022, 518), (1032, 530), 'box', 'urine_N'],
             [(806, 574), (1062, 632), 'text', 'screener'],
             [(812, 630), (870, 662), 'text', 'screener_day'],
             [(906, 638), (1052, 676), 'text', 'screener_month'],
             [(1094, 630), (1192, 666), 'text', 'screener_year']]

    #template
    imgQ1 = cv2.imread('ocrform/template/page1.png')
    imgQ2 = cv2.imread('ocrform/template/page2.png')
    imgQ3 = cv2.imread('ocrform/template/page3.png')

    #input
    imgI1 = cv2.imread('ocrform/form_test/page1.png')
    imgI2 = cv2.imread('ocrform/form_test/page2.png')
    imgI3 = cv2.imread('ocrform/form_test/page3.png')

    # Dictionary for getting results
    dictResult = dict()


    # PAGE 1
    h, w, c = imgQ1.shape
    orb = cv2.ORB_create(5000)
    kp1, des1 = orb.detectAndCompute(imgQ1, None)
    kp1_i, des1_i = orb.detectAndCompute(imgI1, None)
    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_1 = bf.match(des1_i,des1)
    matches_1.sort(key=lambda x: x.distance)
    good_1 = matches_1[:int(len(matches_1) * (per / 100))]
    imgMatch = cv2.drawMatches(imgI1, kp1_i, imgQ1, kp1, good_1, None, flags=2)
    #cv2.imshow('Match',imgMatch)
    srcPoints_1 = np.float32([kp1_i[m.queryIdx].pt for m in good_1]).reshape(-1, 1, 2)
    dstPoints_1 = np.float32([kp1[m.trainIdx].pt for m in good_1]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints_1, dstPoints_1, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(imgI1, M, (w, h))
    #cv2.imshow('Scan',imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi_1):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 160, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        #cv2.imshow('Extact1', imgShow)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        if r[2] == 'number':
            x = readText_num(imgCrop)
            dictResult[r[3]] = x
        if r[2] == 'text':
            x = readText(imgCrop, 'tha')
            dictResult[r[3]] = x
        if r[2] == 'box':
            x = readText(imgCrop, 'eng')
            dictResult[r[3]] = x


    # PAGE 2
    h, w, c = imgQ2.shape
    orb = cv2.ORB_create(5000)
    kp2, des2 = orb.detectAndCompute(imgQ2, None)
    kp2_i, des2_i = orb.detectAndCompute(imgI2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_2 = bf.match(des2_i, des2)
    matches_2.sort(key=lambda x: x.distance)
    good_2 = matches_2[:int(len(matches_2) * (per / 100))]
    srcPoints_2 = np.float32([kp2_i[m.queryIdx].pt for m in good_2]).reshape(-1, 1, 2)
    dstPoints_2 = np.float32([kp2[m.trainIdx].pt for m in good_2]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints_2, dstPoints_2, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(imgI2, M, (w, h))
    #cv2.imshow('Scan',imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi_2):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 160, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        #cv2.imshow('Extact2', imgShow)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        if r[2] == 'number':
            x = readText_num(imgCrop)
            dictResult[r[3]] = x
        if r[2] == 'text':
            x = readText(imgCrop, 'tha')
            dictResult[r[3]] = x
        if r[2] == 'box':
            x = readText(imgCrop, 'eng')
            dictResult[r[3]] = x


    # PAGE 3
    h, w, c = imgQ2.shape
    orb = cv2.ORB_create(5000)
    kp3, des3 = orb.detectAndCompute(imgQ3, None)
    kp3_i, des3_i = orb.detectAndCompute(imgI3, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches_3 = bf.match(des3_i, des3)
    matches_3.sort(key=lambda x: x.distance)
    good_3 = matches_3[:int(len(matches_3) * (per / 100))]
    srcPoints_3 = np.float32([kp3_i[m.queryIdx].pt for m in good_3]).reshape(-1, 1, 2)
    dstPoints_3 = np.float32([kp3[m.trainIdx].pt for m in good_3]).reshape(-1, 1, 2)
    M, _ = cv2.findHomography(srcPoints_3, dstPoints_3, cv2.RANSAC, 5.0)
    imgScan = cv2.warpPerspective(imgI3, M, (w, h))
    #cv2.imshow('Scan',imgScan)
    imgShow = imgScan.copy()
    imgMask = np.zeros_like(imgShow)

    for x, r in enumerate(roi_3):
        cv2.rectangle(imgMask, (r[0][0], r[0][1]), (r[1][0], r[1][1]), (0, 160, 0), cv2.FILLED)
        imgShow = cv2.addWeighted(imgShow, 0.99, imgMask, 0.1, 0)
        #cv2.imshow('Extact3', imgShow)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        if r[2] == 'number':
            x = readText_num(imgCrop)
            dictResult[r[3]] = x
        if r[2] == 'text':
            x = readText(imgCrop, 'tha')
            dictResult[r[3]] = x
        if r[2] == 'box':
            x = readText(imgCrop, 'eng')
            dictResult[r[3]] = x


    for keys in ['group_1_social', 'group_2_home', 'group_3_bed', 'club_Y', 'club_N', 'salary_Y', 'salary_N',
                 'live_1_alone', 'live_2_two', 'live_3_taker', 'exercise_1_often', 'exercise_2_sometimes', 'exercise_3_never',
                 'eat_1_often', 'eat_2_sometimes', 'eat_3_never', 'cough_Y', 'cough_N',
                 'brush_1_no', 'brush_2_onetime', 'brush_3_two', 'brush_4_more', 'brush_5_other', 'brush_6_note',
                 'teethpatse_Y', 'teethpatse_N', 'brushteeth_N', 'brushteeth_Y', 'brushteeth_tools',
                 'risk_1_smoke', 'risk_2_betel', 'risk_3_alcohol',
                 'eye1_N', 'eye1_Y', 'eye2_N', 'eye2_Y', 'eye3_N', 'eye3_Y', 'eye3_L', 'eye3_R',
                 'eye4_N', 'eye4_Y', 'eye4_L', 'eye4_R', 'eye5_N', 'eye5_Y', 'eye5_L', 'eye5_R',
                 'NoProblem', 'Problem', 'depressed_1_N', 'depressed_1_Y', 'depressed_2_N', 'depressed_1_Y',
                 'depressed_normal', 'depressed_risk', 'knee_normal', 'knee_risk',
                 'fall_1_cant', 'fall_2_normal', 'fall_3_fall', 'fall_4_over',
                 'brain1_1', 'brain1_2', 'brain1_3', 'brain1_4', 'brain2_1', 'brain2_2', 'brain2_3', 'brain2_4',
                 'brain3_1', 'brain3_2', 'brain3_3', 'brain3_4', 'brain4_1', 'brain4_2', 'brain4_3', 'brain4_4',
                 'brain5_1', 'brain5_2', 'brain5_3', 'brain5_4', 'brain6_1', 'brain6_2', 'brain6_3', 'brain6_4',
                 'brain7_1', 'brain7_2', 'brain7_3', 'brain7_4', 'brain8_1', 'brain8_2', 'brain8_3', 'brain8_4',
                 'brain9_1', 'brain9_2', 'brain9_3', 'brain9_4', 'brain10_1', 'brain10_2', 'brain10_3', 'brain10_4',
                 'brain11_1', 'brain11_2', 'brain11_3', 'brain11_4','brain12_1', 'brain12_2', 'brain12_3', 'brain12_4',
                 'brain13_1', 'brain13_2', 'brain13_3', 'brain13_4', 'brain14_1', 'brain14_2', 'brain14_3', 'brain14_4',
                 'alzheimers_normal', 'alzheimers_doubt', 'alzheimers_risk', 'food1_Y', 'food1_N', 'food2_Y', 'food2_N',
                 'hear1_L', 'hear2_R', 'hear3_no', 'urine_Y', 'urine_N']:
        if dictResult[keys] != '':
            dictResult[keys] = 'Y'

    #results = json.dumps(dictResult, indent=4, ensure_ascii=False)
    myData = dictResult
    return(myData)
    #cv2.imshow('Template', imgI1)
    #cv2.waitKey(0)
