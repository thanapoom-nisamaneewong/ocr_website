import cv2
import numpy as np
import pytesseract
import matplotlib.pyplot as plt
pytesseract.pytesseract.tesseract_cmd = r'/usr/local/Cellar/tesseract/5.0.1/bin/tesseract'

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
        x = pytesseract.image_to_string(x, lang='DilleniaUPC_eng', config=Num_config)
    if lang == 'tha+eng':
        x = pytesseract.image_to_string(x, lang=lang, config=TH_EN_config)


    x = cleanData(x)
    return x

def readText_num(x):

    Num_config = ('--psm 13 --oem 3 -c tessedit_char_whitelist=0123456789.,-')
    x = pytesseract.image_to_string(x, lang='DilleniaUPC_eng', config=Num_config)
    x = cleanData_num(x)
    return x


per = 25
pixelThreshold = 500

roi_1 = [[(712, 62), (822, 120), 'text', 'อำเภอ'],
        [(142, 222), (364, 268), 'text', 'name'],
        [(436, 228), (606, 260), 'number', 'id'],
        [(660, 230), (700, 262), 'number', 'age'],
        [(814, 228), (856, 260), 'text', 'number_house'],
        [(896, 238), (916, 262), 'number', 'หมู่'],
        [(978, 228), (1068, 274), 'text', 'ตำบล'],
        [(192, 270), (462, 308), 'text', 'คนดูแล'],
        [(592, 268), (770, 304), 'text', 'status'],
        [(894, 270), (1088, 302), 'number', 'tel'],
        [(186, 310), (428, 342), 'text', 'โรคประจำตัว'],
        [(610, 308), (692, 338), 'number', 'ความดัน'],
        [(950, 306), (994, 336), 'number', 'sugar'],
        [(138, 346), (206, 374), 'number', 'weight'],
        [(356, 348), (432, 374), 'number', 'height'],
        [(592, 346), (650, 372), 'number', 'waist'],
        [(790, 346), (854, 372), 'number', 'bmi'],
        [(158, 438), (170, 454), 'box', 'group_1'],
        [(156, 478), (170, 492), 'box', 'group_2'],
        [(158, 516), (168, 532), 'box', 'group_3'],
        [(330, 556), (342, 570), 'box', 'club_Y'],
        [(412, 552), (422, 566), 'box', 'club_N'],
        [(262, 580), (440, 614), 'text', 'อาชีพ'],
        [(476, 594), (486, 602), 'box', 'salary_Y'],
        [(630, 566), (744, 604), 'number', 'salary'],
        [(812, 594), (822, 602), 'box', 'salary_N'],
        [(178, 630), (186, 642), 'box', 'live_1'],
        [(384, 628), (394, 640), 'box', 'live_2'],
        [(684, 630), (692, 640), 'box', 'live_3'],
        [(350, 706), (360, 716), 'box', 'exercise_1'],
        [(450, 706), (458, 718), 'box', 'exercise_2'],
        [(564, 710), (576, 718), 'box', 'exercise_3'],
        [(340, 782), (352, 790), 'box', 'eat_1'],
        [(440, 782), (450, 790), 'box', 'eat_2'],
        [(558, 782), (566, 792), 'box', 'eat_3'],
        [(534, 898), (544, 910), 'box', 'cough_Y'],
        [(596, 898), (606, 908), 'box', 'cough_N'],
        [(232, 984), (244, 996), 'box', 'brush_1'],
        [(524, 984), (536, 994), 'box', 'brush_2'],
        [(802, 984), (814, 998), 'box', 'brush_3'],
        [(238, 1022), (250, 1032), 'box', 'brush_4'],
        [(528, 1022), (542, 1032), 'box', 'brush_5'],
        [(630, 1004), (790, 1056), 'text', 'brush_6'],
        [(230, 1062), (240, 1070), 'box', 'teethpatse_Y'],
        [(306, 1062), (316, 1070), 'box', 'teethpatse_N'],
        [(498, 1098), (508, 1110), 'box', 'brushteeth_N'],
        [(594, 1098), (604, 1106), 'box', 'brushteeth_Y'],
        [(764, 1064), (972, 1120), 'text', 'tools'],
        [(386, 1138), (396, 1146), 'box', 'risk_1'],
        [(636, 1138), (648, 1148), 'box', 'risk_2'],
        [(804, 1138), (816, 1148), 'box', 'risk_3'],
        [(382, 1160), (472, 1188), 'number', 'num_teeth'],
        [(710, 1156), (780, 1190), 'number', 'num_teeth_2'],
        [(884, 1262), (950, 1290), 'box', 'eye1_N'],
        [(960, 1262), (1024, 1296), 'box', 'eye1_Y'],
        [(1036, 1260), (1190, 1296), 'text', 'eye1_t'],
        [(884, 1300), (944, 1328), 'box', 'eye2_N'],
        [(960, 1302), (1018, 1328), 'box', 'eye2_Y'],
        [(1036, 1302), (1190, 1326), 'text', 'eye2_t'],
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
        [(336, 1524), (400, 1546), 'text', 'consider'],
        [(102, 1572), (112, 1584), 'box', 'NoProblem'],
        [(102, 1612), (114, 1618), 'box', 'Problem']]

roi_2 = [[(796, 134), (840, 158), 'box', 'ซึม_1_N'],
         [(858, 134), (900, 160), 'box', 'ซึม_1_Y'],
         [(794, 174), (850, 222), 'box', 'ซึม_2_N'],
         [(856, 178), (902, 220), 'box', 'ซึม_2_Y'],
         [(926, 100), (932, 110), 'box', 'ซึม_ปกติ'],
         [(924, 138), (936, 150), 'box', 'ซึม_เสี่ยง'],
         [(100, 274), (110, 282), 'box', 'เข่า_ปกติ'],
         [(102, 308), (110, 320), 'box', 'เข่า_เสี่ยง'],
         [(532, 386), (582, 420), 'number', 'ล้มนาที'],
         [(628, 392), (666, 422), 'number', 'ล้มวินาที'],
         [(98, 446), (108, 452), 'box', 'ล้ม1'],
         [(242, 446), (252, 452), 'box', 'ล้ม2'],
         [(548, 440), (562, 448), 'box', 'ล้ม3'],
         [(98, 482), (110, 492), 'box', 'ล้ม4'],
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
         [(1090, 1350), (1188, 1374), 'box', 'brain14_14'],
         [(108, 1414), (164, 1436), 'text', 'score'],
         [(344, 1394), (354, 1402), 'box', 'กลุ่มปกติ'],
         [(578, 1392), (584, 1400), 'box', 'กลุ่มสงสัย'],
         [(576, 1428), (586, 1436), 'box', 'กลุ่มไม่ปกติ']]

roi_3 = [[(444, 60), (848, 120), 'text', 'time'],
         [(452, 132), (828, 168), 'text', 'where'],
         [(968, 244), (980, 258), 'box', 'food1_Y'],
         [(1102, 244), (1112, 258), 'box', 'food1_N'],
         [(970, 290), (982, 300), 'box', 'food2_Y'],
         [(1100, 288), (1112, 300), 'box', 'food2_N'],
         [(364, 428), (376, 438), 'box', 'hear1'],
         [(560, 430), (578, 440), 'box', 'hear2'],
         [(716, 424), (730, 438), 'box', 'hear3'],
         [(1014, 430), (1026, 438), 'box', 'hear4'],
         [(924, 518), (942, 530), 'box', 'ปัสสาวะ1'],
         [(1022, 518), (1032, 530), 'box', 'ปัสสาวะ2'],
         [(806, 574), (1062, 632), 'text', 'editor'],
         [(812, 630), (870, 662), 'text', 'day'],
         [(906, 638), (1052, 676), 'text', 'month'],
         [(1094, 630), (1192, 666), 'text', 'year']]
#template
imgQ1 = cv2.imread('template/page1.png')
imgQ2 = cv2.imread('template/page2.png')
imgQ3 = cv2.imread('template/page3.png')

#input
imgI1 = cv2.imread('form_test/test1.png')
imgI2 = cv2.imread('form_test/test2.png')
imgI3 = cv2.imread('form_test/test3.png')

h, w, c = imgQ1.shape

orb = cv2.ORB_create(5000)
kp1, des1 = orb.detectAndCompute(imgQ1, None)
kp2, des2 = orb.detectAndCompute(imgQ2, None)
kp3, des3 = orb.detectAndCompute(imgQ3, None)

kp1_i, des1_i = orb.detectAndCompute(imgI1, None)
kp2_i, des2_i = orb.detectAndCompute(imgI2, None)
kp3_i, des3_i = orb.detectAndCompute(imgI3, None)

bf=cv2.cv2.BFMatcher(cv2.NORM_HAMMING)
matches_1 = bf.match(des1_i,des1)
matches_1.sort(key=lambda x: x.distance)
matches_2 = bf.match(des2_i,des2)
matches_2.sort(key=lambda x: x.distance)
matches_3 = bf.match(des3_i,des3)
matches_3.sort(key=lambda x: x.distance)

good_1 = matches_1[:int(len(matches_1) * (per / 100))]
good_2 = matches_2[:int(len(matches_2) * (per / 100))]
good_3 = matches_3[:int(len(matches_3) * (per / 100))]

imgMatch =cv2.drawMatches(imgI1,kp1_i,imgQ1,kp1,good_1,None,flags=2)
#cv2.imshow('Match',imgMatch)

srcPoints_1 = np.float32([kp1_i[m.queryIdx].pt for m in good_1]).reshape(-1, 1, 2)
dstPoints_1 = np.float32([kp1[m.trainIdx].pt for m in good_1]).reshape(-1, 1, 2)

srcPoints_2 = np.float32([kp2_i[m.queryIdx].pt for m in good_2]).reshape(-1, 1, 2)
dstPoints_2 = np.float32([kp2[m.trainIdx].pt for m in good_2]).reshape(-1, 1, 2)

srcPoints_3 = np.float32([kp3_i[m.queryIdx].pt for m in good_3]).reshape(-1, 1, 2)
dstPoints_3 = np.float32([kp3[m.trainIdx].pt for m in good_3]).reshape(-1, 1, 2)

M, _ = cv2.findHomography(srcPoints_1, dstPoints_1, cv2.RANSAC, 5.0)
imgScan = cv2.warpPerspective(imgI1, M, (w, h))
#cv2.imshow('Scan',imgScan)

imgShow = imgScan.copy()
imgMask = np.zeros_like(imgShow)

dictResult = dict()
for x,r in enumerate(roi_1):
        cv2.rectangle(imgMask,(r[0][0],r[0][1]),(r[1][0],r[1][1]),(0,160,0),cv2.FILLED)
        imgShow=cv2.addWeighted(imgShow,0.99,imgMask,0.1,0)
        cv2.imshow('Extact', imgShow)

        imgCrop = imgScan[r[0][1]:r[1][1], r[0][0]:r[1][0]]
        if r[2] =='number':
            x=readText_num(imgCrop)
            dictResult[r[3]] = x
        if r[2] =='text':
            x=readText(imgCrop,'tha')
            dictResult[r[3]] = x
        if r[2] =='box':
            x=readText(imgCrop,'eng')
            dictResult[r[3]] = x

print(dictResult)
#cv2.imshow('Template', imgI1)
cv2.waitKey(0)



