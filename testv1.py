from PIL import Image, ImageDraw
import pytesseract
import torch
import torch.nn as nn
import cv2
import numpy as np
import tesserocr
from tesserocr import RIL, PSM, PyTessBaseAPI
import io

def extract_ocr_info(image_path):
    image = Image.open(image_path)
    
    # Extract word-level bounding boxes and other properties
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config='--psm 6')
    
    print(data)

    # Filter out entries where 'text' is None or only whitespace
    filtered_data = [(word, left, top, width, height, bold, font_size) 
                    for word, left, top, width, height, bold, font_size in 
                    zip(data['text'], data['left'], data['top'], data['width'], data['height'], data['fonttype'], data['height'])
                    if word and word.strip()]
    
    return filtered_data

def text_spatial_encoding(ocr_data):
    # Extract relevant information from the data
    ocr_text = [entry[0] for entry in ocr_data]
    bbox_info = [(entry[1], entry[2], entry[3], entry[4]) for entry in ocr_data]
    font_info = [(entry[5], entry[6]) for entry in ocr_data]
    
    # Convert to PyTorch tensors
    N = 1  # Batch size (1 image in this case)
    T = len(ocr_text)  # Number of tokens
    D_t = 300  # Text feature dimension
    
    # For demonstration purposes, use random tensors to represent the text features
    ocr_text_tensor = torch.rand(N, T, D_t)
    
    bbox_info_tensor = torch.tensor([bbox_info], dtype=torch.float32)
    font_info_tensor = torch.tensor([font_info], dtype=torch.float32)  # Using float for now, but can be changed as needed
    
    # Combine text, bounding box, and font information
    combined_features = torch.cat([ocr_text_tensor, bbox_info_tensor, font_info_tensor], dim=-1)
    
    return combined_features

def img2text_spetial_encoding(imagefile):
    # Load the image (Replace 'your_image.png' with the path to your image)
    image = Image.open(imagefile)

    # Perform OCR using pytesseract
    # Note: The following assumes that you have Tesseract correctly installed on your system
    #pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update this path accordingly
    result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)

    # Extract text and bounding box information
    # Note: This is a simplified example; in a real-world application you might want to clean up the text and boxes more thoroughly
    # ocr_text = [word for word in result['text'] if word not in ('', ' ')]
    # bbox_info = list(zip(result['left'], result['top'], result['width'], result['height']))
    # Initialize lists to hold OCR text and bounding box information
    ocr_text = []
    bbox_info = []

    # Filter out empty or whitespace strings and keep track of corresponding bounding boxes
    for word, left, top, width, height in zip(result['text'], result['left'], result['top'], result['width'], result['height']):
        if word not in ('', ' '):
            ocr_text.append(word)
            bbox_info.append((left, top, width, height))

    # Convert to PyTorch tensors
    # Note: For demonstration purposes, random tensors are used to represent the text features
    N = 1  # Batch size (1 image in this case)
    T = len(ocr_text)  # Number of tokens
    D_t = 300  # Text feature dimension
    ocr_text_tensor = torch.rand(N, T, D_t)
    bbox_info_tensor = torch.tensor([bbox_info], dtype=torch.float32)

    # Initialize TextSpatialEncoding layer
    text_spatial_encoding = TextSpatialEncoding(text_dim=D_t)

    # Forward pass
    encoded_features = text_spatial_encoding(ocr_text_tensor, bbox_info_tensor)

    return encoded_features

# Define the TextSpatialEncoding class
class TextSpatialEncoding(nn.Module):
    def __init__(self, text_dim, spatial_dim=4):
        super(TextSpatialEncoding, self).__init__()
        self.fc = nn.Linear(text_dim + spatial_dim, text_dim)
        
    def forward(self, ocr_text, bbox_info):
        combined_features = torch.cat([ocr_text, bbox_info], dim=-1)
        encoded_features = self.fc(combined_features)
        return encoded_features

def ocrimg(imagefile, regmargin=4, debugimagefile="debug.jpg", croppedimagefile="cropped.jpg"):
    #with tesserocr.PyTessBaseAPI(lang='jpn', path='/usr/local/share/tessdata', psm=psm.AUTO) as api:
    with tesserocr.PyTessBaseAPI(lang='jpn+eng', path='./tessdata_best', psm=PSM.AUTO) as api:
        #image = Image.open(imagefile)

        (croppedimage, debugimage, imagebox) = readimg_preprocess(imagefile)
        draw = ImageDraw.Draw(debugimage)

        api.SetImage(croppedimage)
        boxes = api.GetComponentImages(RIL.WORD, True)
        #boxes = api.GetComponentImages(RIL.TEXTLINE, True)
        print(f'Found {len(boxes)} word image components.')
        for i , (im, box, _, _) in enumerate(boxes):
            api.SetRectangle(box['x']-regmargin, box['y']-regmargin, box['w']+2*regmargin, box['h']+2*regmargin)
            ocrResult = api.GetUTF8Text()
            conf = api.MeanTextConf()
            #iterator = api.GetIterator()
            #attr = iterator.WordFontAttributes()
            attr = ""
            print( (u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, confidence: {1}, text: {2}, attr: {3}").format(i, conf, ocrResult, str(attr), **box))
            if ocrResult == '' or ocrResult == ' ':
                draw.rectangle([box['x']-regmargin+imagebox[0], box['y']-regmargin+imagebox[1], box['x']+box['w']+2*regmargin, box['y']+box['h']+2*regmargin],outline=(0, 0, 255))
            else:
                draw.rectangle([box['x']-regmargin+imagebox[0], box['y']-regmargin+imagebox[1], box['x']+box['w']+2*regmargin, box['y']+box['h']+2*regmargin],outline=(255, 0, 0))

        debugimage.save(debugimagefile)
        croppedimage.save(croppedimagefile)
    
        #api.Recognize()
        #iterator = api.GetIterator()
        #print(iterator.WordFontAttributes())

def cv2bgr2pil(cv2bgrimg):
    return Image.fromarray(cv2.cvtColor(cv2bgrimg, cv2.COLOR_BGR2RGB))

def readimg_preprocess(cameraimagefilename):
    img = cv2.imread(cameraimagefilename)
    #imgcopy = img.copy()

    # グレースケールに変換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ヒストグラム平坦化（コントラスト調整）
    enhanced = cv2.equalizeHist(gray)

    # ガウシアンブラーでノイズを削減
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)


    # しきい値処理（threshを調整して、目的に応じた分離を行う）
    #_, edges = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    # Cannyエッジ検出
    edges = cv2.Canny(blurred, 50, 150)

    # 輪郭を検出
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 面積が最大の輪郭を見つける、輪郭に基づいて前景（書類）だけを抽出
    #document = max(contours, key=cv2.contourArea)
    #x, y, w, h = cv2.boundingRect(document)
  
    debugimg = enhanced.copy()

    maxwh = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(debugimg, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if maxwh < w * h:
            maxwh = w * h
            document = (x, y, w, h)

    x, y, w, h = document
    cropped_img = enhanced[y:y+h, x:x+w]

    # 輪郭を描画（デバッグ用）
    #cv2.drawContours(imgcopy, [document], -1, (0, 255, 0), 2)

    # ヒストグラム平坦化（コントラスト調整）
    #imagecv2gray = cv2.equalizeHist(imagecv2gray)

    # 輪郭を検出  
    
    # document = img[y:y + h, x:x + w]

    return (cv2bgr2pil(cropped_img), cv2bgr2pil(debugimg), (x, y, w, h))


#print("Encoded Features:", img2text_spetial_encoding('test_01.jpg'))

#ocr_data = extract_ocr_info('test_01.jpg')
#encoded_features = text_spatial_encoding(ocr_data)
#print(encoded_features.shape)

ocrimg('in/test_01.jpg', regmargin=4, debugimagefile="out/debug_test_01_debugimage.jpg", croppedimagefile="out/debug_test_01_croppedforocr.jpg")
ocrimg('in/IMG_5781.jpeg', debugimagefile="out/debug_IMG_5781_debugimage.jpg", croppedimagefile="out/debug_IMG_5781_croppedforocr.jpg")

