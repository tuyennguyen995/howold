from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage

#Import du doan
# Thêm thư viện TensorFlow và tf.keras
import tensorflow
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
from keras.models import model_from_json
#Thư viện open file
from tkinter import filedialog
from tkinter import *
import os
import base64
from PIL import Image
from io import BytesIO
import shutil


import tensorflow as tf
global graph,model
graph = tf.get_default_graph()

# Load json và tạo model
# Load model age
path = os.getcwd()+"/cam/static/model/"
json_file = open(path +"train/age4/age_model.json", "r")
json_string = json_file.read()
json_file.close()
model = model_from_json(json_string)
model.load_weights(path +"train/age4/weights.30.h5")
names = ["0-2", "15-20", "25-32", "38-43", "4-6", "48-53", "60-100", "8-12"]

# Load model gender
json_file1 = open(path +"train/gender3/gender_model.json", "r")
json_string1 = json_file1.read()
json_file1.close()
model1 = model_from_json(json_string1)
model1.load_weights(path +"train/gender3/weights.28.h5")
names1 = ["Nam", "Nu"]

# Khoi tao moduls phat hien khuon mat
model_file = path +"face/opencv_face_detector_uint8.pb"
config_file = path +"face/opencv_face_detector.pbtxt"
net = cv2.dnn.readNetFromTensorflow(model_file, config_file)


# Create your views here.
def index(request):
    return render(request, 'index.html')

def camera(request):
    return render(request, 'camera.html')
#Upload file
from django.conf import settings
from django.core.files.storage import FileSystemStorage

def upload(request):
    context = {}
    if request.method =='POST':
        #Nhan file tu request
        upload_file = request.FILES['image']
        #Save file
        fs = FileSystemStorage()
        name = fs.save(upload_file.name, upload_file)
        context['url'] = fs.url(name)

        #Du doan
        pre(fs.url(name))

    return render(request, 'result.html', context)

#---------------------------------------------------------------    
# Phat hien khuon mat
def detect_face_open_cv_dnn(net, frame, conf_threshold=0.5):
    frame_opencv_dnn = frame.copy()
    frame_height = frame_opencv_dnn.shape[0]
    frame_width = frame_opencv_dnn.shape[1]
    blob = cv2.dnn.blobFromImage(frame_opencv_dnn, 1.0, (300, 300), [104, 117, 123], False, False)

    net.setInput(blob)
    detections = net.forward()
    bboxes = []
    ajust = 0.17
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frame_width)
            y1 = int(detections[0, 0, i, 4] * frame_height)
            x2 = int(detections[0, 0, i, 5] * frame_width)
            y2 = int(detections[0, 0, i, 6] * frame_height)
            ajust_x = int((x2 - x1) * ajust)
            ajust_y = int((y2 - y1) * ajust)
            x1 -= ajust_x
            x2 += ajust_x
            y1 -= int(ajust_y * 1.5)
            y2 += int(ajust_y * 0.5)
            bboxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame_opencv_dnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 350)), 8)
    return frame_opencv_dnn, bboxes

# Xử lý ảnh đầu vào
def pre(img_url):
    path_img = os.getcwd()+img_url
    img_load = cv2.imread(path_img)  # load ảnh

    out_opencv_dnn, bboxes = detect_face_open_cv_dnn(net, img_load)

    for x1, y1, x2, y2 in bboxes:
        try:
            # Lấy khung ảnh
            a = int((x2-x1)*0.1)
            b = a*2
            dd2 = ((y2 - y1) - (x2 - x1))+int(a*0.65)
            dd1 = int(dd2 / 2)+a
            img = cv2.resize(img_load[y1 - a:y2 + b, x1 - dd1:x2 + dd2], (300, 300))

        except Exception as e:
            print('Không nhận dạng được đối tượng')
            dd2 = ((y2 - y1) - (x2 - x1))
            dd1 = int(dd2 / 2)
            img = cv2.resize(img_load[y1:y2, x1 - dd1:x2 + dd2], (300, 300))

        #Tiền sử lý
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_arr = img_to_array(img)/255
        img_arr = img_arr.reshape((1,) + img_arr.shape)

        #Dự đoán
        with graph.as_default():
            prediction = model.predict(img_arr)[0]
            prediction1 = model1.predict(img_arr)[0]
            pos = np.argmax(prediction)
            pos1 = np.argmax(prediction1)

            #Tính độ chính xác
            max_percent = np.amax(prediction) * 100
            max_percent1 = np.amax(prediction1) * 100

            #Tính tỷ lệ khung Box
            face_width = (x2 - x1) / 260
            box_width = int(face_width * 20)
            marin = int(face_width * 10)

            #Hiển thị khung Box
            cv2.rectangle(out_opencv_dnn, (x1 + marin, y1 - box_width + (y2-y1)), (x2 - marin, y1 + (y2-y1)), (0, 255, 0), box_width)
            cv2.putText(out_opencv_dnn, "{} - ({})".format(names1[pos1], names[pos]), (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, face_width, (255, 255, 255), 2, cv2.LINE_4)
        
    cv2.imwrite(path_img, out_opencv_dnn)


def upload_cam(request):
    contest = {}
    if request.method =='POST':
        #Nhan file tu request
        upload_file = request.POST["image"]

        # Chuyển về ảnh
        image = Image.open(BytesIO(base64.b64decode(upload_file.split(",")[1])))

        filename ="image.png"

        image.save(filename, quality=60)
        shutil.move(path + filename, path+"/media")
        context['url'] = path+"/media"+ filename

        #Du doan
        pre(fs.url(context['url']))

    return render(request, 'result.html', context)