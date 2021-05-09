#1
!pip install mediapipe opencv-python
!pip install mediapipe

#2
import mediapipe as mp
import cv2
import numpy as np
import uuid 
import os.path as ops
import numpy as np
import torch
import cv2
import time
import os
import matplotlib.pylab as plt
import sys
from tqdm import tqdm
import imageio
from google.colab.patches import cv2_imshow

#3
from google.colab import drive
drive.mount('/content/gdrive')

#4
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

#5
def ko_face_detection(count):
  with mp_face_detection.FaceDetection(
     min_detection_confidence=0.1) as face_detection:
   for name, image in images.items():
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
     results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

     annotated_image = image.copy()
     for detection in results.detections:
      # print('Nose tip:')
      # print(mp.python.solutions.face_detection.get_key_point(
      #     detection, mp_face_detection.FaceKeyPoints.NOSE_TIP))
       mp_drawing.draw_detection(annotated_image, detection)
       cv2_imshow(annotated_image)
      
#6
def video2segement(video_path):
    # video to frames
    vidcap = cv2.VideoCapture(video_path)
    count = 1
    success = True
    
    while success:
      success,image = vidcap.read()
      cv2.imwrite("/content/gdrive/My Drive/final_hw/before/frame%d.jpg" %count, image)
      print("saved image %d.jpg" %count)
  
      if cv2.waitKey(10) == 27:                    
         break
      count += 1
      if count == 1000:
        break
    pass

  #7
  def seg2video():
  pathOut = "/content/gdrive/My Drive/final_hw/output2.mp4" # will change number of output
  frame_array = []
  fps = 30
  for i in range(1,1000):
     img = cv2.imread("/content/gdrive/My Drive/final_hw/after/frame%d.jpg"%i)
     height, width, layers = img.shape
     size = (width,height)
     frame_array.append(img)

  out = cv2.VideoWriter(pathOut,cv2.VideoWriter_fourcc(*'DIVX'), fps, size)

  for i in range(len(frame_array)):
     # writing to a image array
     out.write(frame_array[i])
  out.release()
  pass

#8
from google.colab import files

uploaded = files.upload()

# Read images with OpenCV.
images = {name: cv2.imread(name) for name in uploaded.keys()}

# Preview the images.
for name, image in images.items():
  print(name)   
  cv2_imshow(image)
  
  
#9
with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
  for name, image in images.items():
    # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # Draw face detections of each face.
    print(f'Face detections of {name}:')
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      # print('Nose tip:')
      # print(mp.python.solutions.face_detection.get_key_point(
      #     detection, mp_face_detection.FaceKeyPoints.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
      cv2_imshow(annotated_image)
      
#10
from IPython.display import display, Javascript
from google.colab.output import eval_js
from base64 import b64decode

def take_photo(filename='photo.jpg', quality=0.8):
  js = Javascript('''
    async function takePhoto(quality) {
      const div = document.createElement('div');
      const capture = document.createElement('button');
      capture.textContent = 'Capture';
      div.appendChild(capture);

      const video = document.createElement('video');
      video.style.display = 'block';
      const stream = await navigator.mediaDevices.getUserMedia({video: true});

      document.body.appendChild(div);
      div.appendChild(video);
      video.srcObject = stream;
      await video.play();

      // Resize the output to fit the video element.
      google.colab.output.setIframeHeight(document.documentElement.scrollHeight, true);

      // Wait for Capture to be clicked.
      await new Promise((resolve) => capture.onclick = resolve);

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      canvas.getContext('2d').drawImage(video, 0, 0);
      stream.getVideoTracks()[0].stop();
      div.remove();
      return canvas.toDataURL('image/jpeg', quality);
    }
    ''')
  display(js)
  data = eval_js('takePhoto({})'.format(quality))
  binary = b64decode(data.split(',')[1])
  with open(filename, 'wb') as f:
    f.write(binary)
  return filename


#11
from IPython.display import Image
try:
  filename = take_photo()
  print('Saved to {}'.format(filename))
  
  # Show the image which was just taken.
  display(Image(filename))
except Exception as err:
  # Errors will be thrown if the user does not have a webcam or if they do not
  # grant the page permission to access it.
  print(str(err))
  
#12
from PIL import Image

mpHands = mp.solutions.hands
hands = mpHands.Hands()

while True:
  filename = take_photo()
  img = Image.open(filename)

  img.save("/content/gdrive/My Drive/final_hw/ex12.jpeg")
  src = cv2.imread("/content/gdrive/My Drive/final_hw/ex12.jpeg", cv2.IMREAD_COLOR)

  uploaded = files.upload()

  images = {name: cv2.imread(name) for name in uploaded.keys()}

  with mp_face_detection.FaceDetection(
    min_detection_confidence=0.5) as face_detection:
    for name, image in images.items():
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # Draw face detections of each face.
    print(f'Face detections of {name}:')
    if not results.detections:
      continue
    annotated_image = image.copy()
    for detection in results.detections:
      # print('Nose tip:')
      # print(mp.python.solutions.face_detection.get_key_point(
      #     detection, mp_face_detection.FaceKeyPoints.NOSE_TIP))
      mp_drawing.draw_detection(annotated_image, detection)
      cv2_imshow(annotated_image)
      annotated_image.save("/content/gdrive/My Drive/final_hw/out11.jpeg")

#13
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

with mp_pose.Pose(
      min_detection_confidence=0.5,
      min_tracking_confidence=0.5
    ) as pose:
    pose_src = cv2.imread("/content/gdrive/My Drive/final_hw/ex12.jpeg", cv2.IMREAD_COLOR)
    cnt =10
    while (cnt>0) :
        cnt-=1
        image = cv2.cvtColor(pose_src, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
