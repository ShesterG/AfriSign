#videos to tensor no splitting 

#frames from videos; https://github.com/Ai-Genome-Saksham/OpenCV/blob/main/OpenCV/%239%20Extracting%20Images%20from%20Video.py
#text from images;   https://github.com/bhadreshpsavani/ExploringOCR/blob/master/OCRusingTesseract.ipynb

#!sudo apt install tesseract-ocr # TODO : INSERT INSIDE REQUIREMENTS ?
#!pip install pytesseract        # TODO : INSERT INSIDE REQUIREMENTS ?

#import pytesseract
import random
import logging
import argparse
from pathlib import Path
import os
import cv2
import re
import shutil
import glob
import io
import pandas as pd
import numpy as np
import math
import sys
import pickle
import gzip
#from PIL import Image
#from PIL import ImageChops
#from PIL import ImageEnhance
import tensorflow as tf
import torch
import json
from PIL import Image
import mediapipe as mp
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main(args):
    lan = "ase" 
    folder_name = "3"


    root_path = Path(args.save_path)

    if Path(f"{root_path}/{lan}_videospath.txt").exists():
        print(f"{lan}_videospath.txt exists.")
        files_grabbed = []
        with open(f"{root_path}/{lan}_videospath.txt", 'r') as filehandle:
            for line in filehandle:
                # Remove linebreak which is the last character of the string
                curr_place = line[:-1]
                # Add item to the list
                files_grabbed.append(curr_place)

    vidnum = 0    
    verses_list=[]
    verse_i = 0
    
    mp_holistic = mp.solutions.holistic # Holistic model
    mp_drawing = mp.solutions.drawing_utils # Drawing utilities
    mp_pose = mp.solutions.pose
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:        
      for videopath in files_grabbed[112:156]: #edit the range here
          #vid = cv2.VideoCapture(str(video_path))
          refB = str(videopath).split('/')[-1].split('.')[0]
          os.system(f"ffprobe -i {videopath} -print_format default -show_chapters -loglevel error > {root_path}/{refB}.json 2>&1")
          
          with open(f"{root_path}/{refB}.json", "r") as infile:
              print(str(videopath))
              data = infile.read()

              if str(data)[:9]=="[CHAPTER]": 
                pass 
              else:  
                data = "[CHAPTER]" + str(data).split("[CHAPTER]",1)[1]
          
          data = data.replace("\n", "|")
          data = data.replace("|[/CHAPTER]|", "\n")
          colnames=["CHAPTER","id","time_base","start","start_time","end","end_time","title"]
          dataIO = io.StringIO(data)
          df = pd.read_csv(dataIO, sep="|", names=colnames, header=None)
          df.drop(['CHAPTER', 'id', 'time_base', 'start', 'end'], axis=1, inplace=True)
          df['LastDigit'] = [x.strip()[-1] for x in df['title']]
          df = df[df['LastDigit'].str.isdigit()]
          df.drop(['LastDigit'], axis=1, inplace=True)
          df["start_time"] = df["start_time"].str.replace("start_time=", "")
          df["end_time"] = df["end_time"].str.replace("end_time=", "")
          df["title"] = df["title"].str.replace("TAG:title=", "")
          #df['title'] = df['title'].str.replace('.', '_') 
          #df['title'] = df['title'].str.replace(':', '_')
          #df['title'] = df['title'].str.replace(' ', '_')        
          
          video = cv2.VideoCapture(str(videopath))
          for index, row in df.iterrows():           
              #opencv_method
              verse_dict = {}              
              verse_dict["name"] = lan + row["title"].strip().replace(" ", "_")
              verse_torchpath = "/content/drive/MyDrive/Sign_Language_Videos/dataset/ALL240/verses_tensors/" + verse_dict["name"] + ".pt" 
              verse_dict["lang"] = lan
              verse_dict["signer"] = "SignerX"            
              verse_dict["duration"] = float(float(row["end_time"]) - float(row["start_time"]))
              verse_dict["gloss"] = "GLOSS GLOSS GLOSS'"
              verse_dict["text"] = "Text"
              if Path(verse_torchpath).exists():
                verse_dict["sign"] = verse_torchpath #torch.load(verse_torchpath)
                verses_list.append(verse_dict)
                verse_i += 1
                print(f"Verse {verse_i} - {verse_dict['name']} done.")
                continue
              ##currentframe = 1
              step = 1/25
              body_feature_list=[]
              mean_distance = 0
              n = 0
              for current_second in np.arange(math.ceil(float(row["start_time"])), math.floor(float(row["end_time"])), step):
                t_msec = 1000*(current_second)
                video.set(cv2.CAP_PROP_POS_MSEC, t_msec)
                success, frame = video.read()
                if success:
                  frame = cv2.resize(frame,(320, 240))
                  frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
                  frame.flags.writeable = False                  # Image is no longer writeable
                  results = holistic.process(frame)              # Make prediction                  
                  pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
                  face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
                  lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
                  rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
                  

                  # a variant of your normalise method here   
                  left_shoulder = np.array([pose[44],pose[45],pose[46]])
                  right_shoulder = np.array([pose[48],pose[49],pose[50]])
                  shoulder_width = (((right_shoulder - left_shoulder)**2).sum())**0.5
                  mean_distance = mean_distance + shoulder_width
                  n = n + 1
                  center_pose = np.array([(left_shoulder[0]+right_shoulder[0])/2,
                                    (left_shoulder[1]+right_shoulder[1])/2,
                                      0,0])
                  center_pose = np.tile(center_pose,33)
                  pose = pose - center_pose 
                  center = np.array([(left_shoulder[0]+right_shoulder[0])/2,
                                    (left_shoulder[1]+right_shoulder[1])/2,
                                      0])
                  center_face = np.tile(center,468)
                  face = face - center_face

                  center_hands = np.tile(center,21)
                  lh = lh - center_hands
                  rh = rh - center_hands
                  
                  image_feature = np.concatenate([pose, face, lh, rh])                  
                  image_feature = torch.from_numpy(image_feature)
                  image_feature = image_feature.to(dtype=torch.float32)
                  image_feature = torch.flatten(image_feature)
                  image_feature = torch.round(image_feature * 10**4) / (10**4)
                  body_feature_list.append(image_feature)               

              try: 
                mean_distance = mean_distance / n
              except ZeroDivisionError:
                print(f"SKIPPED - {verse_dict['name']} duration {verse_dict['duration']}")
                continue  

              try:
                verse_torch = torch.stack(body_feature_list,0)
                verse_torch = torch.mul(verse_torch, (1/mean_distance))
                torch.save(verse_torch, verse_torchpath)
                verse_dict["sign"] = verse_torchpath #torch.load(verse_torchpath)
                verses_list.append(verse_dict)
              except RuntimeError:
                print(f"ERROR: Video {verse_i} - {refB}.")   
              verse_i += 1
              print(f"Verse {verse_i} - {verse_dict['name']} done.")
              
          video.release()
          cv2.destroyAllWindows() 
    file = gzip.GzipFile(f"/content/drive/MyDrive/Sign_Language_Videos/dataset/{lan}{folder_name}.dataset", 'wb')
    file.write(pickle.dumps(verses_list,0))
    file.close()
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--save_path", help="Path to save the video", default="./videos", type=str
    )
    args = parse.parse_args()
    logging.info("Start converting dataset")
    main(args)
    logging.info("Converting dataset finished")
