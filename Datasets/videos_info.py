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
import tensorflow as tf
import torch
import json
from PIL import Image
import mediapipe as mp
import time


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main(args):
    #lan = str(args.sign_lang)
    #root_path = Path(args.save_path + lan)
    root_path = Path(args.save_path)
    
    filepath = "/home/s_gueuwou/gbucketafrisign/all_links_720_29.list"
    with gzip.open(filepath, "rb") as f:
      data = pickle.load(f)    
    verses_list = []
    verses_error = []
    cum_dur = 0
    vidnum = 1
    verse_i = 0
    for obj in data:
      #video_url = obj["videoUrl"]
      video_name = obj["video_name"]
      lan = obj["slang"]
      if video_name is None:
        continue
        
      video_path = Path(f"/home/s_gueuwou/gbucketafrisign/videos/{lan}/{video_name}.mp4")      
      json_path = Path(f"/home/s_gueuwou/gbucketafrisign/vinfo/{lan}/{video_name}.json")      
      refB = video_name

      if Path(json_path).exists():
        pass
      else:  
        json_path.parent.mkdir(parents=True, exist_ok=True)
        os.system(f"ffprobe -i {video_path} -print_format default -show_chapters -loglevel error > {root_path}/{lan}/{refB}.json 2>&1")
      

      try:
        with open(f"{root_path}/{lan}/{refB}.json", "r") as infile:
            print(str(video_path))
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
        
        video = cv2.VideoCapture(str(video_path))
        for index, row in df.iterrows():           
            verse_dict = {}
            verse_i = verse_i + 1  
            verse_dict["verse_num"] = verse_i
            verse_dict["video_num"] = vidnum              
            verse_dict["video_name"] = refB
            verse_dict["verse_lang"] = lan
            verse_dict["verse_name"] = row["title"]  
            verse_dict["verse_start"] = float(row["start_time"]) 
            verse_dict["verse_end"] = float(row["end_time"])          
            verse_dict["duration"] = float(float(row["end_time"]) - float(row["start_time"]))
            verse_dict["cum_duration"] = cum_dur + (verse_dict["duration"]/3600.0)
            #cum_dur = verse_dict["cum_duration"]
            verses_list.append(verse_dict)
      except:
        verses_error.append(obj)


      vidnum += 1 
      print(f"Video {vidnum} - {refB} done.")         
      video.release()
      cv2.destroyAllWindows() 
    filep = gzip.GzipFile(f"/home/s_gueuwou/gbucketafrisign/vinfo720.dict", 'wb')
    fileq = gzip.GzipFile(f"/home/s_gueuwou/gbucketafrisign/verses_e.dict", 'wb')
    filep.write(pickle.dumps(verses_list,0))
    fileq.write(pickle.dumps(verses_error,0))
    filep.close()
    fileq.close()
    print(cum_dur, len(verses_list))
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--save_path", help="Path to save the videos info json files", default="./videos", type=str
    )
    args = parse.parse_args()
    logging.info("Start converting dataset")
    main(args)
    logging.info("Converting dataset finished")
