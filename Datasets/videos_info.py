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
    lan = str(args.sign_lang)
    root_path = Path(args.save_path + lan)
    videos_path = Path(args.save_path + lan + f"/{lan}_videos")

    if Path(f"{root_path}/{lan}_videospath.txt").exists():
        print(f"{lan}_videospath.txt exists.")
        files_grabbed = []
        with open(f"{root_path}/{lan}_videospath.txt", 'r') as filehandle:
            for line in filehandle:
                # Remove linebreak which is the last character of the string
                curr_place = line[:-1]
                # Add item to the list
                files_grabbed.append(curr_place)
    else:
        files_grabbed = sorted(videos_path.glob('*.mp4'), key=os.path.getmtime)
        with open(f"{root_path}/{lan}_videospath.txt", 'w') as filehandle:
            for listitem in files_grabbed:
                filehandle.write(f'{listitem}\n')


    vidnum = 0    
    verses_list=[]
    verse_i = 0
    cum_dur = 0

    for videopath in files_grabbed:
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
        
        video = cv2.VideoCapture(str(videopath))
        for index, row in df.iterrows():           
            verse_dict = {}
            verse_i = verse_i + 1  
            verse_dict["verse_num"] = verse_i
            verse_dict["video_num"] = vidnum              
            verse_dict["video_name"] = refB
            verse_dict["verse_name"] = row["title"]            
            verse_dict["duration"] = float(float(row["end_time"]) - float(row["start_time"]))
            verse_dict["cum_duration"] = cum_dur + (verse_dict["duration"]/3600.0)
            cum_dur = verse_dict["cum_duration"]
            verses_list.append(verse_dict)
                    
        vidnum += 1 
        print(f"Video {vidnum} - {refB} done.")         
        video.release()
        cv2.destroyAllWindows() 
    file = gzip.GzipFile(f"/content/drive/MyDrive/Sign_Language_Videos/dataset/T{lan}240.dataset", 'wb')
    file.write(pickle.dumps(verses_list,0))
    file.close()
    print(cum_dur, len(verses_list))
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--save_path", help="Path to save the video", default="./videos", type=str
    )
    parse.add_argument(
        "--sign_lang", help="Sign Language to work on", default="gse", type=str
    )
    args = parse.parse_args()
    logging.info("Start converting dataset")
    main(args)
    logging.info("Converting dataset finished")
