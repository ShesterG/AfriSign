import urllib.request
import json
import logging
import argparse
from pathlib import Path
import glob
import itertools
import pickle
import gzip

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def main(args):
    #lan = "ase"
    r_path = Path(args.save_path)

    
    filepath = "/content/gbucketafrisign/all_links_720_29.list"
    with gzip.open(filepath, "rb") as f:
      data = pickle.load(f)
    
    #all_links_720 = load_dataset_file("/content/drive/MyDrive/Research/JWSLT/Variables/all720.list")
    
    i=0
    j=0
    k=0
    for obj in data:
        video_url = obj["videoUrl"]
        video_name = obj["video_name"]
        lan = obj["slang"]
        if video_url is None:
            continue
        
        file_path = Path(args.save_path + f"/{lan}/{video_name}.mp4")
        if file_path.exists():
            logging.info(f"{file_path} already exists")
            continue
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
          logging.info(f"Downloading {file_path}")
          logging.info(f"Requesting {video_url}")
          urllib.request.urlretrieve(video_url, file_path)
          i = i + 1  
        except ValueError:
          print(f"SKIPPED SKIPPED SKIPPED SKIPPED SKIPPED - {video_name}")
          j = j + 1
        
        k = k + 1 
    logging.info(f"Downloading all set finished, {k} videos. {i} completed and {j} were skipped.")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--save_path", help="Path to save the video",default="./videos",type=str)
    args = parse.parse_args()
    logging.info("Start downloading dataset")
    main(args)
    logging.info("Downloading dataset finished")
