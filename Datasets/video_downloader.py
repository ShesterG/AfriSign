import urllib.request
import json
import logging
import argparse
from pathlib import Path
import glob
import itertools

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def main(args):
    lan = "ase"
    r_path = Path(args.save_path)

    if Path(f"{r_path}/{lan}240.json").exists():
      print(f"{lan}240.json exists.")
    else:  
      combined = []
      for json_file in r_path.glob("*.json"): #Assuming that your files are json files
          with open(json_file, "rb") as infile:
              combined.append(json.load(infile))
      combined_list = list(itertools.chain.from_iterable(combined))
      with open(f"{r_path}/{lan}240.json", "w") as f:
          json.dump(combined_list, f)
    
    
    logging.info(f"Downloading {lan} set")
    with open(f"{r_path}/{lan}240.json", "r") as f:
        data = json.load(f)
    i=0
    j=0
    k=0
    for obj in data:
        video_url = obj["videoUrl"]
        video_name = video_url.split('/')[-1].split('.')[0]
        if video_url is None:
            continue
        
        file_path = Path(args.save_path + f"/{lan}_videos/{video_name}.mp4")
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
    logging.info(f"Downloading {lan} set finished, {k} videos. {i} completed and {j} were skipped.")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--save_path", help="Path to save the video",default="./videos",type=str)
    args = parse.parse_args()
    logging.info("Start downloading dataset")
    main(args)
    logging.info("Downloading dataset finished")
