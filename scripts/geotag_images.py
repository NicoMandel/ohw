import os.path
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from ohw.dataset import GPSDataset
from ohw.geotag_utils import geotag_image, create_kml

def parse_args():
    parser = ArgumentParser(description="Script for geotagging images.")
    parser.add_argument("input", type=str, help="Location of the image input folder.")
    parser.add_argument("geotag_csv", type=str, help="Location of the csv to be used for geotagging.")

    parser.add_argument("--kml", action="store_true", help="Whether to write .KML files as outputs.")
    parser.add_argument("--compress", action="store_true", help="Whether to compress the image when storing as kml to 0.1 resolution")
    return parser.parse_args()

def geotag_images(input_dir : str, geotag_csv : str = None, kml : bool = False, compress : bool = True):
    site_dir = os.path.abspath(input_dir)
    gps_ds = GPSDataset(root=site_dir, csv_file=geotag_csv, img_dir="backup")

    for imgf, geodata, img_id in tqdm(gps_ds, leave=True):
        geotag_image(imgf, geodata)
        if kml:
            create_kml(imgf, geodata, gps_ds.imgdir , img_id, compress)
        

if __name__=="__main__":
    args = parse_args()
    geotag_images(args.input, args.geotag_csv, args.kml, args.compress)