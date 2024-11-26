import os.path
from pathlib import Path
from argparse import ArgumentParser
from tqdm import tqdm

from ohw.dataset import GPSDataset
from ohw.utils import geotag_image

def parse_args():
    parser = ArgumentParser(description="Script for geotagging images.")
    parser.add_argument("input", type=str, help="Location of the image input folder.")
    parser.add_argument("geotag_csv", type=str, help="Location of the csv to be used for geotagging.")

    parser.add_argument("--kml", action="store_true", help="Whether to write .KML files as outputs.")
    return parser.parse_args()

def geotag_images(input_dir : str, geotag_csv : str = None, kml : bool = False):
    site_dir = os.path.abspath(input_dir)
    gps_ds = GPSDataset(root=site_dir, csv_file=geotag_csv, img_dir="backup")

    # if output is given, write a sister directory
    site_path = Path(site_dir)    
    if kml:
        kml_path = site_path / "kml"
        kml_path.mkdir(exist_ok=True, parents=True)
        print("Created directory {} for writing kml files".format(kml_path))

    for imgf, geodata, img_id in tqdm(gps_ds, leave=True):
        geotag_image(imgf, geodata)
        

if __name__=="__main__":
    args = parse_args()
    geotag_images(args.input, args.geotag_csv, args.output, args.kml)