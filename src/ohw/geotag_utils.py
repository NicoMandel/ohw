import os.path
import numpy as np
import pandas as pd
import subprocess
import simplekml
from PIL import Image
import cv2

from ohw.utils import load_image, save_image
import matplotlib.pyplot as plt

def _compress_img(img_f : str, save_dir, img_id, max_size : int = 512) -> str:
    """
        storing a thumbnail of an image.
        Returns the str where it is returned
    """
    img = Image.open(img_f).copy()
    img.thumbnail((max_size, max_size))
    thumb_path = os.path.join(save_dir, "{}_thumb.png".format(img_id))
    img.save(thumb_path)
    return thumb_path

# Geotagging images
def geotag_image(img_path : np.ndarray, geotag : pd.Series) :
    """
        Function to geotag an image with the exif tool   
    """
    # metad = _format_gps_metadata(geotag)
    lat = geotag['latitude [decimal degrees]']
    long = geotag['longitude [decimal degrees]']
    command=[
        "exiftool",
        f"-GPSLatitude={lat}",
        f"-GPSLatitudeRef={'N' if lat >= 0 else 'S'}",
        f"-GPSLongitude={long}",
        f"-GPSLongitudeRef={'E' if long >= 0 else 'W'}",
        "-overwrite_original",
        img_path
    ]
    subprocess.run(command)

# creating kmls
def create_kml(img_f : str, geodata : pd.Series, save_dir: str, img_id : str, compress : bool = True):
    """
        Function to create a kml of an image with the geolocation and embed the information
    """
    kml = simplekml.Kml()
    photo = kml.newphotooverlay(name=img_id)
    photo.camera = simplekml.Camera(longitude = geodata['longitude [decimal degrees]'], latitude=geodata['latitude [decimal degrees]'], altitude=geodata["altitude [meter]"],
                                    altitudemode=simplekml.AltitudeMode.clamptoground)
    photo.point.coords= [(geodata["longitude [decimal degrees]"], geodata["latitude [decimal degrees]"])]
    photo.viewvolume = simplekml.ViewVolume(-25,25,-15,15,1)
    # save normal kml
    kml_path = os.path.join(save_dir, "{}.kml".format(img_id))
    kml.save(kml_path)
    # add image to save kmz too
    if compress:
        img_f = _compress_img(img_f, save_dir, img_id)
    photo.icon.href = "{}".format(img_f)
    kmz_path = os.path.join(save_dir, "{}.kmz".format(img_id))
    kml.savekmz(kmz_path)
    
    # remove file after compressing into kmz
    if compress:
        os.remove(img_f)
        # pass

def draw_north_arrow(imgf : str, geodata : pd.Series, save_dir: str, img_id : str, color : tuple = (0, 0, 0), lw : int = 20) -> None:
    """
        Method to draw a north arrow on the images
    """
    yaw = geodata['yaw [degrees]']
    img = load_image(imgf)
    h, w = img.shape[:2]

    arr_len = min(h,w) // 8     # 1/8th length
    # TODO adopt this placement calculation
    center_x, center_y = w - (100 + arr_len), (100 + arr_len)       # top right corner

    # arrow tip based on yaw
    ang_rad = np.deg2rad(yaw)
    end_x = int(center_x + arr_len * np.sin(ang_rad))
    end_y = int(center_y + arr_len * np.cos(ang_rad))
    cv2.arrowedLine(img, (center_x, center_y), (end_x, end_y), color, lw, tipLength=0.3)
    imgp = os.path.join(save_dir, img_id + "_N.jpg")
    save_image(img, imgp)
