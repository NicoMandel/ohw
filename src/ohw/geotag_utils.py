import os.path
import numpy as np
import pandas as pd
import subprocess
import simplekml
from PIL import Image

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
    if compress:
        img_f = _compress_img(img_f, save_dir, img_id)
    photo.icon.href = "{}".format(img_f)
    photo.viewvolume = simplekml.ViewVolume(-25,25,-15,15,1)
    kml_path = os.path.join(save_dir, "{}.kmz".format(img_id))
    kml.savekmz(kml_path)
    # remove file after compressing into kmz
    if compress:
        os.remove(img_f)
        # pass