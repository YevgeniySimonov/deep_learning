from json import load
import numpy as np
import cv2
from imutils.object_detection import non_max_suppression
import pytesseract
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor
import os
import traceback as tb
import PIL
import scipy.ndimage as ndimage
from urllib.request import urlopen

load_dotenv()

server = os.environ.get('SERVER', None)
username = os.environ.get('USERNAME', None)
password = os.environ.get('PASSWORD', None)
database = os.environ.get('DATABASE', None)
port = os.environ.get('PORT', None)

working_dir = os.path.dirname(os.path.realpath(__file__))


def get_db_connection():
    conn = psycopg2.connect(
        user=username, 
        password=password, 
        host=server, 
        port=port, 
        database=database)
    return conn

def get_data_from_db():
    data_arr = []
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        with open('query.sql', 'r') as sql_file:
            query = sql_file.read()
        cur.execute(query)
        db_data = cur.fetchall()
        for data in db_data: 
            data_arr.append(dict(data))
        cur.close()
    except (Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
    return data_arr

def main():
    data_arr = get_data_from_db()
    for data in data_arr:
        winname = 'receipt '+ str(data['receiptId'])
        for img_url in data['content']:
            pil_img = PIL.Image.open(urlopen(img_url))
            pil_img_rot = pil_img.rotate(-90, expand=True)
            img_RGB = np.array(pil_img_rot)
            img_BGR = cv2.cvtColor(img_RGB, cv2.COLOR_RGB2BGR)
            if img_BGR.shape[0] < img_BGR.shape[1]: # check if image is horizontal
                img_BGR = ndimage.rotate(img_BGR, 90.0, reshape=True)
            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 40, 30)
            cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow(winname, img_BGR)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('0'):
                print('Stopped.')
                return 
            cv2.destroyAllWindows()

if __name__ == '__main__':
    main()