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
from pytesseract import Output
from skimage.filters import threshold_local

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

## Returns a bounding box and probability score if it is more than minimum confidence
def predictions(prob_score, geo, min_confidence = 0.5):
	(numR, numC) = prob_score.shape[2:4]
	boxes = []
	confidence_val = []

	# loop over rows
	for y in range(0, numR):
		scoresData = prob_score[0, 0, y]
		x0 = geo[0, 0, y]
		x1 = geo[0, 1, y]
		x2 = geo[0, 2, y]
		x3 = geo[0, 3, y]
		anglesData = geo[0, 4, y]

		# loop over the number of columns
		for i in range(0, numC):
			if scoresData[i] < min_confidence:
				continue

			(offX, offY) = (i * 4.0, y * 4.0)

			# extracting the rotation angle for the prediction and computing the sine and cosine
			angle = anglesData[i]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# using the geo volume to get the dimensions of the bounding box
			h = x0[i] + x2[i]
			w = x1[i] + x3[i]

			# compute start and end for the text pred bbox
			endX = int(offX + (cos * x1[i]) + (sin * x2[i]))
			endY = int(offY - (sin * x1[i]) + (cos * x2[i]))
			startX = int(endX - w)
			startY = int(endY - h)

			boxes.append((startX, startY, endX, endY))
			confidence_val.append(scoresData[i])

	# return bounding boxes and associated confidence_val
	return (boxes, confidence_val)

def ocr_processing(image):
    orig = image.copy()
    orig_height, orig_width = image.shape[:2]
    new_width, new_height = 320, 320
    ratio_width = orig_width / float(new_width)
    ratio_height = orig_height / float(new_height)
    image = cv2.resize(image, (new_width, new_height))
    (height, width) = image.shape[:2]
    # construct a blob from the image to forward pass it to EAST model
    blob = cv2.dnn.blobFromImage(image, 1.0, (width, height),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
    
    # load the pre-trained EAST model for text detection 
    net = cv2.dnn.readNet('./frozen_east_text_detection.pb')

    # We would like to get two outputs from the EAST model. 
    #1. Probabilty scores for the region whether that contains text or not. 
    #2. Geometry of the text -- Coordinates of the bounding box detecting a text

    # The following two layer need to pulled from EAST model for achieving this. 
    layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

    #Forward pass the blob from the image to get the desired output layers
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # Find predictions and  apply non-maxima suppression
    (boxes, confidence_val) = predictions(scores, geometry)
    boxes = non_max_suppression(np.array(boxes), probs=confidence_val)

    print(boxes)

    # Text Detection and Recognition 
    results = []

    # loop over the bounding boxes to find the coordinate of bounding boxes
    for (x_start, y_start, x_end, y_end) in boxes:

        # scale the coordinates based on the respective ratios in order to reflect bounding box on the original image
        x_start = int(x_start * ratio_width)
        y_start = int(y_start * ratio_height)
        x_end = int(x_end * ratio_width)
        y_end = int(y_end * ratio_height)

        # print(x_start, y_start, x_end, y_end)


        #extract the region of interest
        r = orig[y_start:y_end, x_start:x_end]

        #configuration setting to convert image to string.  
        configuration = ("-l eng --oem 1 --psm 8")
    
        ##This will recognize the text from the image of bounding box
        text = pytesseract.image_to_string(r, config=configuration)

        # append bbox coordinate and associated text to the list of results 
        results.append(((x_start, y_start, x_end, y_end), text))

    #Display the image with bounding box and recognized text
    orig_image = orig.copy()

    # Moving over the results and display on the image
    for ((x_start, y_start, x_end, y_end), text) in results:
        # display the text detected by Tesseract
        print("{}\n".format(text))

        # Displaying text
        text = "".join([x if ord(x) < 128 else "" for x in text]).strip()
        cv2.rectangle(orig_image, (x_start, y_start), (x_end, y_end),
            (0, 0, 255), 2)
        cv2.putText(orig_image, text, (x_start, y_start - 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0, 255), 2)

    plt.imshow(orig_image)
    plt.title('Output')
    plt.show()

def plot_rgb(image):
    plt.figure(figsize=(16,10))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    return 

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
            # ocr_processing(img_BGR)

            d = pytesseract.image_to_data(img_BGR, output_type=Output.DICT)
            n_boxes = len(d['level'])
            boxes = cv2.cvtColor(img_BGR.copy(), cv2.COLOR_BGR2RGB)
            for i in range(n_boxes):
                (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])    
                boxes = cv2.rectangle(boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # convert image to grey scale
            # image_grey = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
            # thresh = threshold_local(image_grey, 21, offset = 5, method = "gaussian")
            # image_threshold = (image_grey > thresh).astype("uint8") * 255

            extracted_text = pytesseract.image_to_string(img_BGR)

            print(extracted_text)
            # plot_rgb(boxes)

            cv2.namedWindow(winname)
            cv2.moveWindow(winname, 40, 30)
            cv2.setWindowProperty(winname, cv2.WND_PROP_TOPMOST, 1)
            cv2.imshow(winname, boxes)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('0'):
                print('Stopped.')
                return 
            cv2.destroyAllWindows()


if __name__ == '__main__':
    main()