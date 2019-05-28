import pymysql as mariadb
from google.cloud import storage
from PyQt4.QtGui import *
import os
import cv2

def get_filename_suffix(index, num_digits):
    digits = len(str(index))
    remaining_zeros = num_digits - digits
    zeros_str = ''
    for i in range(remaining_zeros):
        zeros_str += '0'

    return zeros_str + str(index)

def get_center(bbox):

    x = round((bbox[0] + bbox[2])/2.0)
    y = round((bbox[1] + bbox[3])/2.0)

    return [x, y]

def enforce_format(bbox):
    bbox[0], bbox[2] = min([bbox[0], bbox[2]]), max([bbox[0], bbox[2]])
    bbox[1], bbox[3] = min([bbox[1], bbox[3]]), max([bbox[1], bbox[3]])

    return bbox

def square_bbox(bbox, image):

    ih, iw = image.shape[::-1]

    center = get_center(bbox)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]

    s = max([w, h])

    if center[0] - s/2.0 < 0:
        center[0] = center[0] - (center[0] - s/2.0)
    elif center[0] + s/2.0 >= iw:
        center[0] = center[0] - (center[0] + s/2.0 - iw)
    if center[1] - s/2.0 < 0:
        center[1] = center[1] - (center[1] - s/2.0)
    elif center[1] + s/2.0 >= ih:
        center[1] = center[1] - (center[1] + s/2.0 - ih)

    top_left = [round(center[0] - s/2.0), round(center[1] - s/2.0)]
    bottom_right = [round(center[0] + s/2.0), round(center[1] + s/2.0)]

    return map(int, top_left + bottom_right)

img_filePath = "GCS_pulls/"
crop_dir = "crops/"
db_host = "35.231.251.154"
db_database = "wasabi_labels"
db_username = "labeler"
db_password = "mlsandbox-1-191918"
image_format = ".tif"

#(track_id, `row`, col, frame, bb_tl_x, bb_tl_y, bb_br_x, bb_br_y, mount, `comment`, labeler)


mariadb_connection = mariadb.connect(host=db_host, user=db_username, password=db_password, database=db_database)
cursor = mariadb_connection.cursor()

track_ids = []

sql = "SELECT track_id FROM track_labels"
cursor.execute(sql)
for track_id in cursor:
    track_ids.append(track_id[0])

track_ids_c = []
for track_id in track_ids:
    if track_id not in track_ids_c:
        track_ids_c.append(track_id)

track_ids = track_ids_c

print(track_ids)
for track_id in track_ids:
    sql = "SELECT row, col, frame, bb_tl_x, bb_tl_y, bb_br_x, bb_br_y FROM track_labels WHERE `track_id` = {}".format(track_id)
    cursor.execute(sql)

    for i, [row, col, frame, bb_tl_x, bb_tl_y, bb_br_x, bb_br_y] in enumerate(cursor):
        row_col_frame = get_filename_suffix(row, 2) + '_' + get_filename_suffix(col, 2) + '/' + get_filename_suffix(frame, 4) + image_format
        row_col_frame_name = row_col_frame.replace('/', '_')

        if row_col_frame_name not in os.listdir(img_filePath):
            src = "imgs_to_label/" + row_col_frame
            storage_client = storage.Client()
            bucket = storage_client.get_bucket("nga-wasabi")
            blob = bucket.blob(src)
            print "Downloading " + src + " to " + img_filePath + "..."
            blob.download_to_filename(img_filePath + row_col_frame_name)

        crop_name = row_col_frame_name.split('.')[0] + '_' + str(track_id) + image_format
        if str(track_id) not in os.listdir(crop_dir):
            os.mkdir(crop_dir + str(track_id))
        if crop_name not in os.listdir(crop_dir + str(track_id)):
            image = cv2.imread(img_filePath + row_col_frame_name, 0)
            try:
                image.shape[::-1]
            except:
                print("Bad Image: " + img_filePath + row_col_frame_name)
                continue
            bbox = enforce_format([bb_tl_x, bb_tl_y, bb_br_x, bb_br_y])
            bbox = square_bbox(bbox, image)
            crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(crop_dir + str(track_id) + '/' + crop_name, crop)