import shutil
import datetime as dt
import os
import cv2
import glob
from config import BB_TOP_LEFT_X, BB_TOP_LEFT_Y, BB_BOTTOM_RIGHT_X, BB_BOTTOM_RIGHT_Y

# Define bounding box
bb_tl_x = BB_TOP_LEFT_X
bb_tl_y = BB_TOP_LEFT_Y
bb_br_x = BB_BOTTOM_RIGHT_X
bb_br_y = BB_BOTTOM_RIGHT_Y

def get_center(bbox):

    x = round((bbox[0] + bbox[2])/2.0)
    y = round((bbox[1] + bbox[3])/2.0)

    return [x, y]

def enforce_format(bbox):
    bbox[0], bbox[2] = min([bbox[0], bbox[2]]), max([bbox[0], bbox[2]])
    bbox[1], bbox[3] = min([bbox[1], bbox[3]]), max([bbox[1], bbox[3]])

    return bbox

def square_bbox(bbox, image):

    #ih, iw = image.shape[::-1]
    ih, iw, _ = image.shape

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

    tl_x = round(center[0] - s/2.0)
    tl_y = round(center[1] - s/2.0)
    
    br_x = round(center[0] + s/2.0)
    br_y = round(center[1] + s/2.0)

    #return map(int, top_left + bottom_right)
    return [tl_x, tl_y, br_x, br_y]


def crop_MSTAR_7(full_pathname_mstar7_dataset_dir, image_format='.png'):

    day_timestamp = dt.datetime.now().strftime("%y-%m-%d-%H-%M")
    img_filePath = full_pathname_mstar7_dataset_dir
    crop_dir = img_filePath + 'crops/'
    crop_dir_old = img_filePath + 'crops_' + day_timestamp + '/'
    if os.path.exists(crop_dir):
        #shutil.rmtree(crop_dir)
        os.rename(crop_dir, crop_dir_old)
    os.mkdir(crop_dir)
    image_format = image_format

    mstar7_imgs = glob.glob(img_filePath + '*' + image_format)

    for fname in mstar7_imgs:
        crop_name = os.path.basename(fname).split('.')[0] + '_cropped_to_24' + image_format

        if crop_name not in os.listdir(crop_dir):
            image = cv2.imread(fname)
            try:
                image.shape[::-1]
            except:
                print("Bad Image: " + img_filePath + fname)
                continue
            bbox = enforce_format([bb_tl_x, bb_tl_y, bb_br_x, bb_br_y])
            bbox = square_bbox(bbox, image)
            crop = image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            cv2.imwrite(crop_dir + '/' + crop_name, crop)

def create_cropped_dir(full_pathname_mstar7_dataset_dir):
    # Now move all cropped images to new MSTAR7_cropped directory
    mstar7_dataset = os.path.basename(full_pathname_mstar7_dataset_dir.rstrip('/'))
    mstar7_dirname = os.path.dirname(full_pathname_mstar7_dataset_dir).rstrip(mstar7_dataset)
    cropped_mstar7_dataset_dir = mstar7_dirname + mstar7_dataset + '_cropped/'
    mstar7_folders = os.listdir(full_pathname_mstar7_dataset_dir)

    os.mkdir(cropped_mstar7_dataset_dir)
    for folder in mstar7_folders:
        cropped_folder = cropped_mstar7_dataset_dir + folder + '/'
        os.mkdir(cropped_folder)
        orig_cropped_dataset_dir = full_pathname_mstar7_dataset_dir + '/' + folder + '/crops/'
        for root, dirs, files in os.walk(orig_cropped_dataset_dir):
            for f in files:
                cropped_file = cropped_folder + f
                shutil.copyfile(os.path.join(root, f), cropped_file)
