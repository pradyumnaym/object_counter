import cv2
import sys
import gc
import numpy as np
import imutils

# params
outlimit = 15
inlimit = 25
interval = 15
videopath = "campus_better.mp4"

# Reference link - https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/

classes_90 = ["background", "person", "bicycle", "car", "motorcycle",
              "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
              "unknown", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
              "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "unknown", "backpack",
              "umbrella", "unknown", "unknown", "handbag", "tie", "suitcase", "frisbee", "skis",
              "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
              "surfboard", "tennis racket", "bottle", "unknown", "wine glass", "cup", "fork", "knife",
              "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
              "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "unknown", "dining table",
              "unknown", "unknown", "toilet", "unknown", "tv", "laptop", "mouse", "remote", "keyboard",
              "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "unknown",
              "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]


def convertToDict(bboxes):
    dictlist = []
    for bbox in bboxes:
        d = dict()
        d['x1'] = bbox[0]
        d['y1'] = bbox[1]
        d['x2'] = bbox[2]
        d['y2'] = bbox[3]
        if not outOfFrame(d, frame, -inlimit):
            dictlist.append(d)
    return dictlist


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return convertToDict(boxes[pick].astype("int"))


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    if (bb1['x1'] > bb2['x1'] and bb1['x2'] < bb2['x2']) or (bb1['x1'] < bb2['x1'] and bb1['x2'] > bb2['x2']):
        return 1
    elif (bb1['y1'] > bb2['y1'] and bb1['y2'] < bb2['y2']) or (bb1['y1'] < bb2['y1'] and bb1['y2'] > bb2['y2']):
        return 1
    return iou


def getTracker(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'GOTURN':
        tracker = cv2.TrackerGOTURN_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    if tracker_type == "CSRT":
        tracker = cv2.TrackerCSRT_create()
    return tracker


def getDNNBoxes(tensorflowNet, frame):
    tensorflowNet.setInput(cv2.dnn.blobFromImage(frame, size=(300, 300), swapRB=True, crop=False))
    networkOutput = tensorflowNet.forward()
    rows, cols, channels = frame.shape
    bboxes = []
    # Loop on the outputs
    for detection in networkOutput[0, 0]:
        score = float(detection[2])
        if score > 0.35:
            detected = classes_90[int(detection[1])]
            if detected in ["bus", "train", "truck", "bicycle", "car", "motorcycle"]:
                newbox = dict()
                newbox['x1'] = detection[3] * cols
                newbox['y1'] = detection[4] * rows
                newbox['x2'] = detection[5] * cols
                newbox['y2'] = detection[6] * rows
                bboxes.append([newbox['x1'], newbox['y1'], newbox['x2'], newbox['y2']])
    return non_max_suppression_fast(np.asarray(bboxes), 0.5)


def getDict(bbox):
    # (x,y,w,h)
    d = dict()
    d['x1'] = bbox[0]
    d['x2'] = bbox[0] + bbox[2]
    d['y1'] = bbox[1]
    d['y2'] = bbox[1] + bbox[3]
    return d


def outOfFrame(bbox, frame, limit=0):
    h, w, _ = frame.shape
    if (bbox['x1'] < 0 - limit) or (bbox['y1'] < 0 - limit) or (bbox['x2'] > w + limit) or (bbox['y2'] > h + limit):
        return True
    return False


tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN', 'MOSSE', 'CSRT']
tracker_type = tracker_types[-1]
tensorflowNet = cv2.dnn.readNetFromTensorflow('frozen_inference_graph_frcnn.pb', 'pbpb_frcnn.pbtxt')

# Read video
def main():
    video = cv2.VideoCapture(videopath)

    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    # Read first frame.
    ok, frame = video.read()
    frame = cv2.resize(frame, (1280, 720))
    #frame = imutils.rotate_bound(frame, 180)

    if not ok:
        print('Cannot read video file')
        sys.exit()

    out = cv2.VideoWriter('outpy' + videopath.replace("/", "_") + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30,
                        (frame.shape[1], frame.shape[0]))

    bboxes = getDNNBoxes(tensorflowNet, frame)
    trackerlist = []
    numdict = {}
    carcount = 0
    framecount = 1

    # Initialize tracker with first frame and bounding box
    for bbox in bboxes:
        carcount += 1
        trackerbox = (bbox['x1'], bbox['y1'], bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1'])
        tracker = getTracker(tracker_type)
        numdict[tracker] = carcount
        ok = tracker.init(frame, trackerbox)
        trackerlist.append(tracker)

    while True:
        # Read a new frame
        ok, frame = video.read()
        frame = cv2.resize(frame, (1280, 720))
        #frame = imutils.rotate_bound(frame, 180)
        if not ok:
            print("breaking inside loop")
            break

        # update the current trackers, remove trackers that are done
        tlist1 = []
        bboxlist = []
        for tracker in trackerlist:
            ok, bbox = tracker.update(frame)
            bbox = getDict(bbox)
            if outOfFrame(bbox, frame, outlimit): 
                ok = False
            if ok:
                tlist1.append(tracker)
                bboxlist.append(bbox)
                cv2.putText(frame, str(numdict[tracker]), (int(bbox['x1']), int(bbox['y1']) - 15), 0, 0.8, (0, 0, 255), 2,
                            cv2.LINE_AA)
                cv2.rectangle(frame, (int(bbox['x1']), int(bbox['y1'])), (int(bbox['x2']), int(bbox['y2'])), (0, 0, 255),
                            thickness=5)
            else:
                a = numdict.pop(tracker, -1)
                print("Deleting tracker", a)

        trackerlist = tlist1
        gc.collect() #collects the old list and the old trackers

        if framecount % interval == 0:
            # use detection
            try:
                bboxes = getDNNBoxes(tensorflowNet, frame)
            except Exception as e:
                framecount -= 1  # try to infer again next frame
                print(e)
            for bbox in bboxes:
                flag = True
                for bbox1 in bboxlist:
                    if get_iou(bbox, bbox1) >= 0.3:
                        flag = False
                if flag:
                    carcount += 1
                    trackerbox = (bbox['x1'], bbox['y1'], bbox['x2'] - bbox['x1'], bbox['y2'] - bbox['y1'])
                    tracker = getTracker(tracker_type)
                    numdict[tracker] = carcount
                    ok = tracker.init(frame, trackerbox)
                    trackerlist.append(tracker)
        # Draw bounding box
        framecount += 1
        cv2.putText(frame, "No of cars : " + str(carcount), (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 4, (50, 170, 50), 2)
        # Display result
        out.write(frame)
        cv2.imshow("Tracking", frame)
        print(carcount)
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27: break
    out.release()
    return carcount

if __name__=="__main__":
    print(main())