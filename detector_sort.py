#!/usr/bin/env python

import torch
import cv2
from torch.autograd import Variable
from darknet import Darknet
from util import process_result, cv_image2tensor, transform_result
import pickle as pkl
import argparse
import math
import os.path as osp
import os
import sys
from datetime import datetime
from sort import *
import pandas as pd


def load_classes(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names


def parse_args():
    parser = argparse.ArgumentParser(description='Object detection/tracking with YOLOv3 and SORT')
    parser.add_argument('-i', '--input', required=True, help='input directory or video')
    parser.add_argument('-t', '--obj-thresh', type=float, default=0.5, help='objectness threshold, DEFAULT: 0.5')
    parser.add_argument('-n', '--nms-thresh', type=float, default=0.4,
                        help='non max suppression threshold, DEFAULT: 0.4')
    parser.add_argument('--min-hits', default=3, help='A tracker needs to match a bounding box for at least this many '
                                                      'frames before it is registered. Prevents false positives')
    parser.add_argument('--max-age', default=10, help='The number of frames a tracker is kept alive without matching '
                                                      'bounding boxes. Useful for tracker while an object is '
                                                      'temporarily blocked')
    parser.add_argument('-o', '--outdir', default='output', help='output directory, DEFAULT: output/')
    parser.add_argument('-w', '--webcam', action='store_true', default=False,
                        help='flag for detecting from webcam. Specify webcam ID in the input. usually 0 for a single '
                             'webcam connected')
    parser.add_argument('--debug-trackers', action='store_true', default=False,
                        help="Show the kalman trackers instead of the YOLO bounding boxes. Useful for debugging "
                             "and setting parameters, no output is saved.")
    parser.add_argument('--cuda', action='store_true', default=False, help='flag for running on GPU')
    parser.add_argument('--no-show', action='store_true', default=False,
                        help='do not show the detected video in real time')

    args = parser.parse_args()

    return args


def create_batches(imgs, batch_size):
    num_batches = math.ceil(len(imgs) // batch_size)
    batches = [imgs[i * batch_size: (i + 1) * batch_size] for i in range(num_batches)]

    return batches


def draw_simple_bbox(img, bbox, text):
    p1 = tuple(bbox[:2].astype(int))
    p2 = tuple(bbox[2:].astype(int))

    color = [80, 80, 80]

    cv2.rectangle(img, p1, p2, color, 4)
    label = text
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)

    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)


def draw_bbox(imgs, bbox, uid, cls_ind, colors, classes):
    img = imgs[0]
    label = classes[cls_ind]
    p1 = tuple(bbox[0:2].astype(int))
    p2 = tuple(bbox[2:4].astype(int))

    color = colors[cls_ind]
    cv2.rectangle(img, p1, p2, color, 4)
    label = label + '_' + str(int(uid))
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 1)[0]
    p3 = (p1[0], p1[1] - text_size[1] - 4)
    p4 = (p1[0] + text_size[0] + 4, p1[1])
    cv2.rectangle(img, p3, p4, color, -1)

    cv2.putText(img, label, p1, cv2.FONT_HERSHEY_SIMPLEX, 1, [225, 255, 255], 1)


def format_output(bbox, uid, cls_ind, classes, read_frames, output_path, fps):
    label = classes[cls_ind]
    p1 = tuple(bbox[0:2].astype(int))
    p2 = tuple(bbox[2:4].astype(int))

    label = label + '_' + str(int(uid))
    start_time_video = output_path.split('det_')[1].split('.')[0]

    return {
        'uid': uid,
        'label': str(label),
        'type': label.split('_')[0],
        'frame_number': str(read_frames),
        'coord_X_0': int(p1[0]),
        'coord_Y_0': int(p1[1]),
        'coord_X_1': int(p2[0]),
        'coord_Y_1': int(p2[1]),
        'process_time': str(datetime.now()),
        'filename': output_path.replace('.avi', '.csv'),
        'start_time_video': start_time_video,
        'fps': fps
    }


def detect_video(model, args):

    input_size = [int(model.net_info['height']), int(model.net_info['width'])]
    mot_tracker = Sort(min_hits=args.min_hits, max_age=args.max_age)
    colors = pkl.load(open("cfg/pallete", "rb"))
    classes = load_classes("cfg/coco.names")

    # We filter out non-relevant classes, and mapping some classes to others. An alternative to this would be
    # retraining the YOLOV3 model to this smaller set of classes
    relevant_classes = [
        "person",
        "car",
        "bus",
        "train",
        "truck"
    ]
    relevant_classes_indices = [classes.index(cls) for cls in relevant_classes]
    class_mapping = {
        classes.index("car"): [classes.index(cls) for cls in ["bus", "train", "truck"]]
    }

    if not osp.isdir(args.outdir):
        os.mkdir(args.outdir)

    if args.webcam:
        cap = cv2.VideoCapture(int(args.input))
        output_path = osp.join(args.outdir, 'det_webcam.avi')
    else:
        cap = cv2.VideoCapture(args.input)
        output_path = osp.join(args.outdir, 'det_' + osp.basename(args.input).rsplit('.')[0] + '.avi')

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = output_path.replace('.avi', '.mp4')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    read_frames = 0

    start_time = datetime.now()
    print('Detecting...')

    lod = []
    while cap.isOpened():
        retflag, frame = cap.read()
        read_frames += 1
        if retflag:
            frame_tensor = cv_image2tensor(frame, input_size).unsqueeze(0)
            frame_tensor = Variable(frame_tensor)

            if args.cuda:
                frame_tensor = frame_tensor.cuda()

            detections = model(frame_tensor, args.cuda).cpu()
            detections = process_result(detections, args.obj_thresh, args.nms_thresh,
                                        relevant_classes_indices, class_mapping)

            if len(detections) != 0:
                detections = transform_result(detections, [frame], input_size)

            if len(detections) == 0:
                tracked_objects = mot_tracker.update(detections.cpu().detach().numpy())
            else:
                tracked_objects = mot_tracker.update(detections[:, 1:].cpu().detach().numpy())

            if args.debug_trackers:
                for n, tracker in enumerate(mot_tracker.trackers):
                    tracker_bb = tracker.predict()[0]
                    tracker_id = tracker.id
                    draw_simple_bbox(frame, tracker_bb, f"{tracker_id}")
            else:
                if len(tracked_objects) != 0:
                    for obj in tracked_objects:
                        bbox = obj[:4]
                        uid = int(obj[4])
                        cls_ind = int(obj[5])
                        draw_bbox([frame], bbox, uid, cls_ind, colors, classes)
                        lod.append(format_output(bbox, uid, cls_ind, classes, read_frames, output_path, fps))

            if not args.no_show:
                cv2.imshow('frame', frame)
            out.write(frame)
            if read_frames % 30 == 0:
                print(f'Number of frames processed: {read_frames/total_frames:0.2f}%')
            if not args.no_show and cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    end_time = datetime.now()
    print(f'Detection finished in {end_time - start_time}')
    print('Total frames:', read_frames)
    cap.release()
    out.release()
    if not args.no_show:
        cv2.destroyAllWindows()

    print('Detected video saved to ' + output_path)

    name = output_path.replace('.mp4', '.csv')
    pd.DataFrame(lod).to_csv(name, index=False)
    print('Detected meta data saved as ' + name)


def main():
    args = parse_args()

    if args.cuda and not torch.cuda.is_available():
        print("ERROR: cuda is not available, try running on CPU")
        sys.exit(1)

    print('Loading network...')
    model = Darknet("cfg/yolov3.cfg")
    model.load_weights('yolov3.weights')
    if args.cuda:
        model.cuda()

    model.eval()
    print('Network loaded')

    detect_video(model, args)


if __name__ == '__main__':
    main()
