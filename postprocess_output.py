#!/usr/bin/env python

import pandas as pd
import numpy as np
from numpy.linalg import norm
import argparse
from matplotlib import cm
from PIL import Image, ImageDraw
from math import atan2

import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser(description='Postprocessing of csv output from video-detection script')
    parser.add_argument('-i', '--input', required=True, help='input csv')
    parser.add_argument('-p', '--plot', action="store_true", help="enables plotting of paths")
    parser.add_argument('-g', '--gap-thresh', type=float, default=0, help='broken path threshold, DEFAULT: 20')
    parser.add_argument('-a', '--angle-thresh', type=float, default=0, help='broken angle threshold, DEFAULT: 0.5')
    parser.add_argument('-m', '--move-thresh', type=float, default=10, help='static object threshold, DEFAULT: 10')
    parser.add_argument('-t', '--time-freq', default="5min", help="time frequency used for aggregating the output, see"
                                                                  "pandas.dt.round() for valid input values. DEFAULT "
                                                                  "'5min'")
    parser.add_argument('-o', '--outdir', default='output', help='output directory, DEFAULT: output/')

    args = parser.parse_args()

    return args


def draw_paths(df, image_size):
    im = Image.new('RGB', image_size, color="black")
    draw = ImageDraw.Draw(im)
    colormap = cm.viridis

    print(f"Drawing {len(df.uid.unique())} lines to image")
    for uid in df.uid.unique():
        path = np.array(df.loc[df["uid"] == uid, ["x", "y"]])
        for i in range(len(path) - 1):
            color = colormap(i / len(path))
            draw.line(path[i:i + 2].reshape(-1).tolist(), fill=tuple(int(255 * c) for c in color[:3]))

    im.show()


def angle_between(v1, v2):
    """
    Helper function which returns the unsigned angle between v1 and v2.
    The output ranges between 0 and 1, where (v1 == v2) -> 0 and (v1 == -v2) -> 1.
    """
    return np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


def fix_broken_paths(df, gap_thresh, angle_thresh):
    """
    This functions tries to match object paths together which were broken due to tracker problems.
    1. Define start/end boolean columns
    2. For each uid, and each end: filter to starts within (x,y,t) distance.
    3. If matches found, update uid of extra path
    4. Remove paths which don't move more than threshold over whole life.
    """

    df["head"] = False
    df["tail"] = False

    df.loc[df.groupby("uid").head(1).index, "head"] = True
    df.loc[df.groupby("uid").tail(1).index, "tail"] = True

    xyt = np.array([df["x"], df["y"], df["frame_number"]]).T
    xy_diff = np.diff(xyt[:, :2], axis=0)
    xy_diff = np.concatenate([xy_diff, [[np.nan, np.nan]]])  # Added to make lengths same as xyt

    uid_mapping = []

    for uid in df["uid"].unique():
        our_tail_bool = (df["uid"] == uid) & df["tail"]
        other_head_bool = (df["uid"] != uid) & df["head"]
        our_xyt = xyt[our_tail_bool]

        # The vector to the tail is indexed 1 less
        tail_vec = xy_diff[np.where([False, False, True])[0][0] - 1]
        dot_product = (tail_vec * xy_diff).sum(axis=1)

        angles = np.arccos(dot_product / (norm(tail_vec) * norm(xy_diff, axis=1))) / np.pi
        distances = np.linalg.norm(xyt - our_xyt, axis=1)

        with np.errstate(invalid='ignore'):  # Needed since "nan < angle_thresh" gives runtime warning
            matches_ind, = np.where((angles < angle_thresh) & (distances < gap_thresh) & other_head_bool)
        if len(matches_ind) > 0:
            best_match = np.argmin(distances[matches_ind])
            match_uid = df.loc[matches_ind[best_match], "uid"]
            uid_mapping.append([uid, match_uid])

    df = df.drop(["head", "tail"], axis=1)

    return uid_mapping


def vec2angle(vec):
    """
    Small helper function to translate an (x,y) vector into an angle in radians
    :param vec: np.array()
    :return: string
    """
    return round(atan2(vec[1], vec[0]), 3)


def get_statics(df, move_thresh):
    """
    Filter out static objects. For each object id we get the corresponding points, and check how far it's moved over
    it's whole lifespan. If this is less than 10px, we remove it.
    """
    ret_ids = []

    for uid in df["uid"].unique():
        path = np.array(df.loc[df["uid"] == uid, ["x", "y"]])
        path_segments = np.diff(path, axis=0)

        if (np.abs(path_segments.sum(axis=0)) < move_thresh).all():
            ret_ids.append(uid)

    print(f"Detected {len(ret_ids)} static objects")

    return ret_ids


def get_screen_area(x, y):
    """
    Hard coded for now, this should rely on some external data format
    :param x:
    :param y:
    :return:
    """
    return NotImplemented


def format_output(df, segments=5):
    """
    Here we convert the raw_data_df from frame-by-frame data on the tracker bounding boxes to
    direction data per uid so that we can aggregate this into traffic numbers.
    """
    lod = []
    for uid in df["uid"].unique():
        path = np.array(df.loc[df["uid"] == uid, ["x", "y"]])
        start_vec = np.diff(path[:segments], axis=0).sum(axis=0)
        end_vec = np.diff(path[-segments:], axis=0).sum(axis=0)

        row = df[df["uid"] == uid].iloc[0]
        lod.append({
            "uid": row["uid"],
            "label": row["type"],
            "ts": row["start_time_video"],
            "start_coord": path[0].tolist(),
            "end_coord": path[-1].tolist(),
            "start_angle": vec2angle(start_vec),
            "end_angle": vec2angle(end_vec)
        })

    return pd.DataFrame(lod)


def main():
    """
    Where the magic happens
    """
    args = parse_args()
    print(args)

    results_df = pd.read_csv(args.input).drop(["label"], axis=1)  # Gets screwed up by postprocessing, so we drop it

    results_df = results_df.assign(x=lambda x: (0.5 * (x['coord_X_0'] + x['coord_X_1'])).astype(int),
                                   y=lambda x: (0.5 * (x['coord_Y_0'] + x['coord_Y_1'])).astype(int))

    height = max(np.max(results_df[["coord_Y_1", "coord_Y_0"]]))
    width = max(np.max(results_df[["coord_X_1", "coord_X_0"]]))

    # Detect paths which have been broken due to detection/tracking problems
    uid_mapping = fix_broken_paths(results_df, gap_thresh=args.gap_thresh, angle_thresh=args.angle_thresh)
    print(f"Found {len(uid_mapping)} breaks to be patched")

    for uid_from, uid_to in uid_mapping:
        results_df.loc[results_df["uid"] == uid_from, "uid"] = uid_to

    # Remove non-moving objects
    static_ind = results_df["uid"].isin(get_statics(results_df, move_thresh=args.move_thresh))
    results_df = results_df[~static_ind]

    if args.plot:
        draw_paths(results_df, [width, height])

    output_df = format_output(results_df, segments=5)

    # Exporting results to csv
    input_filename = osp.split(args.input)[-1]
    output_file = osp.join(args.outdir, "results_" + input_filename)
    output_df.to_csv(output_file, index=False)
    print(f"Output written to {output_file}")


if __name__ == "__main__":
    main()
