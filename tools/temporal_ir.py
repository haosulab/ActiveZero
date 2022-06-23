"""
Author: Isabella Liu 10/28/21
Feature: Extract IR pattern from temporal real images
"""
# TODO: should be moved to a new dir like `tools` because it runs independently
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

parser = argparse.ArgumentParser(
    description="Extract IR pattern from temporal real images"
)
parser.add_argument(
    "--split-file",
    type=str,
    default="/cephfs/datasets/iccv_pnp/messy-table-dataset/rand_scenes/train_list.txt",
    metavar="FILE",
)
parser.add_argument(
    "--data-folder",
    type=str,
    default="/cephfs/datasets/iccv_pnp/messy-table-dataset/rand_scenes/rand_scenes",
)
args = parser.parse_args()

x = np.linspace(0, 6, num=7, dtype=int)
x_avg = np.average(x)
x_avg_diff = x - x_avg


def get_smoothed_ir_pattern(diff: np.array, ks=9, threshold=0.005):
    diff = np.abs(diff)
    diff_avg = cv2.blur(diff, (ks, ks))
    ir = np.zeros_like(diff)
    ir[diff - diff_avg > threshold] = 1
    return ir


def get_regression(y):
    # total_num = len(y)
    # x = np.linspace(0, total_num-1, num=total_num, dtype=int)
    # x_avg = np.average(x)
    # x_avg_diff = x - x_avg
    y_avg = np.average(y)
    y_avg_diff = y - y_avg
    slope = np.sum(x_avg_diff * y_avg_diff) / np.sum(x_avg_diff ** 2)
    intercept = y_avg - slope * x_avg
    pred = slope * x + intercept
    return slope, pred


def main():
    with open(args.split_file, "r") as f:
        prefix = [line.strip() for line in f]
    num = len(prefix)

    for idx in range(num):
        for direction in ["irL", "irR"]:
            p = prefix[idx]
            f0 = os.path.join(args.data_folder, p, f"1024_{direction}_real_off.png")
            f1 = os.path.join(args.data_folder, p, f"1024_{direction}_real_060.png")
            f2 = os.path.join(args.data_folder, p, f"1024_{direction}_real_120.png")
            f3 = os.path.join(args.data_folder, p, f"1024_{direction}_real_180.png")
            f4 = os.path.join(args.data_folder, p, f"1024_{direction}_real_240.png")
            f5 = os.path.join(args.data_folder, p, f"1024_{direction}_real_300.png")
            f6 = os.path.join(args.data_folder, p, f"1024_{direction}_real_360.png")
            img_0 = np.array(Image.open(f0).convert(mode="L"))
            img_1 = np.array(Image.open(f1).convert(mode="L"))
            img_2 = np.array(Image.open(f2).convert(mode="L"))
            img_3 = np.array(Image.open(f3).convert(mode="L"))
            img_4 = np.array(Image.open(f4).convert(mode="L"))
            img_5 = np.array(Image.open(f5).convert(mode="L"))
            img_6 = np.array(Image.open(f6).convert(mode="L"))
            img_temp = np.concatenate(
                (
                    img_0[:, :, None],
                    img_1[:, :, None],
                    img_2[:, :, None],
                    img_3[:, :, None],
                    img_4[:, :, None],
                    img_5[:, :, None],
                    img_6[:, :, None],
                ),
                axis=-1,
            )

            # Get regression fit on temporal images
            print(f"Generating {p} temporal {direction} pattern {idx}/{num}")
            h, w, d = img_temp.shape
            x = np.linspace(0, d - 1, num=d, dtype=int).reshape(1, 1, -1)
            x = np.repeat(x, h, axis=0)
            x = np.repeat(x, w, axis=1)  # [H, W, D]
            x_avg = np.average(x, axis=-1).reshape(h, w, 1)

            y = img_temp  # [H, W, D]
            y_avg = np.average(y, axis=-1).reshape(h, w, 1)

            numerator = np.sum((y - y_avg) * (x - x_avg), axis=-1)
            denominator = np.sum((x - x_avg) ** 2, axis=-1)  # [H, W]
            slope = numerator / denominator  # [H, W]
            slope = slope[:, :, None]
            intercept = y_avg - slope * x_avg
            img_temp_fit = slope * x + intercept

            # Get IR pattern
            diff = (img_temp_fit[:, :, -1] - img_temp_fit[:, :, 0]) / 255
            diff = np.abs(diff)
            # Normalize to [0,1]
            diff = (diff - np.min(diff)) / (np.max(diff) - np.min(diff))
            pattern = get_smoothed_ir_pattern(diff, ks=11, threshold=0.005)

            # Save extracted IR pattern
            plt.imsave(
                os.path.join(
                    args.data_folder, p, f"1024_{direction}_real_temporal.png"
                ),
                pattern,
            )
            plt.close("all")


if __name__ == "__main__":
    main()
