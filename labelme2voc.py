# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import glob
import json
import os
import os.path as osp
import sys

import numpy as np
import PIL.Image

import labelme
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default="datasets/before",help='input annotated directory')
    parser.add_argument('--output_dir',default="VOCdevkit/VOC_1", help='output dataset directory')
    parser.add_argument('--labels', default ="datasets/label.txt" ,help='labels file')
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print('Output directory already exists:', args.output_dir)
        sys.exit(1)
    else:
        os.makedirs(args.output_dir)
        os.makedirs(osp.join(args.output_dir, 'JPEGImages'))
        os.makedirs(osp.join(args.output_dir, 'SegmentationClass'))
        #    os.makedirs(osp.join(args.output_dir, 'SegmentationClassPNG'))
        os.makedirs(osp.join(args.output_dir, 'SegmentationClassVisualization'))
    saved_path = args.output_dir
    if not os.path.exists(os.path.join(saved_path, 'ImageSets', 'Segmentation')):
        os.makedirs(os.path.join(saved_path, 'ImageSets', 'Segmentation'))
    print('Creating dataset:', args.output_dir)

    class_names = []
    class_name_to_id = {}
    for i, line in enumerate(open(args.labels).readlines()):
        print(i)
        class_id = i - 1
        class_name = line.strip()
        class_name_to_id[class_name] = class_id
        if class_id == -1:
            assert class_name == '__ignore__'
            continue
        elif class_id == 0:
            assert class_name == '_background_'
        class_names.append(class_name)
    class_names = tuple(class_names)
    print('class_names:', class_names)
    out_class_names_file = osp.join(args.output_dir, 'class_names.txt')
    with open(out_class_names_file, 'w') as f:
        f.writelines('\n'.join(class_names))
    print('Saved class_names:', out_class_names_file)


    for filename in glob.glob(osp.join(args.input_dir, "*.json")):
        print("Generating dataset from:", filename)

        label_file = labelme.LabelFile(filename=filename)

        base = osp.splitext(osp.basename(filename))[0]
        out_img_file = osp.join(args.output_dir, "JPEGImages", base + ".jpg")
        # out_lbl_file = osp.join(
        #     args.output_dir, "SegmentationClassnpy", base + ".npy"
        # )
        out_png_file = osp.join(
            args.output_dir, "SegmentationClass", base + ".png"
        )
        # if not args.noviz:
        #     out_viz_file = osp.join(
        #         args.output_dir,
        #         "SegmentationClassVisualization",
        #         base + ".jpg",
        #     )

        with open(out_img_file, "wb") as f:
            f.write(label_file.imageData)
        img = labelme.utils.img_data_to_arr(label_file.imageData)

        lbl, _ = labelme.utils.shapes_to_label(
            img_shape=img.shape,
            shapes=label_file.shapes,
            label_name_to_value=class_name_to_id,
        )
        labelme.utils.lblsave(out_png_file, lbl)


    #data agument
    # threadOPS(osp.join(args.output_dir, "JPEGImages"),
    #           osp.join(args.output_dir, "JPEGImages"),
    #           osp.join(args.output_dir, "SegmentationClass"),
    #           osp.join(args.output_dir, "SegmentationClass")
    #           )

    # 6.split files for txt
    txtsavepath = os.path.join(saved_path, 'ImageSets', 'Segmentation')
    ftrainval = open(os.path.join(txtsavepath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(txtsavepath, 'test.txt'), 'w')
    ftrain = open(os.path.join(txtsavepath, 'train.txt'), 'w')
    fval = open(os.path.join(txtsavepath, 'val.txt'), 'w')
    total_files = os.listdir(osp.join(args.output_dir, 'SegmentationClass'))
    total_files = [i.split("/")[-1].split(".png")[0] for i in total_files]
    # test_filepath = ""
    for file in total_files:
        ftrainval.write(file + "\n")
    # test
    # for file in os.listdir(test_filepath):
    #    ftest.write(file.split(".jpg")[0] + "\n")
    # split
    train_files, val_files = train_test_split(total_files, test_size=0.2, random_state=42)
    # train
    for file in train_files:
        ftrain.write(file + "\n")
    # val
    for file in val_files:
        fval.write(file + "\n")

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()


if __name__ == '__main__':
    main()