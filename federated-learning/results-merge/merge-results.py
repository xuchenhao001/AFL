#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import numpy as np
import os
import re


def merge_files(user_number, merge_files_path):
    # read all of the files into lines[]
    lines = []
    for user_id in range(user_number):
        filename = os.path.join(merge_files_path, "result-record_" + str(user_id + 1) + ".txt")
        with open(filename, 'r') as file:
            lines.append(file.readlines())

    # start to process lines[]
    round_num = len(lines[0])

    # first time clean the file
    merged_file_name = os.path.join(merge_files_path, "merged.csv")
    open(merged_file_name, 'w').close()
    for r in range(round_num):
        round_gather_list = []
        for user_id in range(user_number):
            record = lines[user_id][r]
            record_trim = record[13:]
            numbers_str = re.findall(r"[-+]?\d*\.\d+|\d+ ", record_trim)
            numbers = [float(s) for s in numbers_str]
            round_gather_list.append(numbers)
        data = np.array(round_gather_list)
        avg = np.average(data, axis=0).tolist()
        avg_pretty = ["{0:0.2f}".format(i) for i in avg]
        avg_str = ",".join(avg_pretty)
        with open(merged_file_name, 'a') as file:
            file.write(avg_str + "\n")


def main():
    # traverse directory, and list directories as dirs and files as files
    # https://stackoverflow.com/questions/16953842/using-os-walk-to-recursively-traverse-directories-in-python
    for path, dirs, files in os.walk("./output"):
        # print(dirpath, os.path.basename(dirpath))
        if len(files) > 0 and files[0].endswith('.txt'):
            resut_files = [file for file in files if not file.endswith('merged.csv')]
            user_number = len(resut_files)
            merge_files(user_number, path)


if __name__ == "__main__":
    main()
