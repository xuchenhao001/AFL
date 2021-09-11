import os
import re

import numpy as np


def parse_results(file_path):
    # read all lines in file to lines
    with open(file_path, 'r') as file:
        lines = file.readlines()

    file_gather_list = []
    for r in range(len(lines)):
        record = lines[r]
        round_number = int(record[9:12])
        if round_number > 0:  # filter all epochs that greater than zero
            record_trim = record[13:]
            numbers_str = re.findall(r"[-+]?\d*\.\d+|\d+ ", record_trim)
            numbers_float = [float(s) for s in numbers_str]
            file_gather_list.append(numbers_float)
    return file_gather_list


def calculate_files_and_mean(experiment_path):
    result_files = [f for f in os.listdir(experiment_path) if f.startswith('result-record_')]
    result_files.sort()

    files_numbers_3d = []
    for result_file in result_files:
        file_path = os.path.join(experiment_path, result_file)
        file_numbers_2d = parse_results(file_path)  # parse each file into two dimensional array
        files_numbers_3d.append(file_numbers_2d)
    files_numbers_3d_np = np.array(files_numbers_3d)
    files_numbers_mean_2d_np = files_numbers_3d_np.mean(axis=0)
    return files_numbers_3d_np, files_numbers_mean_2d_np


def extract_time_data():
    exp_node_number = "attack"
    model_name = "cnn"
    dataset_name = "cifar"

    experiment_names = ["fed_async", "fed_async_f10"]

    for path, dirs, files in os.walk("./output"):
        if path.endswith(model_name + "-" + dataset_name) and exp_node_number in path:
            for experiment_name in experiment_names:
                print(experiment_name + ":")
                experiment_path = os.path.join(path, experiment_name)
                files_numbers_3d_np, files_numbers_mean_2d_np = calculate_files_and_mean(experiment_path)
                acc_average = [round(i, 2) for i in files_numbers_mean_2d_np[:, 5]]
                acc_node_1 = [round(i, 2) for i in files_numbers_3d_np[0][:, 5]]
                acc_node_2 = [round(i, 2) for i in files_numbers_3d_np[1][:, 5]]
                acc_node_3 = [round(i, 2) for i in files_numbers_3d_np[2][:, 5]]
                acc_node_4 = [round(i, 2) for i in files_numbers_3d_np[3][:, 5]]
                acc_node_5 = [round(i, 2) for i in files_numbers_3d_np[4][:, 5]]
                print("acc_average", "=", acc_average)
                print("acc_node_1", "=", acc_node_1)
                print("acc_node_2", "=", acc_node_2)
                print("acc_node_3", "=", acc_node_3)
                print("acc_node_4", "=", acc_node_4)
                print("acc_node_5", "=", acc_node_5)


def main():
    extract_time_data()


if __name__ == "__main__":
    main()
