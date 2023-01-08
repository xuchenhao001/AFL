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


def calculate_files_mean(experiment_path):
    result_files = [f for f in os.listdir(experiment_path) if f.startswith('result-record_')]

    files_numbers_3d = []
    for result_file in result_files:
        file_path = os.path.join(experiment_path, result_file)
        file_numbers_2d = parse_results(file_path)  # parse each file into two dimensional array
        files_numbers_3d.append(file_numbers_2d)
    files_numbers_3d_np = np.array(files_numbers_3d)
    files_numbers_mean_2d_np = files_numbers_3d_np.mean(axis=0)
    return files_numbers_mean_2d_np


def extract_time_data():
    exp_node_number = "iot-2-v1"
    model_name = "cnn"
    dataset_name = "fashion_mnist"

    # experiment_names = ["fed_async", "fed_avg", "fed_localA", "fed_sync", "local_train"]
    experiment_names = ["fed_asofed", "fed_befl"]

    experiment_results = {}
    for path, dirs, files in os.walk("./output"):
        if path.endswith(model_name + "-" + dataset_name) and exp_node_number in path:
            for experiment_name in experiment_names:
                experiment_path = os.path.join(path, experiment_name)
                files_numbers_mean_2d_np = calculate_files_mean(experiment_path)
                round_time = [round(i, 2) for i in files_numbers_mean_2d_np[:, 1]]
                train_time = [round(i, 2) for i in files_numbers_mean_2d_np[:, 2]]
                test_time = [round(i, 2) for i in files_numbers_mean_2d_np[:, 3]]
                communication_time = [round(i, 2) for i in files_numbers_mean_2d_np[:, 4]]
                result_time_array = [
                    round_time,
                    train_time,
                    test_time,
                    communication_time,
                ]
                experiment_results[experiment_name] = result_time_array

    time_array = ["overall_time", "train_time", "test_time", "communication_time"]
    for time_idx in range(len(time_array)):
        print(time_array[time_idx] + ": ")
        for experiment_name in experiment_names:
            print(experiment_name, "=", experiment_results[experiment_name][time_idx])


def main():
    extract_time_data()


if __name__ == "__main__":
    main()
