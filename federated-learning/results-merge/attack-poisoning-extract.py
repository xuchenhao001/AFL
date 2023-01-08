import os
import re

import pandas as pd


def extract_file_lines(exp_node_number, model_name, dataset_name, experiment):
    result_lines = []
    result_path = ""
    for path, dirs, files in os.walk("./output"):
        if path.endswith(model_name + "-" + dataset_name) and exp_node_number in path:
            if experiment in dirs:
                result_path = os.path.join(path, experiment)
                break
    files = [f for f in os.listdir(result_path) if f.startswith('result-record_')]
    for user_id in range(len(files)):
        file_path = os.path.join(result_path, "result-record_" + str(user_id + 1) + ".txt")
        with open(file_path, 'r') as file:
            result_lines.append(file.readlines())
    return result_lines


def extract_values(user_lines):
    user_values = []
    for user_id in range(len(user_lines)):
        round_gather_list = []
        for rd in range(len(user_lines[user_id])):
            record = user_lines[user_id][rd]
            record_trim = record[13:]
            numbers_str = re.findall(r"[-+]?\d*\.\d+|\d+ ", record_trim)
            values = [float(s) for s in numbers_str]
            round_gather_list.append(values)
        user_values.append(pd.DataFrame(round_gather_list))
    return user_values


def extract_by_timeline(result_value_dfs, sampling_frequency, final_time):
    sampling_time = 0
    # acc_result_list_2d for node 1, 2, ..., average
    acc_result_list_2d = [[] for i in result_value_dfs]
    acc_result_list_2d.append([])

    while True:
        sampling_time += sampling_frequency
        avg_acc_list = []
        for index, df in enumerate(result_value_dfs):
            # locate the largest row smaller than sampling_time
            if df[0].gt(sampling_time).any():
                latest_sample_row = df.iloc[[df[df[0].gt(sampling_time)].index[0] - 1]]
                latest_acc = latest_sample_row.iloc[0, 5]
                acc_result_list_2d[index].append(round(latest_acc, 2))
                avg_acc_list.append(latest_acc)
            else:
                acc_result_list_2d[index].append(None)
        if len(avg_acc_list) == 0:
            avg_acc_list.append(None)
        else:
            avg = sum(avg_acc_list) / len(avg_acc_list)
            acc_result_list_2d[-1].append(round(avg, 2))
        if sampling_time + sampling_frequency >= final_time:
            break
    return acc_result_list_2d


def extract_time_data():
    sampling_frequency = 3  # sampling frequency (seconds)
    final_time = 300
    exp_node_number = "attack"
    model_name = "cnn"
    dataset_name = "cifar"

    experiment_names = ["fed_async_classic", "fed_async_defense_00", "fed_async_defense_80", "fed_async_defense_90"]

    for path, dirs, files in os.walk("./output"):
        if path.endswith(model_name + "-" + dataset_name) and exp_node_number in path:
            for experiment_name in experiment_names:
                print(experiment_name + ":")
                result_lines = extract_file_lines(exp_node_number, model_name, dataset_name, experiment_name)
                result_value_dfs = extract_values(result_lines)
                acc_result_list_2d = extract_by_timeline(result_value_dfs, sampling_frequency, final_time)

                acc_average = acc_result_list_2d[-1]
                acc_node_1 = acc_result_list_2d[0]
                acc_node_2 = acc_result_list_2d[1]
                acc_node_3 = acc_result_list_2d[2]
                acc_node_4 = acc_result_list_2d[3]
                acc_node_5 = acc_result_list_2d[4]
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
