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
    avg_list = []
    while True:
        sampling_time += sampling_frequency
        acc_list = []
        for df in result_value_dfs:
            # locate the largest row smaller than sampling_time
            if df[0].gt(sampling_time).any():
                latest_sample_row = df.iloc[[df[df[0].gt(sampling_time)].index[0] - 1]]
                latest_acc = latest_sample_row.iloc[0, 5]
                acc_list.append(latest_acc)
        if sampling_time + sampling_frequency >= final_time:
            break
        if len(acc_list) == 0:
            avg_list.append(None)
        else:
            avg = sum(acc_list) / len(acc_list)
            avg_list.append(round(avg, 2))
    return avg_list


def main():
    sampling_frequency = 5  # sampling frequency (seconds)
    final_time = 500
    # exp_node_number = "dis-4-network"
    exp_node_number = "dis-4-v1"
    # exp_node_number = "static-fade"
    model_name = "lstm"
    dataset_name = "loop"
    # experiments = ["fed_async", "fed_avg", "fed_sync", "fed_localA", "local_train"]
    experiments = ["fed_asofed", "fed_befl"]
    # experiments = ["fed_async", "fed_async_f05", "fed_async_f10", "fed_async_f15"]
    for experiment in experiments:
        result_lines = extract_file_lines(exp_node_number, model_name, dataset_name, experiment)
        result_value_dfs = extract_values(result_lines)
        sampling_avg_list = extract_by_timeline(result_value_dfs, sampling_frequency, final_time)
        print(experiment, "=", sampling_avg_list)


if __name__ == "__main__":
    main()
