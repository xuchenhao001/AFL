import os

import pandas as pd


# average cost time
def average_column_value(path, column_num):
    df = pd.read_csv(path)
    mean_value = df.iloc[:, column_num].mean()
    return mean_value


def extract_skew_data():
    model_name = "cnn"
    dataset_name = "cifar"

    experiment_names = ["fed_async", "fed_avg", "fed_localA", "fed_sync", "local_train"]

    experiment_results = []
    for path, dirs, files in os.walk("./output"):
        if path.endswith(model_name + "-" + dataset_name):
            for i in range(len(experiment_names)):
                result_file = os.path.join(path, experiment_names[i], "merged.csv")
                result_train = average_column_value(result_file, 2)
                result_test = average_column_value(result_file, 3)
                result_communication = average_column_value(result_file, 4)
                result_round = average_column_value(result_file, 1)
                experiment_results.append([round(result_train, 2), round(result_test, 2),
                                           round(result_communication, 2), round(result_round, 2)])

    print(model_name, "-", dataset_name)

    # print(experiment, "=", sampling_avg_list)

    for i in range(len(experiment_names)):
        print(experiment_names[i], "=", experiment_results[i])
    # print()
    # for j in range(len(experiment_results[0])):
    #     for i in range(len(experiment_names)):
    #         print(experiment_results[i][j], end='\t')
    #     print()


def main():
    extract_skew_data()


if __name__ == "__main__":
    main()
