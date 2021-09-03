import os
import subprocess
from pathlib import Path


def plot_all():
    experiment_names = ["cnn-cifar", "cnn-fmnist", "lstm-loop", "mlp-fmnist"]

    # real_path = os.path.dirname(os.path.realpath(__file__))
    Path("./figures").mkdir(parents=True, exist_ok=True)
    for experiment in experiment_names:
        for path, dirs, files in os.walk("./" + experiment):
            plot_subdir = os.path.join("./figures", path)
            Path(plot_subdir).mkdir(parents=True, exist_ok=True)
            for file in files:
                if file.endswith(".py"):
                    python_file_path = os.path.join(path, file)
                    output_file_path = os.path.join(plot_subdir, file.rstrip(".py") + ".png")
                    subprocess.call(['python3', python_file_path, "save", output_file_path])


def main():
    plot_all()


if __name__ == "__main__":
    main()
