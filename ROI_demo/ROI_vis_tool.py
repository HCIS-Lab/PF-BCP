import argparse
import sys

from PyQt5 import QtWidgets
from utils.controller import MainWindow_controller


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_root', default="./RiskBench_Dataset", type=str)
    parser.add_argument('--metadata_root', default="./metadata/RiskBench", type=str)
    parser.add_argument('--model_root', default="./models", type=str)
    parser.add_argument('--vis_result_path', default="./ROI_vis_result", type=str)
    parser.add_argument('--FPS', default=20, type=int)
    parser.add_argument('--print_text', action='store_true', default=False)
    args = parser.parse_args()

    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow_controller(args)
    window.show()
    sys.exit(app.exec_())
