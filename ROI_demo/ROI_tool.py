import argparse
import json
import os

from utils.utils import read_metadata, filter_roi_scenario
from utils.cal_metric import ROI_evaluation


def show_result(metric_result):

    method = metric_result['Method']
    attribute = metric_result['Attribute']
    data_type = metric_result['type']

    TP, FN, FP, TN = metric_result["confusion matrix"].values()
    OT_R, OT_P, OT_F1 = metric_result["recall"], metric_result["precision"],  metric_result["f1-Score"]
    # accuracy = metric_result["accuracy"]
    # IDcnt, IDsw, IDsw_rate = metric_result['IDcnt'], metric_result['IDsw'], metric_result['IDsw rate']
    PIC = metric_result['PIC']
    wMOTA = metric_result['wMOTA']
    

    print(f"Method: {method}\tAttribute: {attribute}, type: {data_type}")
    print(f"TP: {TP}, FN: {FN}, FP: {FP}, TN: {TN}")
    print(f"OT-Recall: {OT_R}, OT-Precision: {OT_P}, OT-F1: {OT_F1}")
    print(f"PIC: {PIC}, wMOTA: {wMOTA}")

    for i in range(1, 4):
        print(f"OT-F1-{i}s:", metric_result[f"OT-F1-{i}s"])

    print()
    print("="*60)


def save_result(result_path, data_type, method, metric_result):

    save_folder = os.path.join(result_path, method)
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)

    with open(os.path.join(save_folder, f"{data_type}.json"), 'w') as f:
        json.dump(metric_result, f, indent=4)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--method', choices=["FF", "DSA", "RRL", \
                    "BP", "BCP", "TP+BCP", "BS+BCP", "OFDE", "OADE", "PF+BCP"], required=True, type=str)
    parser.add_argument('--data_type', default='interactive', type=str)
    parser.add_argument('--metadata_root', default="./metadata/RiskBench", type=str)
    parser.add_argument('--model_root', default="./models", type=str)
    parser.add_argument('--result_path', default="./ROI_result", type=str)
    parser.add_argument('--save_result', action='store_true', default=False)

    args = parser.parse_args()

    method = args.method
    data_type = args.data_type
    metadata_root = args.metadata_root
    model_root = args.model_root


    behavior_dict, risky_dict, critical_dict = read_metadata(data_type=data_type, metadata_root=metadata_root)
    roi_result = filter_roi_scenario(data_type, method, attr_list="All", model_root=model_root)
    _, metric_result = ROI_evaluation(data_type, method, roi_result, behavior_dict, risky_dict, critical_dict, attribute="All")


    if args.save_result:
        save_result(args.result_path, data_type, method, metric_result)

    show_result(metric_result)
