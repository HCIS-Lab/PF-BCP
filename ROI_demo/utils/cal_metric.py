import numpy as np

FRAME_PER_SEC = 20
PRED_SEC = 3
TOTAL_FRAME = FRAME_PER_SEC*PRED_SEC


def cal_confusion_matrix(data_type, roi_result, risky_dict, behavior_dict=None, critical_dict=None):

    TP, FN, FP, TN = 0, 0, 0, 0

    TOTAL_FRAME = FRAME_PER_SEC*PRED_SEC
    confusion_matrix_sec = np.zeros((PRED_SEC+1, 4))

    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = 999, -999

        if data_type != "non-interactive":
            critical_frame = critical_dict[scenario_weather]
        else:
            critical_frame = 999


        if data_type != "non-interactive":
            risky_id = risky_dict[scenario_weather][0]
        else:
            risky_id = None

        for frame_id in roi_result[scenario_weather]:

            _TP, _FN, _FP, _TN = 0, 0, 0, 0
            all_actor_id = list(roi_result[scenario_weather][str(frame_id)].keys())
            behavior_stop = (start <= int(frame_id) <= end)

            for actor_id in all_actor_id:

                is_risky = roi_result[scenario_weather][str(frame_id)][actor_id]

                if behavior_stop and actor_id == risky_id:
                    if is_risky:
                        _TP += 1

                    else:
                        _FN += 1
                else:
                    if is_risky:
                        _FP += 1
                    else:
                        _TN += 1

            if 0 <= critical_frame - int(frame_id) < TOTAL_FRAME:
                confusion_matrix_sec[(critical_frame - int(frame_id))//FRAME_PER_SEC +
                       1, :] += np.array([_TP, _FN, _FP, _TN])

            TP += _TP
            FN += _FN
            FP += _FP
            TN += _TN

    return [TP, FN, FP, TN], confusion_matrix_sec


def cal_IDsw(data_type, roi_result, risky_dict, behavior_dict):

    IDcnt = {"risky":0, "non-risky":0, "total":0}
    IDsw = {"risky":0, "non-risky":0, "total":0}

    for scenario_weather in roi_result.keys():

        pre_frame_info = None

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = 999, -999

        if data_type != "non-interactive":
            risky_id = risky_dict[scenario_weather][0]
        else:
            risky_id = None

        for frame_id in roi_result[scenario_weather]:

            cur_frame_info = roi_result[scenario_weather][str(frame_id)]
            all_actor_id = list(cur_frame_info.keys())

            behavior_stop = (start <= int(frame_id) <= end)

            for actor_id in all_actor_id:

                if behavior_stop and actor_id == risky_id:
                    IDcnt["risky"] += 1
                    IDcnt["total"] += 1
                    if not pre_frame_info is None:
                        if actor_id in pre_frame_info and cur_frame_info[actor_id] != pre_frame_info[actor_id]:
                            IDsw["risky"] += 1
                            IDsw["total"] += 1

                else:
                    IDcnt["non-risky"] += 1
                    IDcnt["total"] += 1
                    if not pre_frame_info is None:
                        if actor_id in pre_frame_info and cur_frame_info[actor_id] != pre_frame_info[actor_id]:
                            IDsw["non-risky"] += 1
                            IDsw["total"] += 1

            pre_frame_info = cur_frame_info

    IDsw_rate = (IDsw["total"])/IDcnt["total"]
    return IDcnt, IDsw, IDsw_rate


def cal_MOTA(cur_confusion_matrix, IDsw, IDcnt, EPS=1e-05):

    FN, FP = cur_confusion_matrix[1:3]
    wMOTA = 1-((FN+IDsw["risky"])+(FP+IDsw["non-risky"])*IDcnt["risky"]/IDcnt["non-risky"]) / (2*IDcnt["risky"])

    return wMOTA


def cal_PIC(data_type, roi_result, behavior_dict, risky_dict, critical_dict, EPS=1e-08):

    assert data_type != "non-interactive", "non-interactive can not calculate PIC!!!"

    PIC = 0

    for scenario_weather in roi_result.keys():

        basic = '_'.join(scenario_weather.split('_')[:-3])
        variant = '_'.join(scenario_weather.split('_')[-3:])
        if data_type in ["interactive", "obstacle"]:
            start, end = behavior_dict[basic][variant]
        else:
            start, end = 0, 999

        end_frame = critical_dict[scenario_weather]
        risky_id = risky_dict[scenario_weather][0]

        for frame_id in roi_result[scenario_weather]:

            if int(frame_id) > int(end_frame):
                break

            all_actor_id = list(roi_result[scenario_weather][frame_id].keys())

            if len(all_actor_id) == 0:
                continue

            TP, FN, FP, TN = 0, 0, 0, 0
            behavior_stop = (start <= int(frame_id) <= end)

            for actor_id in all_actor_id:

                is_risky = roi_result[scenario_weather][frame_id][actor_id]

                if behavior_stop and actor_id == risky_id:
                    if is_risky:
                        TP += 1
                    else:
                        FN += 1
                else:
                    if is_risky:
                        FP += 1
                    else:
                        TN += 1

            recall, precision, f1 = compute_f1(np.array([TP, FN, FP, TN]).astype(int))

            # exponential F1 loss
            if TP+FP+FN > 0 and 0 <= int(end_frame)-int(frame_id) < 60:
                c = 1.0
                PIC += -(np.exp(c*-(int(end_frame)-int(frame_id))/60)
                        * np.log(f1 + EPS))


    PIC = PIC/len(roi_result.keys())
    return PIC


def compute_f1(confusion_matrix, EPS=1e-5):

    TP, FN, FP, TN = confusion_matrix

    recall = TP / (TP+FN+EPS)
    precision = TP / (TP+FP+EPS)
    f1_score = 2*precision*recall / (precision+recall+EPS)

    return recall, precision, f1_score


def cal_OT_F1_T(confusion_matrix_sec):

    # calculate OT-Fl-T
    _, _, overall_f1 = compute_f1(np.sum(confusion_matrix_sec, axis=0))
    OT_F1_T = {"f1": overall_f1, "Current": {}, "Accumulation": {}}

    # confusion matrix
    confusion_matrix_sec_sum = np.zeros(4)

    for i in range(1, PRED_SEC+1):

        confusion_matrix_sec_sum += confusion_matrix_sec[i]
        r, p, f1 = compute_f1(confusion_matrix_sec_sum)
        OT_F1_T["Accumulation"][i] = f1

        r, p, f1 = compute_f1(confusion_matrix_sec[i])
        OT_F1_T["Current"][i] = f1

    return OT_F1_T


def cal_metric(data_type, method, confusion_matrix, OT_R, OT_P, OT_F1, OT_F1_T, IDcnt, IDsw, IDsw_rate, wMOTA, PIC, attribute="All", EPS=1e-05):

    TP, FN, FP, TN = confusion_matrix


    metric_result_raw = {"Method": method, "Attribute": attribute, "type": data_type,
                         "confusion matrix": {"TP": TP, "FN": FN, "FP": FP, "TN": TN},
                         "recall": OT_R, "precision": OT_P,
                         "accuracy": (TP+TN)/(TP+FN+FP+TN+EPS), "f1-Score": OT_F1,
                         "IDcnt": IDcnt['total'], "IDsw": IDsw['total'], "IDsw rate": IDsw_rate,
                         "wMOTA": wMOTA, "PIC": PIC,
                         "OT-F1-1s": OT_F1_T['Accumulation'][1],
                         "OT-F1-2s": OT_F1_T['Accumulation'][2],
                         "OT-F1-3s": OT_F1_T['Accumulation'][3]}

    for key in metric_result_raw:
        if isinstance(metric_result_raw[key], np.generic):
            metric_result_raw[key] = float(metric_result_raw[key])

    
    metric_result_str = {"Method": method, "Attribute": attribute, "type": data_type,
                         "confusion matrix": {"TP": TP, "FN": FN, "FP": FP, "TN": TN},
                         "recall": f"{OT_R*100:.2f}%", "precision": f"{OT_P*100:.2f}%",
                         "accuracy": f"{(TP+TN)/(TP+FN+FP+TN+EPS)*100:.2f}%", "f1-Score": f"{OT_F1*100:.2f}%",
                         "IDcnt": f"{IDcnt['total']}", "IDsw": f"{IDsw['total']}", "IDsw rate": f"{IDsw_rate*100:.2f}%",
                         "wMOTA": f"{wMOTA*100:.2f}%", "PIC": f"{PIC:.1f}",
                         "OT-F1-1s": f"{OT_F1_T['Accumulation'][1]*100:.2f}%",
                         "OT-F1-2s": f"{OT_F1_T['Accumulation'][2]*100:.2f}%",
                         "OT-F1-3s": f"{OT_F1_T['Accumulation'][3]*100:.2f}%"}


    return metric_result_raw, metric_result_str


def ROI_evaluation(data_type, method, roi_result, behavior_dict=None, risky_dict=None, critical_dict=None, attribute="All"):

    PIC = -1

    confusion_matrix, confusion_matrix_sec = cal_confusion_matrix(data_type, roi_result, risky_dict, behavior_dict, critical_dict)
    OT_R, OT_P, OT_F1 = compute_f1(confusion_matrix)
    OT_F1_T = cal_OT_F1_T(confusion_matrix_sec)
    IDcnt, IDsw, IDsw_rate = cal_IDsw(data_type, roi_result, risky_dict, behavior_dict)
    wMOTA = cal_MOTA(confusion_matrix, IDsw, IDcnt)

    if data_type != "non-interactive":
        PIC = cal_PIC(data_type, roi_result, behavior_dict, risky_dict, critical_dict)
        PIC = PIC/704.483678*100    # normalized PIC

    metric_result_raw, metric_result_str = cal_metric(
        data_type, method, confusion_matrix, OT_R, OT_P, OT_F1, OT_F1_T, IDcnt, IDsw, IDsw_rate, wMOTA, PIC, attribute)

    return metric_result_raw, metric_result_str
