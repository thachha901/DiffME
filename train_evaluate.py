from training_utils import *

def final_evaluation(TP_spot, FP_spot, FN_spot, dataset_name, pred_list, emotion_type, gt_tp_list, log_file="evaluation_log.txt"):
    # Open log file in append mode
    with open(log_file, 'a') as f:
        # Spotting
        precision = TP_spot/(TP_spot+FP_spot) if (TP_spot+FP_spot) > 0 else 0
        recall = TP_spot/(TP_spot+FN_spot) if (TP_spot+FN_spot) > 0 else 0
        F1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Create spotting results string
        spotting_results = [
            '----Spotting----',
            f'Final Result for {dataset_name}',
            f'TP: {TP_spot} FP: {FP_spot} FN: {FN_spot}',
            f'Precision = {round(precision, 4)}',
            f'Recall = {round(recall, 4)}',
            f'F1-Score = {round(F1_score, 4)}'
        ]
        
        # Print and log spotting results
        for line in spotting_results:
            print(line)
            f.write(line + '\n')

        # Recognition (TP only)
        gt_tp_spot = []
        pred_tp_spot = []
        for index in range(len(gt_tp_list)):
            if(gt_tp_list[index] != -1):
                gt_tp_spot.append(gt_tp_list[index])
                pred_tp_spot.append(pred_list[index])
        
        recognition_header = '\n----Recognition (Consider TP only)----'
        print(recognition_header)
        f.write(recognition_header + '\n')
        
        pred_str = f'Predicted    : {pred_tp_spot}'
        gt_str = f'Ground Truth : {gt_tp_spot}'
        print(pred_str)
        print(gt_str)
        f.write(pred_str + '\n')
        f.write(gt_str + '\n')
        
        f1_recog = recognition_evaluation(dataset_name, emotion_type, gt_tp_spot, pred_tp_spot, show=True)

        # Log recognition evaluation results
        f.write(f'Recognition F1-Score: {f1_recog}\n')

        # Without others
        wo_header = '\n----wo others----'
        print(wo_header)
        f.write(wo_header + '\n')
        
        gt_tp_spot_3emo = []
        pred_tp_spot_3emo = []
        for index in range(len(gt_tp_spot)):
            if(gt_tp_spot[index] != 3):
                gt_tp_spot_3emo.append(gt_tp_spot[index])
                pred_tp_spot_3emo.append(pred_tp_spot[index])
        
        pred_str_3emo = f'Predicted    : {pred_tp_spot_3emo}'
        gt_str_3emo = f'Ground Truth : {gt_tp_spot_3emo}'
        print(pred_str_3emo)
        print(gt_str_3emo)
        f.write(pred_str_3emo + '\n')
        f.write(gt_str_3emo + '\n')
        
        f1_recog_3emo = recognition_evaluation(dataset_name, emotion_type-1, gt_tp_spot_3emo, pred_tp_spot_3emo, show=True)
        f.write(f'Recognition F1-Score (wo others): {f1_recog_3emo}\n')

        # SRTS calculation
        srts = round(F1_score * f1_recog, 4)
        srts_str = f'SRTS: {srts}'
        print(srts_str)
        f.write(srts_str + '\n')
        
        # Add separator for multiple runs
        f.write('\n' + '='*50 + '\n\n')