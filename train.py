import os
import time
from sklearn.model_selection import LeaveOneGroupOut
import torch
from torch.utils.data import DataLoader
from Utils.mean_average_precision_str.mean_average_precision import MeanAveragePrecision2d
from numpy import argmax
import torch.nn as nn
from sklearn.utils import class_weight
import torch.nn.functional as F
from training_utils import *
from dataloader import *
from network_dm import *
from feature_extraction import *
import csv
import numpy as np


def tversky_loss(p, y, alpha=0.7, beta=0.3, eps=1e-6):
    p = p.view(-1)
    y = y.view(-1)
    tp = (p * y).sum()
    fp = (p * (1 - y)).sum()
    fn = ((1 - p) * y).sum()
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return 1.0 - tversky


def focal_loss_binary(p, y, alpha=0.75, gamma=2.0, eps=1e-6):
    """
    p: (N,) xác suất model (sigmoid output)
    y: (N,) nhãn 0/1 (float)
    alpha: trọng số cho positive class (micro-expression thường rất hiếm -> alpha > 0.5)
    gamma: focusing parameter (thường 2.0)
    """
    p = torch.clamp(p, eps, 1.0 - eps)
    # positive & negative parts
    pos_term = -alpha * (1 - p) ** gamma * y * torch.log(p)
    neg_term = -(1 - alpha) * (p ** gamma) * (1 - y) * torch.log(1 - p)
    loss = pos_term + neg_term
    return loss.mean()


def train_model(train, X_spot, Y_spot, Y1_spot, groupsLabel_spot, groupsLabel_recog, 
                final_dataset_spotting, final_subjects, final_samples, final_videos, 
                final_emotions, emotion_type, epochs, lr, batch_size, dataset_name, 
                k_p, k, ratio, frame_skip, strategy, note, method_type):
    
    # Create model directory
    if train:
        os.makedirs("save_models/%s_%semo_%s/" % (dataset_name, emotion_type-1, note), exist_ok=True)

    start = time.time()
    loso = LeaveOneGroupOut()
    subject_count = 0
    transform = None
    device = torch.device('cuda')

    # Spot
    spot_train_index = []
    spot_test_index = []
    metric_final = MeanAveragePrecision2d(num_classes=1)

    # LOSO
    for train_index, test_index in loso.split(X_spot, X_spot, groupsLabel_spot):
        spot_train_index.append(train_index)
        spot_test_index.append(test_index)

    total_gt_spot = 0

    pred_list = []
    gt_tp_list = []
    pred_window_list = []
    pred_single_list = []

    # neutral
    pred_neutral_list = []
    gt_tp_neutral_list = []

    # Training and Testing
    subjects_unique = sorted([int(s) for s in np.unique(final_subjects) if s.isdigit()])
    for subject_count in range(len(subjects_unique)): 
        
        # chuẩn bị đường dẫn lưu cho subject hiện tại
        save_dir = f"save_models/{dataset_name}_{emotion_type-1}emo_{note}"
        os.makedirs(save_dir, exist_ok=True)
        subject_id = final_subjects[subject_count]
        best_ckpt_path = os.path.join(save_dir, f"subject_{subject_id}.pkl")  # luôn trỏ vào best
        best_log_path  = os.path.join(save_dir, "best_epochs.csv")

        # reset theo dõi best cho subject này
        best_val = float("inf")
        best_epoch = -1

        # Use copy to ensure the original value is not modified
        X_spot_train, X_spot_test = [X_spot[i] for i in spot_train_index[subject_count]], \
                                     [X_spot[i] for i in spot_test_index[subject_count]]
        Y_spot_train, Y_spot_test = [Y_spot[i] for i in spot_train_index[subject_count]], \
                                     [Y_spot[i] for i in spot_test_index[subject_count]]
        Y1_spot_train, Y1_spot_test = [Y1_spot[i] for i in spot_train_index[subject_count]], \
                                       [Y1_spot[i] for i in spot_test_index[subject_count]]

        print('Subject : ' + str(subject_count+1), ', spNO.', subjects_unique[subject_count])

        # Create final dataset for training
        rem_index = downSampling(Y_spot_train, ratio)
        X_train_final = [X_spot_train[i] for i in rem_index]
        Y_train_final = [Y_spot_train[i] for i in rem_index]
        Y1_train_final = [argmax(Y1_spot_train[i], -1) for i in rem_index]

        rem_index = downSampling(Y_spot_test, ratio)
        X_val_final = [X_spot_test[i] for i in rem_index]
        Y_val_final = [Y_spot_test[i] for i in rem_index]
        Y1_val_final = [argmax(Y1_spot_test[i], -1) for i in rem_index]

        # Create final dataset for testing
        X_test_final = X_spot_test
        Y_test_final = Y_spot_test
        Y1_test_final = argmax(Y1_spot_test, -1).tolist()

        # Initialize training dataloader
        X_train_final = torch.Tensor(np.array(X_train_final))
        # Y_train_final giờ là continuous Gaussian
        Y_train_final = torch.tensor(np.array(Y_train_final), dtype=torch.float32)  # pseudo_y_train từ hàm pseudo_labeling
        Y1_weight_final = torch.Tensor(np.array(Y1_train_final)).type(torch.long)
        Y1_train_final = torch.Tensor(F.one_hot(torch.tensor(Y1_train_final)).float())
        
        train_dl = DataLoader(
            OFFSTRDataset((X_train_final[:, :][:, None, :], Y_train_final, Y1_train_final), 
                         transform=transform, train=True),
            batch_size=batch_size,
            shuffle=True,
        )
        
        # Initialize validation dataloader
        X_val_final = torch.Tensor(np.array(X_val_final).astype(float))
        Y_val_final = torch.Tensor(np.array(Y_val_final))
        Y1_val_final = torch.Tensor(F.one_hot(torch.tensor(Y1_val_final)).float())
        
        val_spot_dl = DataLoader(
            OFFSTRDataset((X_val_final[:, :][:, None, :], Y_val_final, Y1_val_final), 
                         transform=transform, train=False),
            batch_size=batch_size,
            shuffle=False,
        )
        
        # Initialize testing dataloader
        X_test_final = torch.Tensor(np.array(X_test_final))
        Y_test_final = torch.Tensor(np.array(Y_test_final))
        Y1_test_final = torch.Tensor(F.one_hot(torch.tensor(Y1_test_final)).float())
        
        test_spot_dl = DataLoader(
            OFFSTRDataset((X_test_final[:, :][:, None, :], Y_test_final, Y1_test_final), 
                         transform=transform, train=False),
            batch_size=batch_size,
            shuffle=False,
        )

        print('------Initializing Network-------')
        
        if method_type == 1:
            model = DiffME(out_channels=emotion_type).cuda()
        
        model = nn.DataParallel(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_dl)
        )

        # hệ số loss
        lambda_diff = 0.1    # hoặc 0.0 thử trước cho chắc
        lambda_spot = 1.0
        lambda_rec  = 1.0

        if train:  # Train
            loss_fn_spot = nn.MSELoss()
            loss_spot_BCE = nn.BCELoss()
            class_weights = class_weight.compute_class_weight(
                class_weight='balanced', 
                classes=np.array([i for i in range(emotion_type)]), 
                y=np.array(Y1_weight_final[Y1_weight_final != emotion_type])
            )
            class_weights[-1] = class_weights[-1] * 6
            if strategy == 0:
                class_weights[-1] = 0

            class_weights = torch.tensor(class_weights, dtype=torch.float).cuda()
            loss_fn_recog = nn.CrossEntropyLoss(weight=class_weights, reduction='mean') 
            print('Class Weights:', class_weights)

            for epoch in range(epochs):
                model.train()
                for batch in train_dl:
                    x = batch[0].to(device)  # (B,1,36,T)
                    y = batch[1].to(device)  # (B,T) spotting mask
                    y1 = batch[2].to(device) # (B,T,C) one-hot emotion

                    optimizer.zero_grad()

                    if method_type == 1:
                        # DiffME: trả về spot_prob, recog_logits, diff_loss, diff_aux
                        spot_prob, recog_logits, diff_loss, _ = model(x, y_spot_gt=y)
                    else:
                        # METST baseline: vẫn là yhat (spot), yhat1 (recog)
                        yhat, yhat1 = model(x, y_star=y)
                        spot_prob = yhat        # (B,T)
                        recog_logits = yhat1    # (B,T,C)
                        diff_loss = torch.zeros((), device=device)

                    # flatten spotting
                    spot_prob_flat = spot_prob.view(-1)   # (B*T,)
                    y_flat = y.view(-1)                  # (B*T,)

                    # flatten recognition
                    recog_logits_flat = recog_logits.view(-1, emotion_type)  # (B*T,C)
                    y1_flat = y1.view(-1, emotion_type)                       # (B*T,C)
                    # chuyển one-hot -> index cho CrossEntropyLoss
                    y1_idx_flat = y1_flat.argmax(dim=-1)                      # (B*T,)

                    # spotting loss
                    loss_spot = focal_loss_binary(spot_prob_flat, y_flat, alpha=0.75, gamma=2.0) \
                    + 0.3 * tversky_loss(spot_prob_flat, y_flat, alpha=0.4, beta=0.6)
                    # recognition loss
                    loss_recog = loss_fn_recog(recog_logits_flat, y1_idx_flat)

                    loss = lambda_diff * diff_loss \
                         + lambda_spot * (loss_spot * 0.9) \
                         + lambda_rec  * (loss_recog * 0.1)

                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # Validation
                if (epoch % 1 == 0) or (epoch == epochs - 1):
                    model.eval()
                    val_loss = 0.0
                    val_spot_loss = 0.0
                    
                    with torch.no_grad():
                        for batch in val_spot_dl:
                            x = batch[0].to(device)
                            y = batch[1].to(device)
                            y1 = batch[2].to(device)

                            if method_type == 1:
                                spot_prob, recog_logits, diff_loss, _ = model(x, y_spot_gt=y)
                            else:
                                yhat, yhat1 = model(x, y_star=y)
                                spot_prob = yhat
                                recog_logits = yhat1
                                diff_loss = torch.zeros((), device=device)
                            
                            spot_prob_flat = spot_prob.view(-1)
                            y_flat = y.view(-1)

                            recog_logits_flat = recog_logits.view(-1, emotion_type)
                            y1_flat = y1.view(-1, emotion_type)
                            y1_idx_flat = y1_flat.argmax(dim=-1)

                            l_spot = focal_loss_binary(spot_prob_flat, y_flat, alpha=0.75, gamma=2.0) \
                                + 0.3 * tversky_loss(spot_prob_flat, y_flat, alpha=0.4, beta=0.6)
                            l_recog = loss_fn_recog(recog_logits_flat, y1_idx_flat)
                            l_total = lambda_diff * diff_loss \
                                    + lambda_spot * (l_spot * 0.9) \
                                    + lambda_rec  * (l_recog * 0.1)

                            
                            val_loss += l_total.item()
                            val_spot_loss += l_spot.item()
                    
                    val_loss = val_loss / len(val_spot_dl)
                    val_spot_loss = val_spot_loss / len(val_spot_dl)
                    print(f'Epoch {epoch:3d}  Loss: {val_loss:.6f}  Loss_spot: {val_spot_loss:.6f}')

                    # lưu best checkpoint ngay khi tốt hơn
                    if val_loss < best_val:
                        best_val = val_loss
                        best_epoch = epoch
                        torch.save(model.state_dict(), best_ckpt_path)
                        print(f' New BEST epoch {epoch} (val={val_loss:.6f})')

            # ghi log best epoch (append)
            try:
                file_exists = os.path.isfile(best_log_path)
                with open(best_log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    if not file_exists:
                        writer.writerow(["subject_id", "best_epoch", "best_val_loss"])
                    writer.writerow([subject_id, best_epoch, best_val])
            except Exception as e:
                print(f"[WARN] Cannot write best log: {e}")

            # nếu vì lý do nào đó chưa lưu được (không có validation), fallback lưu cuối cùng
            if (best_epoch < 0) and (len(train_dl) > 0):
                torch.save(model.state_dict(), best_ckpt_path)
                print(f'[WARN] No validation improvement recorded. Saved last epoch to {best_ckpt_path}.')

            # ====== Testing (dùng checkpoint best đã lưu) ======
            model.eval()
            # nạp lại best để đảm bảo test đúng bản tốt nhất
            try:
                model.load_state_dict(torch.load(best_ckpt_path), strict = False)
                print(f'Loaded BEST checkpoint for testing: {best_ckpt_path}')
            except Exception as e:
                print(f'[WARN] Failed to load best checkpoint: {e} — using current weights.')

            video_num = []
            for video_index, video in enumerate(final_samples[subject_count]):
                countVideo = len([video for subject in final_samples[:subject_count] for video in subject])
                video_num.append(len(final_dataset_spotting[countVideo+video_index]))

            videocount = 0
            framecount = 0
            result_all = []
            result1_all = []
            result1_logits_all = []                                              # <-- THÊM
            result_video = np.zeros((video_num[videocount]+1)*k//2)
            result1_video = np.zeros((video_num[videocount]+1)*k//2)
            result1_logits_video = np.zeros(((video_num[videocount]+1)*k//2,    # <-- THÊM
                                              emotion_type))
            

            with torch.no_grad():
                for batch in test_spot_dl:
                    x = batch[0].to(device)

                    if method_type == 1:
                        spot_prob, recog_logits, _, _ = model(x, y_spot_gt=None)
                        yhat = spot_prob.cpu().data.numpy()
                        yhat1_logits = recog_logits
                    else:
                        yhat_tensor, yhat1_logits = model(x, y_star=None)
                        yhat = yhat_tensor.cpu().data.numpy()

                    # Lấy softmax probabilities, shape (B, T, C)              # <-- THÊM
                    yhat1_softmax = torch.softmax(yhat1_logits,                # <-- THÊM
                                                  dim=-1).cpu().data.numpy()   # <-- THÊM
                    

                    # xử lý recog strategy
                    if strategy == 2:
                        yhat1 = torch.max(yhat1_logits[:, :, 0:4], 2)[1].tolist()  
                    elif strategy == 1:
                        yhat1_logits[:, :, 4] = yhat1_logits[:, :, 4] * 0
                        yhat1 = torch.max(yhat1_logits, 2)[1].tolist()  
                    else:
                        yhat1 = torch.max(yhat1_logits, 2)[1].tolist() 

                    for i in range(len(yhat)):
                        if framecount == video_num[videocount]:
                            framecount = 0
                            videocount += 1
                            result_all.append(result_video)
                            result1_all.append(result1_video)
                            result1_logits_all.append(result1_logits_video)    # <-- THÊM
                            result_video = np.zeros((video_num[videocount]+1)*k//2)
                            result1_video = np.zeros((video_num[videocount]+1)*k//2)
                            result1_logits_video = np.zeros(               # <-- THÊM
                                ((video_num[videocount]+1)*k//2, emotion_type))
                        
                        if i == 0:
                            result_video[framecount*k//2:(framecount+2)*k//2] = yhat[i]
                            result1_video[framecount*k//2:(framecount+2)*k//2] = yhat1[i]
                            result1_logits_video[framecount*k//2:           # <-- THÊM
                                                 (framecount+2)*k//2] = yhat1_softmax[i]
                        else:
                            result_video[(framecount+1)*k//2:(framecount+2)*k//2] = yhat[i][k//2:]
                            result1_video[(framecount+1)*k//2:(framecount+2)*k//2] = yhat1[i][k//2:]
                            result1_logits_video[(framecount+1)*k//2:       # <-- THÊM
                                                  (framecount+2)*k//2] = yhat1_softmax[i][k//2:]
                        
                        framecount += 1
                    
                    if framecount == video_num[videocount] and videocount == len(video_num) - 1:
                        result_all.append(result_video)
                        result1_all.append(result1_video)
                        result1_logits_all.append(result1_logits_video)        # <-- THÊM

            save_predictions_to_csv(
                subject_id=subject_id,
                video_names=final_videos[subject_count],    # FIX chính
                result_spot=result_all,
                result_recog=result1_all,
                result_recog_logits=result1_logits_all,
                save_base_dir=save_dir,
                num_classes=emotion_type,
                frame_skip=frame_skip,                      # truyền từ ngoài vào
            )

            print('---- Spotting Results ----')
            preds, gt, total_gt_spot, metric_video, metric_final = spotting(
                final_samples, subject_count, result_all, total_gt_spot, 
                0.55, metric_final, k_p
            )
            TP_spot, FP_spot, FN_spot = sequence_evaluation(total_gt_spot, metric_final)
            
            print('---- Recognition Results ----')
            pred_list, preds_reg, gt_tp_list, pred_window_list, pred_single_list = recognition(
                result1_all, preds, metric_video, final_emotions, subject_count, 
                pred_list, gt_tp_list, final_samples, pred_window_list, 
                pred_single_list, frame_skip
            )

        else:  # Test mode (no training)
            model.load_state_dict(
                torch.load(f"weights/{dataset_name}_{emotion_type-1}emo/subject_{final_subjects[subject_count]}.pkl")
            )

            model.eval()

            video_num = []
            for video_index, video in enumerate(final_samples[subject_count]):
                countVideo = len([video for subject in final_samples[:subject_count] for video in subject])
                video_num.append(len(final_dataset_spotting[countVideo+video_index]))

            videocount = 0
            framecount = 0
            result_all = []
            result1_all = []
            result1_logits_all = []                                              # <-- THÊM
            result_video = np.zeros((video_num[videocount]+1)*k//2)
            result1_video = np.zeros((video_num[videocount]+1)*k//2)
            result1_logits_video = np.zeros(((video_num[videocount]+1)*k//2,    # <-- THÊM
                                              emotion_type))
            
            with torch.no_grad():
                for batch in test_spot_dl:
                    x = batch[0].to(device)
                    
                    if method_type == 1:
                        spot_prob, recog_logits, _, _ = model(x, y_spot_gt=None)
                        yhat = spot_prob.cpu().data.numpy()
                        yhat1_logits = recog_logits
                    else:
                        yhat_tensor, yhat1_logits = model(x, y_star=None)
                        yhat = yhat_tensor.cpu().data.numpy()
                    
                    # Lấy softmax probabilities, shape (B, T, C)              # <-- THÊM
                    yhat1_softmax = torch.softmax(yhat1_logits,                # <-- THÊM
                                                  dim=-1).cpu().data.numpy()   # <-- THÊM
                    

                    if strategy == 2:
                        yhat1 = torch.max(yhat1_logits[:, :, 0:4], 2)[1].tolist()  
                    elif strategy == 1:
                        yhat1_logits[:, :, 4] = yhat1_logits[:, :, 4] * 0
                        yhat1 = torch.max(yhat1_logits, 2)[1].tolist()  
                    else:
                        yhat1 = torch.max(yhat1_logits, 2)[1].tolist() 

                    for i in range(len(yhat)):
                        if framecount == video_num[videocount]:
                            framecount = 0
                            videocount += 1
                            result_all.append(result_video)
                            result1_all.append(result1_video)
                            result1_logits_all.append(result1_logits_video)    # <-- THÊM
                            result_video = np.zeros((video_num[videocount]+1)*k//2)
                            result1_video = np.zeros((video_num[videocount]+1)*k//2)
                            result1_logits_video = np.zeros(               # <-- THÊM
                                ((video_num[videocount]+1)*k//2, emotion_type))

                        if i == 0:
                            result_video[framecount*k//2:(framecount+2)*k//2] = yhat[i]
                            result1_video[framecount*k//2:(framecount+2)*k//2] = yhat1[i]
                            result1_logits_video[framecount*k//2:           # <-- THÊM
                                                 (framecount+2)*k//2] = yhat1_softmax[i]
                        else:
                            result_video[(framecount+1)*k//2:(framecount+2)*k//2] = yhat[i][k//2:]
                            result1_video[(framecount+1)*k//2:(framecount+2)*k//2] = yhat1[i][k//2:]
                            result1_logits_video[(framecount+1)*k//2:       # <-- THÊM
                                                  (framecount+2)*k//2] = yhat1_softmax[i][k//2:]

                        framecount += 1
                    
                    if framecount == video_num[videocount] and videocount == len(video_num) - 1:
                        result_all.append(result_video)
                        result1_all.append(result1_video)
                        result1_logits_all.append(result1_logits_video)        # <-- THÊM


            print('---- Spotting Results ----')
            preds, gt, total_gt_spot, metric_video, metric_final = spotting(
                final_samples, subject_count, result_all, total_gt_spot, 
                0.55, metric_final, k_p
            )
            TP_spot, FP_spot, FN_spot = sequence_evaluation(total_gt_spot, metric_final)
            
            print('---- Recognition Results ----')
            pred_list, preds_reg, gt_tp_list, pred_window_list, pred_single_list = recognition(
                result1_all, preds, metric_video, final_emotions, subject_count, 
                pred_list, gt_tp_list, final_samples, pred_window_list, 
                pred_single_list, frame_skip
            )

    end = time.time()
    print(f'Total time taken for training & testing: {end-start:.2f}s')

    TP_neutral = 0
    FP_neutral = 0
    for i in range(len(pred_list)):
        if pred_list[i] == emotion_type - 1:
            if gt_tp_list[i] == -1:
                FP_neutral += 1
            else:
                TP_neutral += 1
        else:
            pred_neutral_list.append(pred_list[i])
            gt_tp_neutral_list.append(gt_tp_list[i])

    return (TP_spot, FP_spot, FN_spot, metric_final, pred_list, gt_tp_list,
            TP_spot - TP_neutral, FP_spot - FP_neutral, FN_spot + TP_neutral, 
            pred_neutral_list, gt_tp_neutral_list)

def save_predictions_to_csv(subject_id, video_names, result_spot, result_recog,
                             result_recog_logits, save_base_dir, num_classes=5, frame_skip=7):
    pred_dir = os.path.join(save_base_dir, "frame_prediction_all")
    os.makedirs(pred_dir, exist_ok=True)
    csv_path = os.path.join(pred_dir, f"subject_{subject_id}_predictions.csv")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        emotion_cols = [f"Prob_Class{i}" for i in range(num_classes)]
        # Frame_Index = frame-skipped index (= int(original_frame / frame_skip))
        # Original_Frame_Approx = Frame_Index * frame_skip (sai số ≤ frame_skip-1)
        writer.writerow(['Video_Name', 'Frame_Index', 'Original_Frame_Approx',
                         'Spotting_Prob', 'Recognition_Class'] + emotion_cols)

        for v_idx, video_name in enumerate(video_names):
            spot_preds  = result_spot[v_idx]
            recog_preds = result_recog[v_idx]
            logit_preds = result_recog_logits[v_idx]

            for frame_idx in range(len(spot_preds)):
                logit_row = [f"{logit_preds[frame_idx][c]:.6f}" for c in range(num_classes)]
                writer.writerow([
                    video_name,
                    frame_idx,
                    frame_idx * frame_skip,          # ← xấp xỉ original frame, sai số ≤ 6
                    f"{spot_preds[frame_idx]:.6f}",
                    int(recog_preds[frame_idx])
                ] + logit_row)

    print(f"[INFO] Saved predictions to: {csv_path}")