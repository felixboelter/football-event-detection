from scipy.interpolate import pchip, interp1d, CubicSpline
from scipy.spatial.distance import euclidean
import numpy as np

def bt_remove_outliers(centers, out_thresh=100):
    pred_centers = [i for i in centers if i != None]
    if len(pred_centers) < 2:
        return centers
    last_pred, next_pred = None, pred_centers[0]
    clean_centers = []
    pred_idx = 0
    for c in centers:
        if (c != None) & (last_pred != None):
            dist = euclidean(c, last_pred)
            dist_next = euclidean(c, next_pred)
            if (dist < out_thresh) & (dist_next < out_thresh):
                clean_centers.append(c)
            else:
                clean_centers.append(None)
        else:
            clean_centers.append(c)

        if c != None:
            last_pred = c
            pred_idx += 1

        if pred_idx < len(pred_centers) - 1:
            next_pred = pred_centers[pred_idx + 1]
    
    return clean_centers

def bt_group_n_pts(centers, n_pts=4):
    pred_centers = [i for i in centers if i != None]
    if len(pred_centers) < 2:
        return None

    ovr_last_n = []
    last_n_preds = [pred_centers[0], pred_centers[1]]
    pred_idx = 2
    for c in centers:
        if c != None:
            if pred_idx < len(pred_centers):
                last_n_preds.append(pred_centers[pred_idx])
            pred_idx += 1
        if len(last_n_preds) > n_pts:
            last_n_preds = last_n_preds[-n_pts:]
        ovr_last_n.append(np.array(last_n_preds.copy()))
        
    return ovr_last_n

def bt_smooth_centers(centers, ovr_last_n, n_pts=4):
    if ovr_last_n is None:
        return centers
    extrap_centers = []
    none_cnt = 0
    for c, l_n in zip(centers, ovr_last_n):
        if c == None:
            none_cnt += 1
        else:
            if none_cnt > 0:
                if (len(l_n) == n_pts):
                    try:
                        f = interp1d(l_n[:, 0], l_n[:, 1], kind='quadratic')
                    except ValueError:
                        f = interp1d(l_n[:, 0], l_n[:, 1], kind='linear')
                    x_pts = np.linspace(l_n[:, 0][0], l_n[:, 0][1], none_cnt + 2)
                    y_pts = f(x_pts)
                    for x,y in zip(x_pts[1:-1], y_pts[1:-1]):
                        extrap_centers.append((x, y))
                else:
                    extrap_centers += [None] * none_cnt

            extrap_centers.append(c)
            none_cnt = 0
    
    extrap_centers += [None] * none_cnt
            
    return extrap_centers
        
def bt_smooth_tracking(centers, n_pts=4, outlier_threshold = 100):
    bt_centers = bt_remove_outliers(centers, out_thresh = outlier_threshold)
    bt_group_n = bt_group_n_pts(bt_centers, n_pts)
    extrap_centers = bt_smooth_centers(bt_centers, bt_group_n, n_pts)
    
    return extrap_centers


def get_cropped_frames(frames, bt_centers, config):
    frame = frames[0]
    crop_h, crop_v = config.crop_frame_size[0] // 2, config.crop_frame_size[1] // 2
    
    crop_l = [max(c[0] - crop_h, 0) for c in bt_centers]
    crop_t = [max(c[1] - crop_v, 0) for c in bt_centers]
    crop_r = [min(c[0] + crop_h, frame.shape[1]) for c in bt_centers]
    crop_b = [min(c[1] + crop_v, frame.shape[0]) for c in bt_centers]

    # if crop on boundary, extend to opposite side to maintain consistent crop size
    
    for idx, c in enumerate(bt_centers):
        if crop_t[idx] == 0:
            crop_b[idx] = config.crop_frame_size[1]
        if crop_b[idx] == frame.shape[0]:
            crop_t[idx] = frame.shape[0] - config.crop_frame_size[1]

        if crop_l[idx] == 0:
            crop_r[idx] = config.crop_frame_size[0]
        if crop_r[idx] == frame.shape[1]:
            crop_l[idx] = frame.shape[1] - config.crop_frame_size[0]

    cropped_frames = [f[int(l):int(r), int(t):int(b)] 
                     for f, l, r, t, b in zip(frames, crop_t, crop_b, crop_l, crop_r)]
    
    return cropped_frames