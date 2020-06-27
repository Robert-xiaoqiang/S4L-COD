import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from .TrainHelper import AverageMeter

import math

class Evaluator:

    # evaluate only weight f-measure
    @staticmethod
    def weightedf_evaluate(preds, masks):
        assert len(preds) == len(masks), 'diff in length between prediction and map'
        maes = AverageMeter()
        ss = AverageMeter()

        iterable = list(zip(preds, masks))
        tqdm_iterable = tqdm(iterable, total=len(iterable), leave=False, desc='Evaluating')
        for pred, mask in tqdm_iterable:
            pred = np.asarray(pred)
            mask = np.asarray(mask)          
            
        results = {
            'WeightF': wfs.average() 
        }
        
        return results

    # evaluate without f-measure and s-measure
    @staticmethod
    def fast_evaluate(preds, masks):
        assert len(preds) == len(masks), 'diff in length between prediction and map'
        maes = AverageMeter()
        ss = AverageMeter()

        iterable = list(zip(preds, masks))
        tqdm_iterable = tqdm(iterable, total=len(iterable), leave=False, desc='Evaluating')
        for pred, mask in tqdm_iterable:
            pred = np.asarray(pred)
            mask = np.asarray(mask)          
            
            mae = Evaluator.cal_mae(pred, mask)
            maes.update(mae)

            s = Evaluator.cal_s(pred, mask)
            ss.update(s)
        results = {
            'MAE': maes.average(),
            'S': ss.average()
        }
        
        return results

    # list of Image on cpu
    @staticmethod
    def evaluate(preds, masks):
        assert len(preds) == len(masks), 'diff in length between prediction and map'
        pres = [AverageMeter() for _ in range(256)]
        recs = [AverageMeter() for _ in range(256)]
        maes = AverageMeter()
        maxes = AverageMeter()
        ss = AverageMeter()

        iterable = list(zip(preds, masks))
        tqdm_iterable = tqdm(iterable, total=len(iterable), leave=False, desc='Evaluating')
        for pred, mask in tqdm_iterable:
            pred = np.asarray(pred)
            mask = np.asarray(mask)
            ps, rs, mae = Evaluator.cal_pr_mae(pred, mask)
            for pidx, pdata in enumerate(zip(ps, rs)):
                p, r = pdata
                pres[pidx].update(p)
                recs[pidx].update(r)
            maes.update(mae)
            
            maxe = Evaluator.cal_maxe(pred, mask)
            maxes.update(maxe)

            s = Evaluator.cal_s(pred, mask)
            ss.update(s)
        maxf = Evaluator.cal_maxf([ pre.average() for pre in pres], [rec.average() for rec in recs ])
        results = {
            'MAE': maes.average(),
            'MAXF': maxf,
            'MAXE': maxes.average(),
            'S': ss.average()
        }
        
        return results

    sigma2 = 5
    coe1 = 1.0 / math.sqrt(2 * math.PI * sigma2)
    coe2 = -1.0 / (2 * sigma2)
    @staticmethod
    def cal_wf(prediction, gt):
        assert prediction.dtype == np.uint8
        assert gt.dtype == np.uint8
        assert prediction.shape == gt.shape
        
        if prediction.max() == prediction.min():
            prediction = prediction / 255
        else:
            prediction = ((prediction - prediction.min()) /
                        (prediction.max() - prediction.min()))
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 128] = 1

        # absoluate error
        e = np.abs(prediction - hard_gt)

        H, W = e.shape
        N = H * W
        er = e.reshape(-1)
        A = np.zeros((N, N))
        B = np.ones((N, ))
        D2 = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                iindex = np.asarray(np.unravel_index(i, (H, W)))
                jindex = np.asarray(np.unravel_index(j, (H, W)))
                diff = iindex - jindex
                d2 = np.sum(np.power(diff, 2), axis = 0)
                # d = np.sqrt(d2)
                D2[i, j] = d2
        return mae

    @staticmethod
    def cal_mae(prediction, gt):
        assert prediction.dtype == np.uint8
        assert gt.dtype == np.uint8
        assert prediction.shape == gt.shape
        
        # 确保图片和真值相同 ##################################################
        # if prediction.shape != gt.shape:
        #     prediction = Image.fromarray(prediction).convert('L')
        #     gt_temp = Image.fromarray(gt).convert('L')
        #     prediction = prediction.resize(gt_temp.size)
        #     prediction = np.array(prediction)
        
        # 获得需要的预测图和二值真值 ###########################################
        if prediction.max() == prediction.min():
            prediction = prediction / 255
        else:
            prediction = ((prediction - prediction.min()) /
                        (prediction.max() - prediction.min()))
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 128] = 1

        # MAE ##################################################################
        mae = np.mean(np.abs(prediction - hard_gt))

        return mae

    @staticmethod
    def cal_pr_mae(prediction, gt):
        assert prediction.dtype == np.uint8
        assert gt.dtype == np.uint8
        assert prediction.shape == gt.shape
        
        # 确保图片和真值相同 ##################################################
        # if prediction.shape != gt.shape:
        #     prediction = Image.fromarray(prediction).convert('L')
        #     gt_temp = Image.fromarray(gt).convert('L')
        #     prediction = prediction.resize(gt_temp.size)
        #     prediction = np.array(prediction)
        
        # 获得需要的预测图和二值真值 ###########################################
        if prediction.max() == prediction.min():
            prediction = prediction / 255
        else:
            prediction = ((prediction - prediction.min()) /
                        (prediction.max() - prediction.min()))
        hard_gt = np.zeros_like(gt)
        hard_gt[gt > 128] = 1

        # MAE ##################################################################
        mae = np.mean(np.abs(prediction - hard_gt))
        
        # MeanF ################################################################
        # threshold_fm = 2 * prediction.mean()
        # if threshold_fm > 1:
        #     threshold_fm = 1
        # binary = np.zeros_like(prediction)
        # binary[prediction >= threshold_fm] = 1
        # tp = (binary * hard_gt).sum()
        # if tp == 0:
        #     meanf = 0
        # else:
        #     pre = tp / binary.sum()
        #     rec = tp / hard_gt.sum()
        #     meanf = 1.3 * pre * rec / (0.3 * pre + rec)
        
        # PR curve #############################################################
        t = np.sum(hard_gt)
        precision, recall = [], []
        for threshold in range(256):
            threshold = threshold / 255.
            hard_prediction = np.zeros_like(prediction)
            hard_prediction[prediction >= threshold] = 1
            
            tp = np.sum(hard_prediction * hard_gt)
            p = np.sum(hard_prediction)
            if tp == 0:
                precision.append(0)
                recall.append(0)
            else:
                precision.append(tp / p)
                recall.append(tp / t)
        
        return precision, recall, mae

    # MaxF #############################################################
    @staticmethod
    def cal_maxf(ps, rs):
        assert len(ps) == 256
        assert len(rs) == 256
        maxf = []
        for p, r in zip(ps, rs):
            if p == 0 or r == 0:
                maxf.append(0)
            else:
                maxf.append(1.3 * p * r / (0.3 * p + r))
        
        return max(maxf)

    @staticmethod
    def cal_maxe(y_pred, y, num = 255):
        score = np.zeros(num)
        for i in range(num):
            fm = y_pred - y_pred.mean()
            gt = y - y.mean()
            align_matrix = 2 * gt * fm / (gt * gt + fm * fm + 1e-20)
            enhanced = ((align_matrix + 1) * (align_matrix + 1)) / 4
            score[i] = np.sum(enhanced) / (y.size - 1 + 1e-20)
        return score.max()

    @staticmethod
    def cal_s(pred, gt):
        alpha, avg_q, img_num = 0.5, 0.0, 0.0
        pred = pred.astype(np.float32) / 255.0
        gt = gt.astype(np.float32) / 255.0

        y = gt.mean()
        if y == 0:
            x = pred.mean()
            Q = 1.0 - x
        elif y == 1 or y == 255:
            x = pred.mean()
            Q = x
        else:
            Q = alpha * Evaluator._S_object(pred, gt) + (1-alpha) * Evaluator._S_region(pred, gt)
            if Q.item() < 0:
                Q = np.zeros(1)
        return Q.item()

    @staticmethod
    def _S_object(pred, gt):
        fg = np.where(gt==0, np.zeros_like(pred), pred)
        bg = np.where((gt==1) | (gt==255), np.zeros_like(pred), 1-pred)
        o_fg = Evaluator._object(fg, gt)
        o_bg = Evaluator._object(bg, 1-gt)
        u = gt.mean()
        Q = u * o_fg + (1-u) * o_bg
        return Q

    @staticmethod
    def _object(pred, gt):
        temp = pred[(gt==1) | (gt==255)]
        x = temp.mean()
        sigma_x = temp.std()
        score = 2.0 * x / (x * x + 1.0 + sigma_x + 1e-20)
        
        return score

    @staticmethod
    def _S_region(pred, gt):
        X, Y = Evaluator._centroid(gt)
        gt1, gt2, gt3, gt4, w1, w2, w3, w4 = Evaluator._divideGT(gt, X, Y)
        p1, p2, p3, p4 = Evaluator._dividePrediction(pred, X, Y)
        Q1 = Evaluator._ssim(p1, gt1)
        Q2 = Evaluator._ssim(p2, gt2)
        Q3 = Evaluator._ssim(p3, gt3)
        Q4 = Evaluator._ssim(p4, gt4)
        Q = w1*Q1 + w2*Q2 + w3*Q3 + w4*Q4
        # print(Q)
        return Q

    @staticmethod
    def _centroid(gt):
        rows, cols = gt.shape[-2:]
        gt = gt.reshape(rows, cols)
        if gt.sum() == 0:
            X = np.eye(1) * round(cols / 2)
            Y = np.eye(1) * round(rows / 2)
        else:
            total = gt.sum()
            i = np.arange(0,cols).astype(np.float32)
            j = np.arange(0,rows).astype(np.float32)
            X = np.round((gt.sum(axis=0)*i).sum() / total)
            Y = np.round((gt.sum(axis=1)*j).sum() / total)
        return X.astype(np.int64), Y.astype(np.int64)

    @staticmethod
    def _divideGT(gt, X, Y):
        h, w = gt.shape[-2:]
        area = h*w
        gt = gt.reshape(h, w)
        LT = gt[:Y, :X]
        RT = gt[:Y, X:w]
        LB = gt[Y:h, :X]
        RB = gt[Y:h, X:w]
        X = X.astype(np.float32)
        Y = Y.astype(np.float32)
        w1 = X * Y / area
        w2 = (w - X) * Y / area
        w3 = X * (h - Y) / area
        w4 = 1 - w1 - w2 - w3
        return LT, RT, LB, RB, w1, w2, w3, w4

    @staticmethod
    def _dividePrediction(pred, X, Y):
        h, w = pred.shape[-2:]
        pred = pred.reshape(h, w)
        LT = pred[:Y, :X]
        RT = pred[:Y, X:w]
        LB = pred[Y:h, :X]
        RB = pred[Y:h, X:w]
        return LT, RT, LB, RB

    @staticmethod
    def _ssim(pred, gt):
        gt = gt.astype(np.float32)
        h, w = pred.shape[-2:]
        N = h*w
        x = pred.mean()
        y = gt.mean()
        sigma_x2 = ((pred - x)*(pred - x)).sum() / (N - 1 + 1e-20)
        sigma_y2 = ((gt - y)*(gt - y)).sum() / (N - 1 + 1e-20)
        sigma_xy = ((pred - x)*(gt - y)).sum() / (N - 1 + 1e-20)
        
        aplha = 4 * x * y *sigma_xy
        beta = (x*x + y*y) * (sigma_x2 + sigma_y2)

        if aplha != 0:
            Q = aplha / (beta + 1e-20)
        elif aplha == 0 and beta == 0:
            Q = 1.0
        else:
            Q = 0
        return Q

class FullModelForTest(nn.Module):
    """
    Distribute the loss on multi-gpu to reduce 
    the memory cost in the main gpu.
    You can check the following discussion.
    https://discuss.pytorch.org/t/dataparallel-imbalanced-memory-usage/22551/21
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss = None

    def forward(self, inputs):
        outputs = self.model(inputs)
        # will be concatenated along batch axis
        return outputs