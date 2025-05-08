import numpy as np
from sklearn import metrics
import os
import csv
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib
import matplotlib.pyplot as plt

def compute_all_metrics(conf, label, pred):
    np.set_printoptions(precision=3)
    # recall = 0.95
    detection_thresh, auroc, aupr_in, aupr_ood, fpr = auc_and_fpr_recall(conf, label) #, recall)
    #TO DO: can I get indices of samples identified as ID from prev operation?
    # BATCH WISE: use those predicted "ID" samples from all task masks to predict the task ID, that task mask with highest number fo predicted ID samples will be the predicted task_id
    # After the task_id is predicted, perform classification using that task_id
    
    accuracy = in_distribution_acc(pred, label) 
    results = [detection_thresh, auroc, aupr_in, aupr_ood, fpr, accuracy]
    return results


def in_distribution_acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label =  label[label != -1]

    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label)
    
    return acc

def accuracy(pred, label):
    num_tp = np.sum(pred == label)
    acc = num_tp / len(label)

    return acc*100


def auc_and_fpr_recall(conf, label):
    # following convention in ML we treat OOD as positive and ID as 0
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1
    
    # in the postprocessor we assume ID samples will have larger
    # "conf" values than OOD samples
    # therefore here we need to negate the "conf" values

    # computing threshold using the gmean of specificity and sensitivity
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)        # negating conf to give ood scores larger value
    
    detection_thresh, fpr, f1 = get_gmean_threshold(fpr_list, tpr_list, thresholds, conf, ood_indicator)
    
    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(1 - ood_indicator, conf) # ID as poitive class
    precision_ood, recall_ood, thresholds_ood = metrics.precision_recall_curve(ood_indicator, -conf) # OOD as positive class

    # metrics not dependent on the threshold
    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_ood = metrics.auc(recall_ood, precision_ood)
    # auroc here is same as auc we get from compute_auc() in utils file
    return detection_thresh, auroc, aupr_in, aupr_ood, fpr

def get_auc_aupr(conf, label):
    """
    AUROC and AUPR: These metrics are threshold independent
    Useful is evaluating the distinguishability of ID from OOD just by looking at the scores
    """
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)  
    precision_in, recall_in, thresholds_in = metrics.precision_recall_curve(1 - ood_indicator, conf) # ID as poitive class
    precision_ood, recall_ood, thresholds_ood = metrics.precision_recall_curve(ood_indicator, -conf) # OOD as positive class
    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_ood = metrics.auc(recall_ood, precision_ood)

    return auroc, aupr_in, aupr_ood





def evaluate_detection_thresholds(conf, label):
    # following convention in ML we treat OOD as positive and ID as 0
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf) 

    thresh_gmean, fpr_gmean, f1_gmean = get_gmean_threshold(fpr_list, tpr_list, thresholds, conf, ood_indicator)
    thresh_95tpr, fpr_95tpr, f1_95tpr = get_95tpr_threshold(tpr_list, fpr_list, thresholds, conf, ood_indicator)
    thresh_jscore, fpr_jscore, f1_jscore = get_Jscore_threshold(fpr_list, tpr_list, thresholds, conf, ood_indicator)
    thresh_fscore, fpr_fscore, f1_fscore = get_fscore_threshold(fpr_list, tpr_list, thresholds, conf, ood_indicator)


def get_gmean_threshold(fpr_list, tpr_list, thresholds, conf, ood_indicator):
    """
    This function is used to calculate the threshold at which ID is separated from OOD. Here OOD is considered the positive class. And the threshold is calculated using Geometric Mean (GMean) of specificity and sensitivity, also supporting the imbalanced nature of the data.
    """
    gmeans = np.sqrt(tpr_list * (1 - fpr_list))
    ix = np.argmax(gmeans)
    thresh = thresholds[ix]
    fpr = fpr_list[ix]
    ood_det = (-conf > thresh).astype(np.float32)
    f1 = metrics.f1_score(ood_indicator, ood_det)
    return thresh, fpr, f1

def get_95tpr_threshold(tpr_list, fpr_list, thresholds, conf, ood_indicator):
    """
    
    """
    ix = np.argmax(tpr_list >= 0.95)
    thresh = thresholds[ix]
    fpr = fpr_list[ix]
    ood_det = (-conf > thresh).astype(np.float32)
    f1 = metrics.f1_score(ood_indicator, ood_det)
    return thresh, fpr, f1

def get_Jscore_threshold(fpr_list, tpr_list, thresholds, conf, ood_indicator):
    """
    """
    j = tpr_list - fpr_list
    ix = np.argmax(j)
    thresh = thresholds[ix]
    fpr = fpr_list[ix]
    ood_det = (-conf > thresh).astype(np.float32)
    f1 = metrics.f1_score(ood_indicator, ood_det)
    return thresh, fpr, f1

def get_fscore_threshold(fpr_list, tpr_list, thresholds, conf, ood_indicator):
    precision_ood, recall_ood, thresholds_ood = metrics.precision_recall_curve(ood_indicator, -conf, pos_label=1)
    fscore = (2 * precision_ood * recall_ood) / (precision_ood + recall_ood)
    ix = np.argmax(fscore)
    thresh = thresholds_ood[ix]
    ood_det = (-conf >= thresh).astype(np.float32)
    f1 = metrics.f1_score(ood_indicator, ood_det)
    return thresh, fpr, f1

def save_metrics(args, task_mask, metrics_list, dataset_name, ood_det, csv_path, metric_file):
    [fpr, thresh, auroc, aupr_in, aupr_out, id_acc, precision_in, precision_ood, recall_in, recall_ood, avg_prec_in, avg_prec_ood] = metrics_list

    write_content = {
        'dataset': dataset_name,
        'detector': ood_det,
        'model' : args.eval_model,
        'task': task_mask,
        'FPR@95': '{:.2f}'.format(100 * fpr),
        'AUROC': '{:.2f}'.format(100 * auroc),
        'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
        'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
        'ID_ACC': '{:.2f}'.format(100 * id_acc)
    }

    fieldnames = list(write_content.keys())

    # print ood metric results
    # args.logger.print(f'Dataset: {dataset_name}\t detector: {ood_det}')
    print(f'Dataset: {dataset_name}\t detector: {ood_det}')
    print('FPR@95: {:.2f} at threshold: {:.2f}\nAUROC: {:.2f}\nAUPR_IN: {:.2f}\nAUPR_OUT: {:.2f}\nIN_ACC: {:.2f}'.format(100 * fpr, thresh, 100 * auroc, 100 * aupr_in, 100 * aupr_out, id_acc * 100), end=' ', flush=True)
    # args.logger.print('FPR@95: {:.2f} at threshold: {:.2f}\nAUROC: {:.2f}\nAUPR_IN: {:.2f}\nAUPR_OUT: {:.2f}\nIN_ACC: {:.2f}'.format(100 * fpr, thresh, 100 * auroc, 100 * aupr_in, 100 * aupr_out, id_acc * 100), end=' ', flush=True)
    # args.logger.print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(100 * aupr_in, 100 * aupr_out), flush=True)
    # args.logger.print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
    
    # args.logger.print(u'\u2500' * 70, flush=True)
    print(u'\u2500' * 70, flush=True)


    if not os.path.exists(csv_path):
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(write_content)
    else:
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(write_content)



    np.savez(metric_file,
            ood_det = args.ood_det,
            thresh = thresh,
            fpr95 = 100 * fpr,
            auroc = 100 * auroc,
            aupr_in = 100 * aupr_in,
            aupr_out = 100 * aupr_out,
            precision_in = precision_in,
            precision_ood = precision_ood,
            recall_in = recall_in,
            recall_ood = recall_ood,
            avg_prec_in = avg_prec_in, 
            avg_prec_ood = avg_prec_ood,
            id_acc = id_acc
            )


def plot_precision_recall(precision, recall):
    display = PrecisionRecallDisplay(precision, recall)
    # _ = display.ax_.set_title("ID-OOD Precision Recall Curve")
    display.plot()
    # plt.show