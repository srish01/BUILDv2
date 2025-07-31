import os
import torch
import csv
import numpy as np
import torch.nn.functional as F

def get_activations(fname):
    """
    Input:
        fname: name and path of the file whose activations need to be retrieved
    Returns:
        activations: numpy array of activations of shape (rows, 384); rows = 10000 when loading activations from training data, rows = 2000 when loading activations from testing data
    """
    activations = torch.load(fname)
    return activations

def get_logits(logit_file):
    logits = torch.load(logit_file)
    return(logits)

def get_labels(label_file):
    labels = torch.load(label_file)
    return(labels)

def get_fc_w_b(args, fc_file):
    # args.logger.print(f'loading fc layer from mask {task_id}')
    # fc_file = args.load_dir + f'/fc_layer_model_m{task_id}.npz'
    fc = np.load(fc_file)
    w = torch.from_numpy(fc['weight'])
    b = torch.from_numpy(fc['bias'])
    return w, b

def my_fc(args, model, activations, task_id):
    # fc_w, fc_b = get_fc_w_b(args, fc_file)
    fc_w = model.net.state_dict()[f'head.{task_id}.weight'].detach().cpu().numpy()
    fc_b = model.net.state_dict()[f'head.{task_id}.bias'].detach().cpu().numpy()
    my_logits = torch.matmul(activations, fc_w.T) + fc_b
    return my_logits


def get_react_logits(args, model, t, test_activations, thresh):
    """
    Get the original activations, get the threshold, clip the activations using the threshold and then pass the clipped activations to the last fc layer to get new logits
    
    Input:
        activations: original test activations
        thresh: threshold for clipping derived from the training data
        task_mask: task head against which the operation is performed
    Returns:
        my_logits: logits after clipped activations are passthed through the last trained fc layer
    """

    r_activations = test_activations.clip(max = float(thresh))
    r_activations = r_activations.view(r_activations.size(0), -1)
    # args.logger.print(f'[react] fc_file: {fc_file}')
    # r_logits = my_fc(args, model, r_activations, t)
    if "more" in args.model:
        r_logits = model.net.head[t](r_activations)
    else:
        # TODO: modified here
        r_logits = model.net.head(r_activations)[:, t * args.num_cls_per_task : (t + 1) * args.num_cls_per_task]
    return r_logits, r_activations

def dice_calculate_mask(args, mean_act, w, p):
    contrib = mean_act[None, :] * w.squeeze()        # w.data.squeeze().cpu().numpy()
    contrib = np.abs(contrib)
    thresh = np.percentile(contrib, p)
    mask = torch.Tensor((contrib > thresh)).to(args.device)
    masked_w = torch.from_numpy(w).to(args.device) * mask
    return masked_w

def get_ashb_logits(args, model, test_activations, percentile, t):
    ash_act = ash_b(test_activations.view(test_activations.size(0), -1, 1, 1), percentile)
    ash_act = ash_act.view(ash_act.size(0), -1)
    # ash_logits = my_fc(args, ash_act.cpu(), fc_file)
    # ash_logits = model.net.head[t](ash_act)

    if "more" in args.model:
        ash_logits = model.net.head[t](ash_act)
    else:
        # TODO: modified here
        ash_logits = model.net.head(ash_act)[:, t * args.num_cls_per_task : (t + 1) * args.num_cls_per_task]
    return ash_logits


def get_scale_logits(args, model, test_activations, percentile, t):
    scale_act = scale(test_activations.view(test_activations.size(0), -1, 1, 1), percentile)
    scale_act = scale_act.view(scale_act.size(0), -1)
    # args.logger.print(f'[scale] fc_file: {fc_file}')
    # scale_logits = my_fc(args, model, scale_act.cpu(), t)
    # scale_logits = model.net.head[t](scale_act)
    if "more" in args.model:
        scale_logits = model.net.head[t](scale_act)
    else:
        # TODO: modified here
        scale_logits = model.net.head(scale_act)[:, t * args.num_cls_per_task : (t + 1) * args.num_cls_per_task]
    return scale_logits


def ash_b(x, percentile):
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])

    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    fill = s1 / k
    fill = fill.unsqueeze(dim=1).expand(v.shape)
    t.zero_().scatter_(dim=1, index=i, src=fill)
    
    return x


def scale(x, percentile):
    input = x.clone()
    assert x.dim() == 4
    assert 0 <= percentile <= 100
    b, c, h, w = x.shape

    # calculate the sum of the input per sample
    s1 = x.sum(dim=[1, 2, 3])
    n = x.shape[1:].numel()
    k = n - int(np.round(n * percentile / 100.0))
    t = x.view((b, c * h * w))
    v, i = torch.topk(t, k, dim=1)
    t.zero_().scatter_(dim=1, index=i, src=v)

    # calculate new sum of the input per sample after pruning
    s2 = x.sum(dim=[1, 2, 3])

    # apply sharpening
    scale = s1 / s2
    
    return input * torch.exp(scale[:, None, None, None])


# def save_metrics(args, model_task, metrics_list, dataset_name, ood_det, csv_path):
#     [fpr, auroc, aupr_in, aupr_out, id_acc] = metrics_list

#     write_content = {
#         'dataset': dataset_name,
#         'model_task': model_task,
#         'detector': ood_det,
#         'FPR@95': '{:.2f}'.format(100 * fpr),
#         'AUROC': '{:.2f}'.format(100 * auroc),
#         'AUPR_IN': '{:.2f}'.format(100 * aupr_in),
#         'AUPR_OUT': '{:.2f}'.format(100 * aupr_out),
#         'ID_ACC': '{:.2f}'.format(100 * id_acc)
#     }

#     fieldnames = list(write_content.keys())

#     # print ood metric results
#     args.logger.print(f'Dataset: {dataset_name}\t detector: {ood_det}')
#     args.logger.print('FPR@95: {:.2f}\tAUROC: {:.2f}\tAUPR_IN: {:.2f}\tAUPR_OUT: {:.2f}\tIN_ACC: {:.2f}'.format(100 * fpr, 100 * auroc, 100 * aupr_in, 100 * aupr_out, id_acc * 100), end=' ', flush=True)
#     # args.logger.print('AUPR_IN: {:.2f}, AUPR_OUT: {:.2f}'.format(100 * aupr_in, 100 * aupr_out), flush=True)
#     # args.logger.print('ACC: {:.2f}'.format(accuracy * 100), flush=True)
#     args.logger.print(u'\u2500' * 70, flush=True)


#     if not os.path.exists(csv_path):
#         with open(csv_path, 'w', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writeheader()
#             writer.writerow(write_content)
#     else:
#         with open(csv_path, 'a', newline='') as csvfile:
#             writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#             writer.writerow(write_content)

    
def get_energy_scores(logits):
    energy_scores = torch.logsumexp(logits, dim = 1)
    return energy_scores


# def get_fnames_from_trained_model(args, model_task_id):

#     if os.path.exists(args.load_dir):
#         trained_id_activation_file = args.load_dir + f'/feature_task_{model_task_id}'
#         trained_id_orig_label_file = args.load_dir + f'/label_task_{model_task_id}'
#         trained_fc_file = args.load_dir + f'/extra/fc_layer_model_{model_task_id}'
    
#         return trained_id_activation_file, trained_id_orig_label_file, trained_fc_file
#     else:
#         raise NotImplementedError(args.log_folder, "Wrong log folder")


# def get_fnames(args, model_task_id, input_task_id, distr):
    
#     if os.path.exists(args.load_dir):
#         activation_fname = args.load_dir + f'/extra/test_{distr}_features_task_{input_task_id}_model_{model_task_id}'
#         logit_fname =  args.load_dir + f'/extra/test_{distr}_logits_task_{input_task_id}_model_{model_task_id}'
#         label_fname =  args.load_dir + f'/extra/test_{distr}_labels_task_{input_task_id}_model_{model_task_id}'
#         pred_labels_fname = args.load_dir + f'/extra/test_{distr}_predlabels_task_{input_task_id}_model_{model_task_id}'
#         softmax_scores_fname = args.load_dir + f'/extra/test_{distr}_softmaxscores_task_{input_task_id}_model_{model_task_id}'
#         mdscores_fname = args.load_dir + f'/extra/test_{distr}_mdscores_task_{input_task_id}_model_{model_task_id}'

#         return activation_fname, logit_fname, label_fname, pred_labels_fname, softmax_scores_fname, mdscores_fname
    
#     else:
#         raise NotImplementedError(args.load_dir, "Wrong log folder")
    


    

    