import torch
import numpy as np
from utils.utils import *
from utils.ood_utils import get_fc_w_b, get_react_logits, dice_calculate_mask, get_scale_logits
from utils.ood_eval import accuracy
import json
# from scipy.spatial.distance import mahalanobis
import csv

    
    

def eval_InDistribution(args, detector, method):

    args.logger.print(f'\n**** Evaluating InDistribution performance on: {method.upper()} ****')
    
    
    if detector == 'base':
        scoring = ['sm_scores', 'smmd_scores', 'en_scores', 'enmd_scores', 'logit_scores', 'maha_dist']
    elif detector in ['react', 'dice', 'scale']:
        scoring = ['sm_scores', 'smmd_scores', 'en_scores', 'enmd_scores']
    else:
        NotImplementedError(f'{detector} not implemented')

    header = ['scorer', 'ACA', 'AIA', 'AF']

    
    for scorer in scoring:
        args.logger.print(f'\n      Detector: {detector}\n     Scoring Function: {scorer}')
        cil_tracker = {}
        cil_accuracy = Tracker(args)

        for eval_model in range(args.n_tasks):
            fname = os.path.join(args.load_dir + f'/m{eval_model}_test_scores_{method}.npz')
            out = np.load(fname, allow_pickle=True)
            base, react, dice, scale = out['base'].tolist(), out['react'].tolist(), out['dice'].tolist(), out['scale'].tolist()

            if detector == 'base':
                det = base
            elif detector == 'react':
                det = react
            elif detector == 'dice':
                det = dice
            elif detector == 'scale':
                det = scale
        
            for d in range(eval_model+ 1):
                scores, sm_pred, smmd_pred = [], [], []
                labels = base[d, 0]['gt']

                for t in range(eval_model+ 1):
                    sm_pred.append(det[d, t]['sm_pred'])
                    scores.append(det[d, t][scorer])

                all_scores = np.column_stack(scores)
                all_sm_pred = np.column_stack(sm_pred)

                task_id = np.argmax(all_scores, axis=1)
                pred = all_sm_pred[np.arange(all_scores.shape[0]), task_id]
                cil_correct = accuracy(pred, labels) 

                cil_accuracy.update(cil_correct, eval_model, d)   
        
        args.logger.print("######################")
        args.logger.print(f'{detector}: {scorer}: CIL result')
        cil_accuracy.print_result(t, type = 'acc')    
        cil_accuracy.print_result(t, type = 'forget')
            
        cil_tracker[scorer] = cil_accuracy

    return cil_tracker
            


def eval_nOOD_performance(args, detector, method):
    args.logger.print(f'\n**** Evaluating N-OOD performance on: {method.upper()} ****')
    # detector = args.detector
    
    if detector == 'base':
        scoring = ['sm_scores', 'smmd_scores', 'en_scores', 'enmd_scores', 'logit_scores', 'maha_dist']
    elif detector in ['react', 'dice', 'scale']:
        scoring = ['sm_scores', 'smmd_scores', 'en_scores', 'enmd_scores']
    else:
        NotImplementedError(f'{detector} not implemented')

    for scorer in scoring:
        args.logger.print(f'\n      Detector: {detector}\n     Scoring Function: {scorer}')
        auc_tracker, aupr_tracker = {}, {}
        Auc, Aupr = OWTracker(args), OWTracker(args)

        for eval_model in range(args.n_tasks-1):
            fname = os.path.join(args.load_dir + f'/m{eval_model}_test_scores_{method}.npz')
            # args.logger.print(f'Loading scores from file: {fname}')
            out = np.load(fname, allow_pickle=True)
            base, react, dice, scale = out['base'].tolist(), out['react'].tolist(), out['dice'].tolist(), out['scale'].tolist()

            if detector == 'base':
                det = base
            elif detector == 'react':
                det = react
            elif detector == 'dice':
                det = dice
            elif detector == 'scale':
                det = scale

            in_scores, out_scores, out_data_id = [], [], []

            for d in range(args.n_tasks):
                scores = []

                for t in range(eval_model + 1):
                    scores.append(det[d, t][scorer])

                all_scores = np.column_stack(scores)
                final_score = np.max(all_scores, axis = 1)

                if d<=t:
                    in_scores.append(final_score)
                else:
                    out_data_id.append(d)
                    out_scores.append(final_score)

            for d_out, d_out_scores in zip(out_data_id, out_scores):
                auc = compute_auc(in_scores, d_out_scores)
                Auc.update(auc * 100, t, d_out)

                aupr_in = compute_aupr_in(in_scores, d_out_scores)
                Aupr.update(aupr_in * 100, t, d_out)
        
        args.logger.print("######################")
        args.logger.print(f'{detector}: {scorer} scores: AUC')
        Auc.print_result(eval_model, type='auc')
        args.logger.print(f'{detector}: {scorer} scores: AUPR')
        Aupr.print_result(eval_model, type='aupr')

        auc_tracker[scorer] = Auc
        aupr_tracker[scorer] = Aupr

    return auc_tracker, aupr_tracker





def eval(args, model, train_data, test_data):
    start_time = datetime.now() 
    np.random.seed(args.seed)
    method = args.method
    detector = ['base', 'react', 'dice', 'scale']
    collect_test_scores(args, model, method, train_data, test_data)
    for det in detector:
        cil_tracker = eval_InDistribution(args, det, method)
        auc_tracker, aupr_tracker = eval_nOOD_performance(args, det, method)
        # with open(args.load_dir + f'/cil_tracker_{args.method}_{args.dataset}-{args.n_tasks}T_{det}.json', "w") as out:
        #     json.dump(cil_tracker, out, indent = 4)
        # with open(args.load_dir + f'/auc_tracker_{args.method}_{args.dataset}-{args.n_tasks}T_{det}.json', "w") as out:
        #     json.dump(auc_tracker, out, indent = 4)
        # with open(args.load_dir + f'/aupr_tracker_{args.method}_{args.dataset}-{args.n_tasks}T_{det}.json', "w") as out:
        #     json.dump(aupr_tracker, out, indent = 4)
    time_elapsed = datetime.now() - start_time 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))

    # ood_detectors = ['base', 'base_smmd', 'max_logit', 'react', 'dice', 'scale']

    # for detector in ood_detectors:
    #     torch.save(cil_tracker[detector].mat, args.load_dir() + '/cil_tracker_test')
    #     torch.save(til_tracker[detector].mat, args.load_dir() + '/til_tracker_train_test')
    #     torch.save(interim_auc_tracker[detector], args.load_dir() + '/auc_tracker_headwise_test')
    #     torch.save(interim_aupr_tracker[detector], args.load_dir() + '/aupr_tracker_headwise_test')
    #     torch.save(nood_osr_auc_tracker.mat[detector], args.load_dir() + '/nood_auc_osr_tracker')
    #     torch.save(nood_osr_aupr_in_tracker[detector].mat, args.load_dir() + '/nood_aupr_in_tracker')
    #     torch.save(nood_osr_aupr_out_tracker[detector].mat, args.load_dir() + '/nood_aupr_out_tracker')
    #     torch.save(nood_osr_auc_enmd_tracker[detector].mat, args.load_dir() + '/nood_auc_osr_enmd_tracker')
    #     torch.save(nood_osr_aupr_in_enmd_tracker[detector].mat, args.load_dir() + '/nood_aupr_in_enmd_tracker')
    #     torch.save(nood_osr_aupr_out_enmd_tracker[detector].mat, args.load_dir() + '/nood_aupr_out_enmd_tracker')
    
    
    
      