import torch
from utils.utils import *
from tqdm import tqdm
from utils.ood_utils import *
from collections import Counter
from datetime import datetime 
import matplotlib.pyplot as plt
import json

def train_build(task_list, args, train_data, model):
    start_time = datetime.now() 
    
    gt_list, feature_list = [], []
    total_loss_list, iter_list, total_iter = [], [], 0
    train_param = {}
    
    # zeroshot = Zeroshot(args.model_clip, args)
    
    # training
    for task_id in range(len(task_list)):
        train_param[task_id] = {}
        task_loss_list = []
        per_epoch_acc, per_epoch_loss = [], []
        
        t_train = train_data.make_dataset(task_id)
        train_loader = make_loader(t_train, args, train='train') 

        if hasattr(model, 'preprocess_task'):
            model.preprocess_task(names=train_data.task_list[task_id][0],
                                  labels=train_data.task_list[task_id][1],
                                  task_id=task_id,
                                  loader=train_loader)
            args.logger.print('Label', Counter(train_loader.dataset.targets))

        args.logger.print(f'[TRAINING] task: {task_id}...')

        if "pass" in args.model and task_id > 0:
            model.after_task(task_id)
        
        for epoch in range(args.n_epochs):
            iters = []
            loss_list, acc_list = [], []
            model.reset_eval()
            
            
            for b, (x, y, f_y, names, orig) in tqdm(enumerate(train_loader)):
                # if b > 50:
                #     break
                f_y = f_y[:, 1]
                x, y = x.to(args.device), y.to(args.device)
                if "more" in args.model:
                    loss = model.observe(x, y, names, x, f_y, task_id=task_id, b=b, B=len(train_loader))        # EDIT: changed from train_loaders[-1] to train_loader
                elif "derpp"  in args.model:
                    loss = model.observe(x, y, not_aug_inputs=x)
                elif "pass" in args.model:
                    loss = model.observe(x, y, task_id=task_id)
                else:
                    raise ValueError(f"{args.model} is not supported yet.")

                total_loss_list.append(loss) 
                task_loss_list.append(loss)
                loss_list.append(loss)
                cum_acc = model.correct / model.total * 100     # EDIT: added
                acc_list.append(model.correct / model.total * 100)
                iters.append(total_iter)
                total_iter += 1
                # args.logger.print(f"e:[{epoch + 1}/{args.n_epochs}], b:[{b}/{len(train_loader)}]"\
                                #    f"-- Loss: {loss}, Cum. Acc.: {cum_acc}")  

            iter_list.append(iters)
            per_epoch_acc.append(np.mean(acc_list))
            per_epoch_loss.append(np.mean(loss_list))


            # At last epoch, 
            if (epoch + 1) == args.n_epochs:
                args.logger.print(f'[TRAINING]: task {task_id}: at last epoch...')

                args.logger.print(f'Per epoch accuracy score: {per_epoch_acc}')
                args.logger.print(f'Per epoch accuracy loss: {per_epoch_loss}')

                # save activations, logits, labels, predictions and conf_scores -- to compute MD
                total_num = 0
                collect_features, collect_logits, collect_gt = [], [], []
                
                model.reset_eval()
                args.logger.print("Collecting features from training data...")
                for b, (x, y, _, _, _) in tqdm(enumerate(train_loader)):             # EDIT: changed from train_loaders[-1] to train_loader
                    # if b > 50:
                    #     break
                    x, y = x.to(args.device), y.to(args.device)
                    # normalized_labels = y % args.num_cls_per_task
                    with torch.no_grad():
                        model.net.eval()
                        if "more" in args.model:
                            features, _ = model.net.forward_features(task_id, x, s = args.smax)
                            logits = model.net.forward_classifier(task_id, features)[:, task_id * args.num_cls_per_task: (task_id+1) * args.num_cls_per_task]
                        else:
                            features = model.net.forward_features(x)
                            logits = model.net.forward_classifier(features)
                            if "pass" in args.model:
                                logits = logits[:,::4] # each group of 4 logits is dedicated to 90° rotations

                        # score, pred = torch.max(torch.softmax(logits, dim=1), dim=1)
                        
                        collect_gt.append(y.data.cpu().numpy().tolist())
                        collect_features.append(features.data.cpu().numpy())
                        collect_logits.append(logits.data.cpu().numpy()) 
                       
                model.net.train()
                gt_list = np.concatenate(collect_gt)
                feature_list = np.concatenate(collect_features)
                # logit_list = np.concatenate(collect_logits)
                # added foll MD section
                args.logger.print("Storing MD Statistics")
                cov_list = []
                ys = list(sorted(set(gt_list)))
                for y in ys:
                    idx = np.where(gt_list == y)[0]
                    f = feature_list[idx]
                    cov = np.cov(f.T)
                    cov_list.append(cov)
                    mean = np.mean(f, 0)
                    if "pass" in args.model:
                        model._protos.append(mean)
                        model._radiuses.append(np.trace(cov)/f.shape[1]) 
                        model._radius = np.sqrt(np.mean(model._radiuses))
                    args.logger.print(f'Saving means for class {y}')
                    np.save(args.logger.dir() + f'mean_label_{y}', mean)
                cov = np.array(cov_list).mean(0)
                args.logger.print(f'Saving covariance for task {task_id}')
                np.save(args.logger.dir() + f'cov_task_{task_id}', cov)
                mean_task = np.mean(feature_list, axis=0)
                args.logger.print(f'Saving mean for task {task_id}')
                np.save(args.logger.dir() + f'mean_task_{task_id}', mean_task)

                
                
                torch.save(feature_list, args.logger.dir() + f'/m{task_id}_d{task_id}_train_activations')         # EDIT: commented
                # torch.save(logit_list, args.logger.dir() + f'/m{task_id}_d{task_id}_train_logits')                # EDIT: commented 

                
                # for react detector
                react_percentile = [70, 75, 80, 85, 90, 95, 99]     # HARDCODED: change the percerntile range: [70, 75, 80, 85, 90, 95, 99] 
                for p in react_percentile:
                    react_thresh = np.percentile(feature_list, p)       # this threshold is c from eq 1 in React paper
                    args.logger.print(f'Threshold at percentile {p} over id data is {react_thresh}')
                    # np.save(args.logger.dir() + f'react_thresh_task_{task_id}_percentile_{p}', react_thresh)
                    np.save(args.logger.dir() + f'm{task_id}_react_thresh_percentile_{p}', react_thresh)
                
                
                # for dice detector
                mean_act = feature_list.mean(0)
                args.logger.print(f'DICE mean activation for task {task_id} is of shape {mean_act.shape}')
                # np.save(args.logger.dir() + f'dice_mean_activation_task{task_id}', mean_act)
                np.save(args.logger.dir() + f'm{task_id}_dice_mean_activation', mean_act)

                # for ash detector
                # rectified scaling at test time

                # for scale detector
                # scaling at the test time

                # save original weights and biases of the corresponding head for individual model
                if f'head.{task_id}.weight' in model.net.state_dict():
                    weight = model.net.state_dict()[f'head.{task_id}.weight'].detach().cpu().numpy()
                    bias = model.net.state_dict()[f'head.{task_id}.bias'].detach().cpu().numpy()
                    # np.savez(args.logger.dir() + f'fc_layer_model_{task_id}', weight=weight, bias=bias)
                    np.savez(args.logger.dir() + f'fc_layer_model_m{task_id}', weight=weight, bias=bias)
                elif f'head.weight' in model.net.state_dict():
                    weight = model.net.state_dict()[f'head.weight'].detach().cpu().numpy()
                    bias = model.net.state_dict()[f'head.bias'].detach().cpu().numpy()
                    # Take into account the task id anyway bacause the whole head change every time
                    np.savez(args.logger.dir() + f'fc_layer_model_m{task_id}', weight=weight, bias=bias)
                else:
                    args.logger.print("Head weights not found...")

                
                if (epoch + 1) == args.n_epochs:
                    args.logger.print("End task...")
                    # End task
                    if hasattr(model, 'end_task'):
                        model.end_task(task_id + 1, train_loader=train_loader)     # EDIT: changed from train_loaders[-1] to train_loader

                if hasattr(model, 'save'):
                    model.save(state_dict=model.net.state_dict(),
                                optimizer=model.optimizer,
                                task_id=task_id,
                                epoch=epoch + 1)

        # Save trained model
        torch.save(model.net.state_dict(), args.logger.dir() + f'buildv2_model_{task_id}')

        args.logger.print(f'########### Training task {task_id} ends ###########')

       
        train_param[task_id]['loss'] = per_epoch_loss
        train_param[task_id]['acc'] = per_epoch_acc
        
    with open(args.logger.dir() + "/train_param.json", "w") as out:
        json.dump(train_param, out, indent = 4)
    
    time_elapsed = datetime.now() - start_time 
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))


    
    
    
    
    # plt.plot(cum_acc_list)
    # xticks = [l[0] for l in iter_list]
    # xticks.append(iter_list[-1][-2])
    # plt.xticks(xticks)
    # plt.xlabel('Training Time')
    # plt.ylabel('Cumulative Accuracy')
    # plt.title('Cumulative Accuracy over Training Time')
    # plt.savefig(args.logger.dir() + 'cumulative_acc.png')
    # plt.close()
            

            




# TODO:
# 1. add a function for react layer [DONE]
    # 1. whats the best approach?: a) save activation and logits and load them to get the react layer
        # b. apply operation in the for loop for all individual layers and save those layes
    # 2.  also chck the placement of the fc layer

# 2. subsequent evaluation while training: add a function that is instntiated right after training each task.
# 3. [DONE]change percentile to[70, 75, 80, 85, 90, 95, 99] 

# 03/03/2025
# 4. check the train_loader (some issue with make_dataset)
# 5. someproblem with mean. collect mean task-id wise
