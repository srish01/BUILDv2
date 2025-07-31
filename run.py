import os
import sys
from itertools import chain
from torch.utils.data import DataLoader
from datasets.cont_data import *
from common import parse_args
import torch.nn as nn
import numpy as np
from utils.utils import *
from scripts.training_more import train
from scripts.training_build import train_build
from networks.net import Net
from copy import deepcopy
from datetime import datetime
import clip
import timm
import resource
import tracemalloc


if __name__ == '__main__':
    
    tracemalloc.start()  # Start tracking memory

    args = parse_args()
    args.logger = Logger(args, args.folder)
    args.logger.now()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Assign None to feature extractors
    args.model_clip, args.model_vit = None, None
    id_dataset = args.dataset

    if args.dynamic is not None:
        args.n_components = args.dynamic

    np.random.seed(args.seed)
    torch.manual_seed(0)
    args.device = device

    train_fulldata, test_fulldata = get_data(args)     
    
    # adding a condition to grab far ood data
    if args.farood:
        assert args.dataset != args.farood_data
        farood_data = get_ood_data(args)

  
    if args.task_type == 'standardCL_randomcls':
        task_list = generate_random_cl(args)
        train_data = StandardCL(train_fulldata, args, task_list)
        test_data = StandardCL(test_fulldata, args, task_list)
        
        # check if we can do args.dataset = args.farood
        if args.farood:
            assert args.dataset != args.farood_data
            args.dataset = args.farood_data
            task_list_farood = generate_random_cl(args)
            farood_data = StandardCL(farood_data, args, task_list_farood)
            args.dataset = id_dataset

    
    
    args.sup_labels = []
    for task in task_list:
        args.logger.print(task)
        for name in task[0]:
            if name not in args.sup_labels:
                args.sup_labels.append(name)

    args.logger.print('\n\n',
                        os.uname()[1] + ':' + os.getcwd(),
                        'python', ' '.join(sys.argv),
                      '\n\n')

    # number of heads after final task
    args.out_size = len(args.sup_labels)
    args.num_cls_per_task = int(args.out_size // args.n_tasks)
    args.logger.print('\n', args, '\n')

    ############## transformer; Deit or ViT ############        
    if 'adapter' in args.model:
        if 'vitadapter' in args.model:
            model_type = 'vit_base_patch16_224'
            if '_more' in args.model:
                from networks.my_vit_hat import vit_base_patch16_224 as transformer
            else:
                from networks.my_vit import vit_base_patch16_224 as transformer
        elif 'deitadapter' in args.model:
            model_type = 'deit_small_patch16_224'
            if '_more' in args.model:
                from networks.my_vit_hat import deit_small_patch16_224 as transformer
            elif 'owm_' in args.model:
                from networks.my_vit_owm import deit_small_patch16_224 as transformer
            else:
                from networks.my_vit import deit_small_patch16_224 as transformer
        
        if 'pass' in args.model:
            num_classes = args.total_cls * 4
        else:
            num_classes = args.total_cls

        if '_hat' in args.model and args.use_buffer:
            num_classes = args.num_cls_per_task + 1 # single head
        args.net = transformer(pretrained=True, num_classes=num_classes, latent=args.adapter_latent, args=args).to(device)
        
        
        if args.distillation:
            teacher = timm.create_model(model_type, pretrained=False, num_classes=num_classes).cuda()

        if 'deitadapter' in args.model:
            load_deit_pretrain(args, args.net)

        if args.model == 'vitadapter_more' or args.model == 'deitadapter_more':
            args.model_clip, args.clip_init = None, None
            from apprs.vitadapter import ViTAdapter as Model
        
        if args.model == 'derpp_deitadapter':
            args.model_clip, args.clip_init = None, None
            from apprs.derpp import Derpp as Model

    args.criterion = Criterion(args, args.net)
    model = Model(args)

    
    # model called ViTAdapeter has two parameters: criterion: crossentropy loss and net: MyVisionTransformer (stored as args.net)
    if args.distillation:
        if args.model in ['vitadapter', 'clipadapter', 'clipadapter_hat']:
            args.logger.print("Load teacher")
            model.teacher = teacher       
        if args.model in ['clipadapter', 'clipadapter_hat']:
            args.logger.print("Load teacher net")
            # model.teacher_net = teacher_net       

    model.set_seed()
    if args.framework == 'more' and args.load_dir is None:
        args.train = True
        model.training = True
        train(task_list, args, train_data, test_data, model)      
    elif args.framework == 'more' and args.train_clf:
        args.train = True
        model.training = True
        from scripts.train_clf import train
        train(task_list, args, train_data, test_data, model)      
    else:
        # if args.framework == 'build' and args.load_dir is None:
        if args.framework == 'build' and args.train:
            args.train = True
            model.training = True
            train_build(task_list, args, train_data, model)       
        elif args.framework == 'build' and args.val:
            args.train = False
            model.training = False
            model.eval()
            from hyperparam import search_hyperparam     
            search_hyperparam(args, model, train_data) 
        elif args.framework == 'build' and args.test:
            args.train = False
            model.training = False
            model.eval()
            if args.farood:
                from testing_build import test_farood
                test_farood(args, train_data, farood_data, model) 
            else:
                from scripts.eval import eval
                model.eval()
                eval(args, model, train_data, test_data)   # eval for both more and build  
     
            
    usage = resource.getrusage(resource.RUSAGE_SELF)
    max_mem = usage.ru_maxrss  # in kilobytes on most Unix
    args.logger.print(f"Max memory usage: {max_mem / 1024:.2f} MB")

    current, peak = tracemalloc.get_traced_memory()
    args.logger.print(f"Current memory usage: {current / 1024**2:.2f} MB")
    args.logger.print(f"Peak memory usage: {peak / 1024**2:.2f} MB")
    tracemalloc.stop()  # Stop tracking
    
    args.logger.now()


# TODO:
    # 1. integrate val_build_step2 in run.py