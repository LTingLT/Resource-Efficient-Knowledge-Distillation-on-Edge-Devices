import time
import torchvision
import torch
import os
from transform import TransformsSimCLR
from resnet_Simclr import Branch_resnet64V2,Branch_resnet256_halfV2
# from nt_xentloss import NTXentLoss,NTXent
import scheduler_wrapper
from lars import LARS
from nt_xent_dist import NT_Xent_dist
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from LinearWarmupAndCosineAnneal import LinearWarmupAndCosineAnneal 
from gather import GatherLayer
import torch.nn.functional as F



def SSLoss(zis_S, zjs_S,zis_T, zjs_T,batch_size,T): 

        concate1 = torch.cat([zis_T, zjs_T], dim=0)
        concate2 = torch.cat([zis_S, zjs_S], dim=0)
        similarity_matrix_T = F.cosine_similarity(concate1.unsqueeze(1), concate1.unsqueeze(0),dim=-1)
        similarity_matrix_S = F.cosine_similarity(concate2.unsqueeze(1), concate2.unsqueeze(0),dim=-1)
        similarity_matrix_T = similarity_matrix_T/T
        similarity_matrix_S = similarity_matrix_S/T

        softmax_similarity_matrix_T = F.softmax(similarity_matrix_T,dim=1)      

        logsoftmax_similarity_matrix_S = F.log_softmax(similarity_matrix_S,dim=1)      

        softmax_similarity_matrix_T1d = softmax_similarity_matrix_T.view(4*batch_size*batch_size,-1)

        
        logsoftmax_similarity_matrix_S1d = logsoftmax_similarity_matrix_S.view(4*batch_size*batch_size,-1)
        
        loss = F.kl_div(logsoftmax_similarity_matrix_S1d,softmax_similarity_matrix_T1d,reduction='sum')
        
        return loss

def linearWarmup_cosine_scheduler(args,optimizer, last_epoch=-1):
    total_epochs = args.epochs - args.warmup
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs, last_epoch=last_epoch)
    if args.warmup > 0:
        warmup = scheduler_wrapper.LinearWarmup(optimizer, warmup_steps=args.warmup, last_epoch=last_epoch)
        sched = scheduler_wrapper.Scheduler(sched, warmup)

    return sched




def exclude_from_wd_and_adaptation(name):
    if 'bn' in name:
            return True
    if 'bias' in name:
            return True
    
    return False

def prarmater_groups(args,model):
    param_groups = [
        {
            'params': [p for name, p in model.named_parameters() if not exclude_from_wd_and_adaptation(name)],
            'weight_decay': args.decay,
            'ignore': False,
        },
        {
            'params': [p for name, p in model.named_parameters() if exclude_from_wd_and_adaptation(name)],
            'weight_decay': 0.,
            'ignore': True,
        },
    ]
    
    return param_groups

def lars_wrapper(program_groups, lr):
    base_optimizer = torch.optim.SGD(program_groups,momentum=0.9, lr=lr)
    optimizer = LARS(base_optimizer)
    return optimizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-g', '--gpus', default=2, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of total epochs to run')
    args = parser.parse_args()
    # args.world_size = torch.cuda.device_count()
   
    args.modelname_teacher = "resnet50"
    args.modelname_student = "resnet18"
   
    args.imagesieze = 224
    args.batch_size_total = 256
    args.projection = 128
    args.loss_temp = 1
    args.loss_temp_str = "1"
    
    args.lr = 0.6
    args.decay = 1.0e-6
    args.warmup = 10
    args.warmup_ratio = 0.1
    args.dist_address = '127.0.0.1:1234'
    args.seed = 42
    args.num_workers = 16
    
    args.training = "MLCKD"
    args.dp = "dist"
    args.dataset = "IMAGENET"
    args.optimizer = "lars"
    args.warmupornot = "warmup"
    # args.transform = "transform" 
    # args.arch = "v2"   
    
    args.training_comb = "-"+args.training+"-"+args.dp+args.dataset+"-"+args.optimizer+"-"+args.warmupornot+"-"+"-"+"-temp"+args.loss_temp_str+"-"
  
    args.world_size = args.nodes*args.gpus
    args.iters = 500500
    args.finalprojection = True
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'   
    os.environ['MASTER_PORT'] = '8001'   
    mp.spawn(test_worker, nprocs=args.world_size, args=(args,))
    
    
def test_worker(gpu,args):
    print("current device:{}".format(gpu))
    if gpu == 0:
        print(args)
        
    torch.cuda.set_device(gpu)
    dist.init_process_group(
            backend='nccl',
            init_method='tcp://%s' % args.dist_address,
            world_size=args.world_size,
            rank=gpu,
        )
    
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
   

    train_dataset = torchvision.datasets.ImageFolder(
        root='.././imagenet/train',  #dataset path
        transform=TransformsSimCLR(size=args.imagesieze),
    )   
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    
    print(f'Process {dist.get_rank()}: {len(train_sampler)} training samples per epoch')
    batch_size_single=args.batch_size_total//args.world_size
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size_single,
        shuffle=(train_sampler is None),
        pin_memory=True,
        drop_last=True,
        num_workers=args.num_workers,
        sampler=train_sampler,    
        persistent_workers=True
    )
    
    

    device = torch.device("cuda", gpu)


    student_model = Branch_resnet64V2(args.modelname_student,args.projection)
    student_model = student_model.to(device)
    student_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(student_model)
    student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[gpu])
    
    student_model.train()
    
    
    teacher_model = Branch_resnet256_halfV2(args.modelname_teacher,args.projection)
    file = './runs/2024-03-13 04:44:29/resnet50-branches-distIMAGENET-lars-warmup-transform-v2-temp0_1-epoch-99.pth'  #load the trained base network checkpoint
    teacher_model.load_completenetwork(file)
    teacher_model = teacher_model.to(device)
     
    
    teacher_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(teacher_model)
    teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[gpu])    
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()

    
 
    
    program_groups = prarmater_groups(args,student_model)
    
    optimizer = lars_wrapper(program_groups, lr=args.lr)
    
    
    scheduler = LinearWarmupAndCosineAnneal(optimizer,args.warmup_ratio,args.iters,last_epoch=-1) 
    
    
    if gpu == 0:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        trainfile_path = "./runs/"+t
        os.makedirs(trainfile_path)
        logfilepath = "Trainlog"+args.training_comb+'-'+args.modelname_teacher+'-'+args.modelname_student+'-'+t+".txt"
        logfilepath = logfilepath.replace(" ", "-")
        logfilepath = logfilepath.replace(":", "-")
        logfilepath = trainfile_path+"/"+logfilepath
        print(logfilepath)

        logfile = open(logfilepath, "w", encoding='utf-8')

        logfile.write(file)

        for k,v in sorted(vars(args).items()):
            logfile.write("{} = {}".format(k,v))
            logfile.write('\n')
            
        logfile.write('\n')
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        loss_epoch = 0.0
        if gpu == 0:
            print("Training epoch:{}".format(epoch))

        for batch_idx,((xis,xjs),_) in enumerate(train_loader):
            optimizer.zero_grad()
            xis = xis.to(device)
            xjs = xjs.to(device)
            with torch.no_grad():    
                projection1_t,projection2_t,auxiliary1_t,auxiliary2_t = teacher_model(xis,xjs)####

            projection1_s,projection2_s,auxiliary1_s,auxiliary2_s = student_model(xis,xjs)####


            projection1_t =  torch.cat(GatherLayer.apply(projection1_t), dim=0)
            projection2_t =  torch.cat(GatherLayer.apply(projection2_t), dim=0)
            
            for i in range(len(auxiliary1_t)):
                auxiliary1_t[i] = torch.cat(GatherLayer.apply(auxiliary1_t[i]), dim=0)
            for i in range(len(auxiliary2_t)):
                auxiliary2_t[i] = torch.cat(GatherLayer.apply(auxiliary2_t[i]), dim=0)            
            
            projection1_s =  torch.cat(GatherLayer.apply(projection1_s), dim=0)
            projection2_s =  torch.cat(GatherLayer.apply(projection2_s), dim=0)
            for i in range(len(auxiliary1_s)):
                auxiliary1_s[i] = torch.cat(GatherLayer.apply(auxiliary1_s[i]), dim=0)
            for i in range(len(auxiliary2_s)):
                auxiliary2_s[i] = torch.cat(GatherLayer.apply(auxiliary2_s[i]), dim=0)  

            loss = 0.0
            if args.finalprojection:
                loss = SSLoss(projection1_s,projection2_s,projection1_t,projection2_t,args.batch_size_total,args.loss_temp)  

            for i in range(4):
                loss += SSLoss(auxiliary1_s[i],auxiliary2_s[i],auxiliary1_t[i],auxiliary2_t[i],args.batch_size_total,args.loss_temp)   



         
            loss.backward()

            optimizer.step()
            if gpu == 0:
                loss_epoch += loss.item()
            
            if gpu == 0:
                if (batch_idx + 1) % 100 == 0:
                    print('[TRAIN INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}'.format(epoch + 1,batch_idx + 1,loss.item()))
                    logfile.write('[TRAIN INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}'.format(epoch + 1,batch_idx + 1,loss.item()))
                    logfile.write('\n')
                    print('learning rate-{}'.format(optimizer.param_groups[0]["lr"]))
            
            scheduler.step()        
        
        if gpu == 0:
            print('[TRAIN INFO] Total_loss-Epoch-{}: Train: Loss-{:.4f}'.format(epoch + 1,loss_epoch))
            logfile.write('[TRAIN INFO] Total_loss-Epoch-{}: Train: Loss-{:.4f}'.format(epoch + 1,loss_epoch))
            logfile.write('\n')
        
            if (epoch + 1) % 10 == 0:
                ckpt = args.modelname_student+args.training_comb+"epoch-{}.pth".format(epoch)
                ckpt = trainfile_path+"/"+ckpt

                
                
                print('Saving..')
                state = {
                        'net': student_model.module.state_dict(),
                        'epoch': epoch,
                    }
                torch.save(state, ckpt)    
            print('learning rate-{}'.format(optimizer.param_groups[0]["lr"]))
        


    
    

if __name__ == "__main__":
    
    main()

    
    
    
    

