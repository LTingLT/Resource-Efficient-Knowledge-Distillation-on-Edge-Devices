import time
import torchvision
import torch
import os
from transform import TransformsSimCLR
from resnet_Simclr import SimclrResnetv2,Branch_resnet256_halfV2
# from nt_xentloss import NTXentLoss,NTXent
import scheduler_wrapper
from lars import LARS
from nt_xent_dist import NT_Xent_dist
import argparse
import torch.multiprocessing as mp
import torch.distributed as dist
from LinearWarmupAndCosineAnneal import LinearWarmupAndCosineAnneal 


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
      
    args.modelname = "resnet50"
    args.imagesieze = 224
    args.batch_size_total = 256
    args.projection = 128
    args.loss_temp = 0.1
    args.loss_temp_str = "0_1"  #temperature 0.1
    
    args.lr = 0.6
    args.decay = 1.0e-6
    args.warmup = 10
    args.warmup_ratio = 0.1
    args.dist_address = '127.0.0.1:1236'
    args.seed = 42
    args.num_workers = 16
    
    args.training = "branches"
    args.dp = "dist"
    args.dataset = "IMAGENET"
    args.optimizer = "lars"
    args.warmupornot = "warmup"
    # args.transform = "transformAA" 
    # args.arch = "v2"   
    
    args.training_comb = "-"+args.training+"-"+args.dp+args.dataset+"-"+args.optimizer+"-"+args.warmupornot+"-"+"-temp"+args.loss_temp_str+"-"
    
    args.world_size = args.nodes*args.gpus
    args.iters = 500500
    
    os.environ['MASTER_ADDR'] = '127.0.0.1'   
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
        root='.././imagenet/train',
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
    
    
    
    # initialize model
    modelname = 'resnet50'
    model = Branch_resnet256_halfV2(modelname,args.projection)

    file = './checkpoint/net50-checkpoint-500400.pth.tar'
    model.load_mainnetwork2(file)
    model = model.to(device)
    for p in model.mainframe.parameters():
            p.requires_grad = False
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])  
    
      



    program_groups = prarmater_groups(args,model)
    
    optimizer = lars_wrapper(program_groups, lr=args.lr)
    
    # scheduler = linearWarmup_cosine_scheduler(args,optimizer)    
    
    scheduler = LinearWarmupAndCosineAnneal(optimizer,args.warmup_ratio,args.iters,last_epoch=-1)

    
    criterion = NT_Xent_dist(batch_size_single,args.loss_temp,args.world_size)
    criterion = criterion.to(device)    
    
    
    if gpu == 0:
        t = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        trainfile_path = "./runs/"+t
        os.makedirs(trainfile_path)
        logfilepath = "Trainlog"+args.training_comb+args.modelname+'-'+t+".txt"
        logfilepath = logfilepath.replace(" ", "-")
        logfilepath = logfilepath.replace(":", "-")
        logfilepath = trainfile_path+"/"+logfilepath
        print(logfilepath)

        logfile = open(logfilepath, "w", encoding='utf-8')

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
            _,_,auxiliary1,auxiliary2 = model(xis,xjs)####
            
            loss = 0.0
            loss += criterion(auxiliary1[0],auxiliary2[0])
            loss += criterion(auxiliary1[1],auxiliary2[1])
            loss += criterion(auxiliary1[2],auxiliary2[2])
            loss += criterion(auxiliary1[3],auxiliary2[3])
                     
            loss.backward()

            optimizer.step()
            if gpu == 0:
                loss_epoch += loss.item()
            
            if gpu == 0:
                if (batch_idx + 1) % 500 == 0:
                    print('[TRAIN INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}'.format(epoch + 1,batch_idx + 1,loss.item()))
                    logfile.write('[TRAIN INFO] Epoch-{}-Batch-{}: Train: Loss-{:.4f}'.format(epoch + 1,batch_idx + 1,loss.item()))
                    logfile.write('\n')
                    print('learning rate-{}'.format(optimizer.param_groups[0]["lr"]))
            
            scheduler.step()        
        
        if gpu == 0:
            print('[TRAIN INFO] Total_loss-Epoch-{}: Train: Loss-{:.4f}'.format(epoch + 1,loss_epoch))
            logfile.write('[TRAIN INFO] Total_loss-Epoch-{}: Train: Loss-{:.4f}'.format(epoch + 1,loss_epoch))
            logfile.write('\n')
        
            if (epoch + 1) % 50 == 0:
                ckpt = args.modelname+args.training_comb+"epoch-{}.pth".format(epoch)
                
                print('Saving..')
                state = {
                        'net': model.module.state_dict(),
                        'epoch': epoch,
                    }
                torch.save(state, ckpt)    
            print('learning rate-{}'.format(optimizer.param_groups[0]["lr"]))
            
    

if __name__ == "__main__":
    
    main()

    
    
    
    

