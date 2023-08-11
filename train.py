import argparse
import torch,time,os

torch.backends.cudnn.benchmark = True

from src.utils.misc import save_checkpoint, adjust_learning_rate
import src.models as models

import datasets as datasets
from options import Options
import numpy as np

import sys
class Logger(object):
    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()




def main(args):
    args.seed = 1
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.dataset = args.dataset.lower()
    if args.dataset == 'clwd':
        dataset_func = datasets.CLWDDataset
    elif args.dataset == 'lvw':
        dataset_func = datasets.LVWDataset
    elif args.dataset == 'logo':
        dataset_func = datasets.LOGODataset
    else:
        raise ValueError("Not known dataset:\t{}".format(args.dataset))

    train_loader = torch.utils.data.DataLoader(dataset_func('train',args),batch_size=args.train_batch, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    
    val_loader = torch.utils.data.DataLoader(dataset_func('val',args),batch_size=args.test_batch, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    lr = args.lr
    data_loaders = (train_loader,val_loader)

    model = models.__dict__[args.models](datasets=data_loaders, args=args)
    print('============================ Initization Finish && Training Start =============================================')

    for epoch in range(model.args.start_epoch, model.args.epochs):
        lr = adjust_learning_rate(data_loaders, model, epoch, lr, args)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        model.record('lr',lr, epoch)        
        model.train(epoch)
        # model.validate(epoch)
        if args.freq < 0:
            model.validate(epoch)
            model.flush()
            filename = '{}_epoch_checkpoint.pth.tar'.format(epoch)
            model.save_checkpoint(filename)

if __name__ == '__main__':
    '''
        全是在打印参数，没啥好看的
    '''
    torch.backends.cudnn.benchmark = True
    parser=Options().init(argparse.ArgumentParser(description='WaterMark Removal'))
    args = parser.parse_args()
    logger = Logger(args.log)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    print('==================================== WaterMark Removal =============================================')
    print('==> {:50}: {:<}'.format("Start Time",time.ctime(time.time())))
    print('==> {:50}: {:<}'.format("USE GPU",os.environ['CUDA_VISIBLE_DEVICES']))
    print('==================================== Stable Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) == ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) == parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Changed Parameters =============================================')
    for arg in vars(args):
        if type(getattr(args, arg)) == type([]):
            if ','.join([ str(i) for i in getattr(args, arg)]) != ','.join([ str(i) for i in parser.get_default(arg)]):
                print('==> {:50}: {:<}({:<})'.format(arg,','.join([ str(i) for i in getattr(args, arg)]),','.join([ str(i) for i in parser.get_default(arg)])))
        else:
            if getattr(args, arg) != parser.get_default(arg):
                print('==> {:50}: {:<}({:<})'.format(arg,getattr(args, arg),parser.get_default(arg)))
    print('==================================== Start Init Model  ===============================================')
    main(args)
    print('==> {:50}: {:<}'.format("End Time",time.ctime(time.time())))
    print('==================================== FINISH WITHOUT ERROR =============================================')
