from engine import Trainer
# import argparse



# parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')

# parser.add_argument('-a', '--arch', default='resnet18')

# # lr: 0.06 for batch 512 (or 0.03 for batch 256)
# parser.add_argument('--lr', '--learning-rate', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
# parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
# parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int, help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
# parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

# parser.add_argument('--batch-size', default=512, type=int, metavar='N', help='mini-batch size')
# parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

# # moco specific configs:
# parser.add_argument('--moco-dim', default=128, type=int, help='feature dimension')
# parser.add_argument('--moco-k', default=4096, type=int, help='queue size; number of negative keys')
# parser.add_argument('--moco-m', default=0.99, type=float, help='moco momentum of updating key encoder')
# parser.add_argument('--moco-t', default=0.1, type=float, help='softmax temperature')

# parser.add_argument('--bn-splits', default=8, type=int, help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')

# parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')

# # knn monitor
# parser.add_argument('--knn-k', default=200, type=int, help='k in kNN monitor')
# parser.add_argument('--knn-t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')

# # utils
# parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
# parser.add_argument('--results-dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')

'''
args = parser.parse_args()  # running in command line
'''
# args = parser.parse_args('')  # running in ipynb

# set command line arguments here when running in ipynb
# args.epochs = 200
# args.cos = True
# args.schedule = []  # cos in use
# args.symmetric = False
# if args.results_dir == '':
#     args.results_dir = './cache-' + datetime.now().strftime("%Y-%m-%d-%H-%M-%S-moco")

# print(args)


if __name__ == '__main__':
    Trainer().run()