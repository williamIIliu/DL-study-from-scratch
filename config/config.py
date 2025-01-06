import argparse


def get_options(parser=argparse.ArgumentParser()):
    parser = argparse.ArgumentParser(description='hyper-parameters; net parameters; training options')
    # hyper-parameters
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size, default=64')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-3, help='select the learning rate, default=1e-3')

    # training option
    parser.add_argument('--seed', type=int, default=118, help="random seed")
    parser.add_argument('--workers', type=int, default=0,
                        help='number of data loading workers, you had better put it '
                             '4 times of your gpu')
    parser.add_argument('--dataset', type=str, default='JTEXT',help="dataset")
    parser.add_argument('--cuda', action='store_true', default=True, help='enables cuda')
    parser.add_argument('--model',type=str,required=False,default='GET',help="model name")
    parser.add_argument('--test',action='store_true',help="test the model")
    parser.add_argument('--retrain',action='store_true',help="retrain the model")
    parser.add_argument('--less',action='store_true',help="train using less data")

    # file path
    parser.add_argument('--output', type=str, default='../logs/',
                        help='Path to save net parameters, logs, and result')

    opt = parser.parse_args()

    if opt.output:
        print(f'The params are:')
        print(opt)

    return opt

if __name__ == '__main__':
    opt = get_options()
