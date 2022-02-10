import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, help='name of dataset mnist or cifar or cifar100', required=True)

    # user participation parameters
    parser.add_argument('--epochs',           type=int,   help="number of rounds of training", required=True)
    parser.add_argument('--frac',             type=float, help='the fraction of participating clients for each round', required=True)
    parser.add_argument('--num_users',        type=int,   help='number of users: n', required=True)
    parser.add_argument('--data_dist',        type=str,   help='IID or non-IID', required=True)
    parser.add_argument('--num_of_label_k',   type=int,   help='each client data label number, k (default: None)', default=None)
    parser.add_argument('--random_num_label', action='store_true', help='flag of client has random number of labels, otherwise constant')
    parser.add_argument('--unequal',          action='store_true', help='flag of whether to use unequal data splits for non-i.i.d setting')

    # privacy parameters
    parser.add_argument('--epsilon', type=float, default=None, help='privacy budget epsilon (default: None)')
    parser.add_argument('--delta',   type=float, default=None, help='privacy budget delta (default: None)')
    parser.add_argument('--dp',      action='store_true',  help="if set, perform dp aggregation")
    parser.add_argument('--sigma',        type=float, default=1.12,  help='the standard deviation of gaussian noise (default: 1.12)')
    parser.add_argument('--clipping',     type=float, default=1.0,   help='clipping threshold (default: 1.0)')
    
    # secure aggregation 
    parser.add_argument('--secure_agg',      action='store_true',    help='flag of secure aggregation with enclave')
    parser.add_argument('--aggregation_alg', type=str, default=None, help='oblivious aggregation algorithm (default: None)', choices=['advanced', 'nips19', 'baseline', 'non_oblivious', 'path_oram'])

    # sparsification parameter
    parser.add_argument('--alpha',        type=float, default=None,  help='sparse rate (default: None)')
    
    # model parameters
    parser.add_argument('--model',        type=str,   help='model name [mlp, cnn]', required=True)
    parser.add_argument('--num_channels', type=int,   default=1,     help="number of channels of imgs (default: 1)")
    parser.add_argument('--num_classes',  type=int,   default=10,    help="number of classes (default: 10)")
    parser.add_argument('--optimizer',    type=str,   default='sgd', help="type of optimizer sgd or adam (default: sgd)")

    # Local training parameters
    parser.add_argument('--local_ep', type=int,   default=10,   help="the number of local epochs: E (default: 10)")
    parser.add_argument('--local_bs', type=int,   default=10,   help="local batch size: B (default: 32)")
    parser.add_argument('--lr',       type=float, default=0.01, help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5,  help='SGD momentum (default: 0.5)')

    # Protection
    parser.add_argument('--protection',      type=str,   default=None, help='index-privacy, k-anonymization, cacheline (default: None)', choices=['index-privacy', 'k-anonymization', 'cacheline', ''])
    parser.add_argument('--index_privacy_r', type=float, default=None, help='the amount of random indices, r times (default: None)')

    # Attacker
    parser.add_argument('--attack',                 type=str, default=None, help='clustering, nn (default: None)' , choices=['clustering', 'nn'])
    parser.add_argument('--fixed_inference_number', type=int, default=None, help='number of inference label (default: None)')
    parser.add_argument('--single_model',           action='store_true',    help='flag of (single model with concatenated data, otherwise model divided per round')
    parser.add_argument('--attack_from_cache',      action='store_true',    help='read adversarial training data from attack, otherwise run fl and get adversarial training data')
    parser.add_argument('--attacker_batch_size',    type=int, default=None, help="batch size of attacker's training data (default: None)")
    parser.add_argument('--per_round',              action='store_true',    help="attack at for each epoch")

    # Other
    parser.add_argument('--seed',    type=int, default=0,    help='random seed (default: 0)')
    parser.add_argument('-v', '--verbose', action='store_true',    help='verbose')
    parser.add_argument('--gpu_id',  type=int, default=None, help="To use cuda, set to a specific GPU ID. Default set to use CPU. (default: None)")
    parser.add_argument('--prefix',  type=str, default=None,    help='prefix of result file')
    parser.add_argument('--local_skip',  action='store_true',    help='skip local learning')

    args = parser.parse_args()
    
    # check
    if args.data_dist == 'non-IID':
        assert args.num_of_label_k, 'num_of_label_k must be set'
        
    if args.protection == 'index-privacy':
        assert args.index_privacy_r, 'index_privacy_r must be set'

    if bool(args.secure_agg):
        assert bool(args.attack) == False, 'either secure aggregation or attack is need'
        assert args.aggregation_alg, 'aggregation_alg must be set'
    else:
        assert bool(args.attack) == True, 'either secure aggregation or attack is need'
        
    if args.attack == 'clustering':
        assert bool(args.attacker_batch_size) == False, 'attacker_batch_size must not be set'
        
    if args.dp:
        assert bool(args.epsilon) and bool(args.delta), 'epsilon and delta must be set'
        
    if args.dataset == 'mnist' or args.dataset == 'cifar10':
        assert args.num_classes == 10, 'The number of claases is 10'
    elif args.dataset == 'cifar100' or args.dataset == 'purchase100':
        assert args.num_classes == 100, 'The number of claases is 100'

    return args


def exp_details(args):
    print('\nExperimental details:')
    print(args)
    if args.secure_agg:
        print('  [Secure Aggregation Mode]')
    else:
        print('  [Non Secure Aggregation Mode]')
    print(f'    Seed      : {args.seed}')
    if args.dp:
        print(f'    DP (Epsilon, Delta) : {args.epsilon}, {args.delta}')
    else:
        print('     DP        : no')
    print(f'    Model     : {args.model}')
    print(f'    Optimizer : {args.optimizer}')
    print(f'    Learning  : {args.lr}')
    print(f'    Global Rounds   : {args.epochs}\n')

    print('    Federated parameters:')
    print(f'    {args.data_dist}')

    print(f'    Fraction of users  : {args.frac}')
    print(f'    Local Batch size   : {args.local_bs}')
    print(f'    Local Epochs       : {args.local_ep}\n')

    print('    Attacker parameters:')
    print(f'    single_model           : {args.single_model}')
    print(f'    fixed_inference_number : {args.fixed_inference_number}')
    print(f'    attack     : {args.attack}')
    print(f'    protection : {args.protection}')
    print(f'    num_of_label_k : {args.num_of_label_k}')
    print(f'    random_num_label : {args.random_num_label}')
    return


def get_aggregation_alg_code(aggregation_alg):
    if aggregation_alg == 'advanced':
        return 1
    elif aggregation_alg == 'nips19':
        return 2
    elif aggregation_alg == 'baseline':
        return 3
    elif aggregation_alg == 'non_oblivious':
        return 4
    elif aggregation_alg == 'path_oram':
        return 5
    else:
        exit('Error: unrecognized model')
