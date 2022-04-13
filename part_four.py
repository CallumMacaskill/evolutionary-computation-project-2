from spinup.utils.run_utils import ExperimentGrid
from spinup import ppo_pytorch
import torch
import random
import pandas

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()
    seed = random.randint(0, 1000)
    eg = ExperimentGrid(name='ppo-pyt-bench')
    eg.add('env_name', 'CartPole-v1', '', True)
    eg.add('max_ep_len', 500)
    eg.add('seed', [seed])
    eg.add('epochs', 100)
    eg.add('gamma', 0.5)
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(64,64)], 'hid')
    eg.add('ac_kwargs:activation', [torch.nn.Tanh], '')
    eg.run(ppo_pytorch, num_cpu=args.cpu)

    output_file = 'spinningup/data/ppo-pyt-bench_cartpole-v1/ppo-pyt-bench_cartpole-v1_s' + str(seed) + '/progress.txt'
    output_results = pandas.read_csv(output_file, sep='\s+')
    performance = output_results['AverageEpRet'].mean()
    
    print('Average Learning Performance: ' + str(performance))