# -*- coding: utf-8 -*-

from run_mcgs import run_mcgs
from train import train_net
from evaluator_model import evaluate_nets
from argparse import ArgumentParser
from new_train import NeuralNetWorkWrapper
import logging

logging.basicConfig(format='%(asctime)s [%(levelname)s]: %(message)s', \
                    datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logger = logging.getLogger(__file__)

class Args:

    def __init__(self):
        self.search_node = 200

        self.iteration = 0 # 当前迭代次数
        self.total_iterations = 1 # 总迭代次数
        self.mcgs_num_processes = 1 # 多少个进程同时自博弈
        self.num_games_per_mcgs_process = 1 # 每个进程博弈多少句游戏
        self.temperature_mcgs = 1.1
        self.num_evaluator_games = 100
        self.neural_net_name = "hex_net_"
        self.batch_size = 1
        self.num_epochs = 100
        self.lr = 0.001
        self.gradient_acc_steps = 1 # 梯度累计步数
        self.max_norm = 1.0 # 允许的最大梯度范数
        self.save_interval = 100
        self.save_dir = "trained_nets/"
        self.load_dir = "trained_nets/"


if __name__ == "__main__":
    args = Args()
    net_warpper = NeuralNetWorkWrapper(args)

    parser = ArgumentParser()
    parser.add_argument("--search_node", type=int, default=args.search_node, help="每次搜索节点个数")
    parser.add_argument("--iteration", type=int, default=args.iteration, help="Current iteration number to resume from")
    parser.add_argument("--total_iterations", type=int, default=args.total_iterations, help="Total number of iterations to run")
    parser.add_argument("--MCGS_num_processes", type=int, default=args.mcgs_num_processes, help="Number of processes to run MCTS self-plays")
    parser.add_argument("--num_games_per_MCGS_process", type=int, default=args.num_games_per_mcgs_process, help="Number of games to simulate per MCTS self-play process")
    parser.add_argument("--temperature_MCTS", type=float, default=args.temperature_mcgs, help="Temperature for first 10 moves of each MCTS self-play")
    parser.add_argument("--num_evaluator_games", type=int, default=100, help="No of games to play to evaluate neural nets")
    parser.add_argument("--neural_net_name", type=str, default=args.neural_net_name, help="Name of neural net")
    parser.add_argument("--batch_size", type=int, default=args.batch_size, help="Training batch size")
    parser.add_argument("--num_epochs", type=int, default=args.num_epochs, help="No of epochs")
    parser.add_argument("--lr", type=float, default=args.lr, help="learning rate")
    parser.add_argument("--gradient_acc_steps", type=int, default=args.gradient_acc_steps, help="Number of steps of gradient accumulation")
    parser.add_argument("--max_norm", type=float, default=args.max_norm, help="Clipped gradient norm")
    args = parser.parse_args()
    
    logger.info("开启主线程...")
    for i in range(args.iteration, args.total_iterations): 
        # run_mcgs(args, start_idx=0, iteration=i)
        net_warpper.train_net(iteration=i)
        # train_net(args, iteration=i, new_optim_state=True)
        if i >= 1:
            winner = evaluate_nets(args, i, i + 1)
            counts = 0
            while (winner != (i + 1)):
                logger.info("Trained net didn't perform better, generating more MCTS games for retraining...")
                run_MCTS(args, start_idx=(counts + 1)*args.num_games_per_MCTS_process, iteration=i)
                counts += 1
                train_connectnet(args, iteration=i, new_optim_state=True)
                # 判断哪个模型好，先不用
                # winner = evaluate_nets(args, i, i + 1)
