import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from normalization import Normalization, RewardScaling
from replay_buffer import ReplayBuffer
from mappo_mpe import MAPPO_MPE
from routing_env_parallel import ParallelRoutingEnv
import matplotlib.pyplot as plt


def make_env(episode_limit=200, render_mode="None"):
    # 实例化您的AEC环境
    parallel_env = ParallelRoutingEnv(max_steps=episode_limit, render_mode=render_mode)
    return parallel_env


class Runner_MAPPO:
    def __init__(self, args, env_name, number, seed):
        # 初始化函数，用于初始化类的属性和创建实例
        self.args = args  # 将args参数赋值给类的args属性
        self.env_name = env_name  # 将env_name参数赋值给类的env_name属性
        self.number = number  # 将number参数赋值给类的number属性
        self.current_episode = 0  # 初始化回合数为0

        # 设置随机种子
        self.seed = seed  # 将seed参数赋值给类的seed属性
        np.random.seed(self.seed)  # 使用传入的种子设置了NumPy的随机种子
        torch.manual_seed(self.seed)  # 使用传入的种子设置了PyTorch的随机种子

        # 创建环境
        self.env = make_env(self.args.episode_limit, render_mode=args.render_mode)  # 调用make_env函数创建了一个环境，并将其赋值给类的env属性
        self.args.N = self.env.max_num_agents  # 将环境中的最大代理数量赋值给self.args.N

        # 计算N个代理的观察空间和动作空间维度
        self.args.obs_dim_n = [self.env.observation_spaces[agent].shape[0] for agent in self.env.possible_agents]
        self.args.action_dim_n = [self.env.action_spaces[agent].n for agent in self.env.possible_agents]
        self.args.obs_dim = self.args.obs_dim_n[0]  # 将第一个代理的观察空间维度赋值给self.args.obs_dim
        self.args.action_dim = self.args.action_dim_n[0]  # 将第一个代理的动作空间维度赋值给self.args.action_dim
        self.args.state_dim = np.sum(self.args.obs_dim_n)  # 计算所有代理的观察空间维度总和

        # 创建N个代理和重放缓冲区
        self.agent_n = MAPPO_MPE(self.args)  # 使用self.args参数创建了一个MAPPO_MPE实例
        self.replay_buffer = ReplayBuffer(self.args)  # 使用self.args参数创建了一个ReplayBuffer实例

        # 创建一个tensorboard记录器
        self.writer = SummaryWriter(
            log_dir='runs/MAPPO/MAPPO_env_{}_number_{}_seed_{}'.format(self.env_name, self.number, self.seed))

        self.evaluate_rewards = []  # 用于记录评估过程中的奖励
        self.episode_rewards = []  # 用于记录每个回合的奖励
        self.total_steps = 0  # 用于跟踪总训练步骤数

        # 根据args中的标志，选择性地初始化奖励规范化或奖励缩放的实例
        if self.args.use_reward_norm:
            print("------use reward norm------")
            self.reward_norm = Normalization(shape=self.args.N)  # 初始化奖励规范化实例
        elif self.args.use_reward_scaling:
            print("------use reward scaling------")
            self.reward_scaling = RewardScaling(shape=self.args.N, gamma=self.args.gamma)  # 初始化奖励缩放实例

    def run(self):
        # 当前总训练步数小于最大训练步数时，持续执行训练流程
        while self.total_steps < self.args.max_train_steps:
            # 如果当前总训练步数能整除评估频率，则对策略进行评估
            if self.total_steps % self.args.evaluate_freq == 0:
                self.evaluate_policy()  # 每隔一定步数评估一次策略

            # 执行一个训练周期，并将周期步数加到总步数中
            episode_reward, episode_steps = self.run_episode_mpe()
            self.episode_rewards.append(episode_reward)  # 记录每个回合的奖励
            self.total_steps += episode_steps
            self.current_episode += 1  # 更新回合数

            # 如果重放缓冲区中的周期数等于批次大小，则进行训练并重置缓冲区
            if self.replay_buffer.episode_num == self.args.batch_size:
                self.agent_n.train(self.replay_buffer, self.total_steps)  # 训练
                self.replay_buffer.reset_buffer()

        # 完成训练后对训练过程进行最终评估，并关闭环境
        self.evaluate_policy()
        self.env.close()

        # 绘制每个回合的奖励图
        self.plot_rewards()

    def run_episode_mpe(self, evaluate=False):
        # 初始化奖励总和
        episode_reward = 0
        # 重置环境并获取初始观察和信息
        observations, infos = self.env.reset()

        # 将观察转换为numpy数组
        obs_n = np.array([observations[agent] for agent in observations.keys()])

        # 如果使用奖励缩放，重置奖励缩放器
        if self.args.use_reward_scaling:
            self.reward_scaling.reset()

        # 如果使用RNN，重置Q网络的rnn_hidden
        if self.args.use_rnn:
            self.agent_n.actor.rnn_hidden = None
            self.agent_n.critic.rnn_hidden = None

        # 循环执行每个训练周期内的步骤，episode_step ==step; episode_limit ==max_steps
        for episode_step in range(self.args.episode_limit):
            # 选择动作并获取对应的log概率
            a_n, a_logprob_n = self.agent_n.choose_action(obs_n, evaluate=evaluate)

            # 将局部观察拼接成全局状态
            s = obs_n.flatten()

            # 获取N个代理的状态价值
            v_n = self.agent_n.get_value(s)

            # 将动作转换为字典形式
            actions = {}
            for i, agent in enumerate(self.env.agents):
                actions[agent] = a_n[i]

            # 执行动作并获取下一个观察、奖励、done标志
            obs_next_n, r_n, done_n, _, _ = self.env.step(actions)

            # 将done标志转换为数组形式
            done_n = np.array([done_n[agent] for agent in done_n.keys()])

            # 所有智能体当前回合的奖励
            agents_reward = sum(r_n.values())

            # 当前episode的累积奖励
            episode_reward += agents_reward

            # 如果不是评估过程
            if not evaluate:
                # 如果使用奖励规范化，对奖励进行规范化
                if self.args.use_reward_norm:
                    r_n = self.reward_norm(r_n)
                # 如果使用奖励缩放，对奖励进行缩放
                elif args.use_reward_scaling:
                    r_n = self.reward_scaling(r_n)

                # 存储转移数据到重放缓冲区
                self.replay_buffer.store_transition(episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n)

            # 更新观察
            obs_n = np.array([obs_next_n[agent] for agent in obs_next_n.keys()])

            # 如果所有代理都完成了当前训练周期，则跳出循环
            if all(done_n):
                break

        # 打印每个回合的累积奖励和回合数
        # print(f"Episode {self.current_episode} Reward: {episode_reward}")

        # 如果不是评估过程
        if not evaluate:
            # 存储最后一步的状态价值到重放缓冲区
            s = np.array(obs_n).flatten()
            v_n = self.agent_n.get_value(s)
            self.replay_buffer.store_last_value(episode_step + 1, v_n)

        # 返回训练周期内累积奖励和步数
        return episode_reward, episode_step + 1

    def evaluate_policy(self):
        # 初始化评估奖励
        evaluate_reward = 0
        # 执行多次评估
        for _ in range(self.args.evaluate_times):
            # 运行一个评估episode，并获取其奖励
            episode_reward, _ = self.run_episode_mpe(evaluate=True)
            # 累加奖励
            evaluate_reward += episode_reward

        # 计算平均评估奖励
        evaluate_reward = evaluate_reward / self.args.evaluate_times
        # 将评估奖励添加到列表中
        self.evaluate_rewards.append(evaluate_reward)
        # 打印总步数和评估奖励
        print("total_steps:{} \t evaluate_reward:{}".format(self.total_steps, evaluate_reward))
        # 将评估奖励写入TensorBoard
        self.writer.add_scalar('evaluate_step_rewards_{}'.format(self.env_name), evaluate_reward,
                               global_step=self.total_steps)
        # 保存奖励和模型
        # np.save('./data_train/MAPPO_env_{}_number_{}_seed_{}.npy'.format(self.env_name, self.number, self.seed),
        #         np.array(self.evaluate_rewards))
        # self.agent_n.save_model(self.env_name, self.number, self.seed, self.total_steps)

    def plot_rewards(self):
        # 绘制回合奖励图
        plt.figure(figsize=(10, 5))
        plt.plot(self.episode_rewards, label='Episode Rewards')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Rewards Over Time')
        plt.legend()
        plt.savefig('episode_rewards_plot.png')
        plt.show()


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Hyperparameters Setting for MAPPO with Custom Environment")
    parser.add_argument("--max_train_steps", type=int, default=int(10e4), help=" Maximum number of training steps")
    parser.add_argument("--episode_limit", type=int, default=200, help="Maximum number of steps per episode")
    parser.add_argument("--evaluate_freq", type=float, default=int(5000),
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=3, help="Evaluate times")

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size (the number of episodes)")
    parser.add_argument("--mini_batch_size", type=int, default=8, help="Minibatch size (the number of episodes)")
    parser.add_argument("--rnn_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the rnn")
    parser.add_argument("--mlp_hidden_dim", type=int, default=64,
                        help="The number of neurons in hidden layers of the mlp")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="GAE parameter")
    parser.add_argument("--K_epochs", type=int, default=15, help="GAE parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=True, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False,
                        help="Trick 4:reward scaling. Here, we do not use it.")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_relu", type=float, default=False, help="Whether to use relu, if False, we will use tanh")
    parser.add_argument("--use_rnn", type=bool, default=False, help="Whether to use RNN")
    parser.add_argument("--add_agent_id", type=float, default=False,
                        help="Whether to add agent_id. Here, we do not use it.")
    parser.add_argument("--use_value_clip", type=float, default=False, help="Whether to use value clip.")
    parser.add_argument('--render_mode', type=str, default='None', help='File path to my result')
    parser.add_argument("--seed", type=int, default=64, help="Random seed")

    args = parser.parse_args()

    # 环境名称和实验编号可以是固定的或者也可以通过命令行参数来设置
    env_name = "RoutingEnvAEC"
    number = 1

    # 创建Runner_MAPPO实例
    runner = Runner_MAPPO(args, env_name=env_name, number=number, seed=64)

    # 开始运行
    runner.run()
