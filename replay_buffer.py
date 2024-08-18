import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args):
        # 创建一个缓冲区（字典）
        self.N = args.N  # 智能体的数量
        self.obs_dim = args.obs_dim  # 观察空间的维度
        self.state_dim = args.state_dim  # 状态空间的维度
        self.episode_limit = args.episode_limit  # 每个回合的最大步数
        self.batch_size = args.batch_size  # 批量大小
        self.episode_num = 0  # 当前存储的回合数量
        self.buffer = None  # 初始化缓冲区
        self.reset_buffer()  # 重置缓冲区

    def reset_buffer(self):
        # 初始化缓冲区，使用空数组存储每个回合的数据
        self.buffer = {'obs_n': np.empty([self.batch_size, self.episode_limit, self.N, self.obs_dim]),
                       's': np.empty([self.batch_size, self.episode_limit, self.state_dim]),
                       'v_n': np.empty([self.batch_size, self.episode_limit + 1, self.N]),
                       'a_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'a_logprob_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'r_n': np.empty([self.batch_size, self.episode_limit, self.N]),
                       'done_n': np.empty([self.batch_size, self.episode_limit, self.N])
                       }
        self.episode_num = 0  # 重置回合计数

    def store_transition(self, episode_step, obs_n, s, v_n, a_n, a_logprob_n, r_n, done_n):
        # 存储当前步的数据到缓冲区
        self.buffer['obs_n'][self.episode_num][episode_step] = obs_n  # 存储观测
        self.buffer['s'][self.episode_num][episode_step] = s  # 存储状态
        self.buffer['v_n'][self.episode_num][episode_step] = v_n  # 存储值函数
        self.buffer['a_n'][self.episode_num][episode_step] = a_n  # 存储动作
        self.buffer['a_logprob_n'][self.episode_num][episode_step] = a_logprob_n  # 存储动作的对数概率
        self.buffer['r_n'][self.episode_num][episode_step] = r_n  # 存储奖励
        self.buffer['done_n'][self.episode_num][episode_step] = done_n  # 存储是否结束标志

    def store_last_value(self, episode_step, v_n):
        # 存储最后一步的值函数
        self.buffer['v_n'][self.episode_num][episode_step] = v_n
        self.episode_num += 1  # 完成一个回合后，增加回合计数

    def get_training_data(self):
        batch = {}
        for key in self.buffer.keys():
            # 将缓冲区中的数据转换为张量格式
            if key == 'a_n':
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.long)  # 动作使用long类型
            else:
                batch[key] = torch.tensor(self.buffer[key], dtype=torch.float32)  # 其余数据使用float32类型
        return batch  # 返回训练数据批次
