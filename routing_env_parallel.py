from pettingzoo import ParallelEnv
from collections import deque
import numpy as np
import random
from gymnasium.spaces import Box, Discrete
import networkx as nx

# 全局变量，用于跟踪episode编号
episode_counter = 0


class Packet():
    def __init__(self, source, destination):
        self.source = source
        self.destination = destination
        self.hops = 0
        self.states = []
        self.queuetime = []
        self.nodes = [source]
        self.actions = []
        self.rewards = []
        self.propagation_delay = 0
        self.transmission_delay = 0
        self.queueing_delay = 0


class ParallelRoutingEnv(ParallelEnv):
    metadata = {'render.modes': ['human'], 'is_parallelizable': True}

    def __init__(self, max_steps, render_mode=None, seed=64):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.packet_size = 1
        self.current_step = 0
        self.seed = seed

        # 标志变量，记录是否已经打印过路径
        self.path_printed = False

        # 初始化随机数生成器
        self._set_random_seed(seed)

        # 初始化图结构
        self.graph = self.create_graph()

        # 路由环境特有的初始化
        self.num_of_packets = 1e2
        self.max_bandwidth = self.num_of_packets * 10
        self.source_node = 0
        self.destination_node = 8

        self.neighbours = {node: list(self.graph.neighbors(node)) for node in self.graph.nodes()}
        self.agents = [f"agent_{node}" for node in sorted(self.graph.nodes())]
        self.possible_agents = self.agents[:]

        max_neighbors = max(len(neighbors) for neighbors in self.neighbours.values()) + 1
        obs_space_size = 6 + max_neighbors
        self.observation_spaces = {agent: Box(low=-np.inf, high=np.inf, shape=(obs_space_size,), dtype=np.float32) for
                                   agent in self.agents}
        self.action_spaces = {agent: Discrete(max_neighbors) for agent in self.agents}

        # 创建矩阵
        self.propagation_delay_matrix = self.create_propagation_delay_matrix()
        self.bandwidth_matrix = self.create_bandwidth_matrix()
        self.arrival_rate_matrix = self.create_arrival_rate_matrix()
        self.service_rate_matrix = self.create_service_rate_matrix()
        self.queue_capacity_matrix = self.create_queue_capacity_matrix()

        # 初始化能耗模型参数
        self.energy_per_packet_sent = 0.5
        self.energy_per_packet_received = 0.2
        self.cumulative_energy = {node: 0.0 for node in self.graph.nodes()}  # 每个节点的累积能耗

        # 初始化成功传输的数据包记录
        self.successfully_transmitted_packets = []

        # 设置奖励项的权重系数
        self.weight_success_reward = 10.0  # 成功传输数据包的奖励权重
        self.weight_revisit_penalty = 5.0  # 数据包重复访问节点的惩罚权重

        self.weight_delay_penalty = 3  # 延迟惩罚的权重
        self.weight_energy_penalty = 3  # 能耗惩罚的权重

        self.weight_queue_efficiency = 1.0  # 队列使用效率的权重

        # 生成初始数据包，只在 __init__ 中调用一次
        self.generate_packets()

        # 初始化环境状态
        self.reset()

    def _set_random_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def create_graph(self):
        G = nx.Graph()
        edges = [(8, 7), (7, 6), (8, 5), (7, 4), (5, 4), (6, 3), (4, 3), (5, 2), (4, 1), (2, 1), (3, 0), (1, 0)]
        G.add_edges_from(edges)
        return G

    def create_propagation_delay_matrix(self):
        size = len(self.graph.nodes())
        delay_matrix = np.zeros((size, size))
        for node in self.graph.nodes():
            for neighbor in self.graph.neighbors(node):
                delay_matrix[node][neighbor] = np.random.uniform(0, 0.5)
        return delay_matrix

    def create_bandwidth_matrix(self):
        size = len(self.graph.nodes())
        bandwidth_matrix = np.zeros((size, size))
        for node in self.graph.nodes():
            for neighbor in self.graph.neighbors(node):
                bandwidth_matrix[node][neighbor] = np.random.uniform(0, self.max_bandwidth)
        return bandwidth_matrix

    def create_arrival_rate_matrix(self):
        size = len(self.graph.nodes())
        arrival_rate_matrix = np.random.uniform(0.1, 1.0, size)
        return arrival_rate_matrix

    def create_service_rate_matrix(self):
        size = len(self.graph.nodes())
        service_rate_matrix = np.random.uniform(1.0, 2.0, size)
        return service_rate_matrix

    def create_queue_capacity_matrix(self):
        size = len(self.graph.nodes())
        queue_capacity_matrix = np.random.randint(10, 20, size)
        return queue_capacity_matrix

    def update_topology(self):
        pass

    def generate_packets(self):
        # 生成 num_of_packets 个数据包，并将其加入到源节点的队列中
        self.queues = {node: deque() for node in self.graph.nodes()}  # 初始化队列
        for _ in range(int(self.num_of_packets)):
            packet = Packet(self.source_node, self.destination_node)
            self.queues[self.source_node].append(packet)

    def reset(self, seed=None, options=None):
        global episode_counter
        episode_counter += 1  # 每次重置时，增加 episode 计数

        self.current_step = 0
        if seed is not None:
            self._set_random_seed(seed)

        self.rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}

        # 重置路径打印标志
        self.path_printed = False

        # 重置队列状态，源节点的队列包含 num_of_packets 个数据包，其他节点的队列清空
        self.queues = {node: deque() for node in self.graph.nodes()}
        self.channels = {node: {} for node in self.graph.nodes()}
        for node in self.channels:
            for neighbour in self.neighbours[node]:
                self.channels[node][neighbour] = deque()

        # 在源节点生成 num_of_packets 个数据包
        for _ in range(int(self.num_of_packets)):
            packet = Packet(self.source_node, self.destination_node)
            self.queues[self.source_node].append(packet)

        self.successfully_transmitted_packets = []
        self.cumulative_energy = {node: 0.0 for node in self.graph.nodes()}  # 重置每个节点的累积能耗

        initial_observations = {agent: self.observe(agent) for agent in self.agents}
        initial_infos = {agent: {} for agent in self.agents}

        return initial_observations, initial_infos

    def step(self, actions):
        self.current_step += 1
        observations = {}
        rewards = {agent: 0 for agent in self.agents}
        dones = {}
        infos = {agent: {} for agent in self.agents}

        for agent_id, action in actions.items():
            if self.terminations[agent_id]:
                continue
            node = int(agent_id.split('_')[1])
            packet_reached_destination = self.perform_action(node, action)
            rewards[agent_id] += self.rewards[agent_id]
            if packet_reached_destination or self.current_step >= self.max_steps:
                self.terminations[agent_id] = True
                self.truncations[agent_id] = self.current_step >= self.max_steps

        self.transfer_packets()

        for agent_id in self.agents:
            observations[agent_id] = self.observe(agent_id)
            dones[agent_id] = self.terminations[agent_id]

        if all(dones.values()) or self.current_step >= self.max_steps:
            for agent_id in self.agents:
                dones[agent_id] = True
            self.print_info()

        self.update_topology()

        return observations, rewards, dones, dones, infos

    def perform_action(self, node, action):
        packet_reached_destination = False

        if action == len(self.neighbours[node]):
            return packet_reached_destination

        if not self.queues[node]:
            return packet_reached_destination

        packets_to_send = len(self.queues[node])
        sent_packets = []

        if action < len(self.neighbours[node]):
            next_node = self.neighbours[node][action]
            for _ in range(packets_to_send):
                packet = self.queues[node].popleft()
                sent_packets.append(packet)
                packet.nodes.append(next_node)  # 记录路径中的节点
                packet.hops += 1

                # 高优先级：数据包重复访问节点的惩罚
                if packet.nodes.count(next_node) > 1:
                    self.rewards[f"agent_{node}"] -= self.weight_revisit_penalty * 20

                # 高优先级：成功传输数据包的奖励
                if next_node == packet.destination:
                    packet_reached_destination = True
                    self.rewards[f"agent_{node}"] += self.weight_success_reward * 10
                    self.successfully_transmitted_packets.append(packet)

                    # 只在第一次成功传输时打印路径
                    if not self.path_printed:
                        print(f"Packet from {packet.source} to {packet.destination} reached destination.")
                        print(f"Path: {' -> '.join(map(str, packet.nodes))}")
                        self.path_printed = True

                    continue  # 成功传输后不再处理其他目标

                # 中优先级：延迟的惩罚
                total_delay = self.calculate_total_delay(node, next_node)
                packet.propagation_delay += self.calculate_propagation_delay(node, next_node)
                packet.transmission_delay += self.calculate_transmission_delay(node, next_node, packets_to_send)
                W, _ = self.calculate_queueing_delay(next_node)
                packet.queueing_delay += W
                self.rewards[f"agent_{node}"] -= self.weight_delay_penalty * total_delay

                # 中优先级：能耗的惩罚
                send_energy = self.calculate_energy_for_sending(packets_to_send)
                self.cumulative_energy[node] += send_energy
                receive_energy = self.calculate_energy_for_receiving(len(sent_packets))
                self.cumulative_energy[next_node] += receive_energy
                self.rewards[f"agent_{node}"] -= self.weight_energy_penalty * (send_energy + receive_energy)

                # 低优先级：队列使用效率的奖励或惩罚
                if len(self.queues[next_node]) < self.queue_capacity_matrix[next_node] / 2:
                    self.rewards[f"agent_{node}"] += self.weight_queue_efficiency * 2
                if len(self.queues[next_node]) >= self.queue_capacity_matrix[next_node]:
                    self.rewards[f"agent_{node}"] -= self.weight_queue_efficiency * 3

                if len(self.queues[next_node]) < self.queue_capacity_matrix[next_node]:
                    self.channels[node][next_node].append(packet)
                else:
                    self.rewards[f"agent_{node}"] -= 5
                    self.queues[node].appendleft(packet)
        else:
            self.queues[node].extendleft(sent_packets)

        return packet_reached_destination

    def calculate_energy_for_sending(self, packets_to_send):
        return packets_to_send * self.energy_per_packet_sent

    def calculate_energy_for_receiving(self, packets_received):
        return packets_received * self.energy_per_packet_received

    def calculate_propagation_delay(self, node, next_node):
        return self.propagation_delay_matrix[node][next_node]

    def calculate_transmission_delay(self, node, next_node, packets_to_send=1):
        bandwidth = self.bandwidth_matrix[node][next_node]
        if bandwidth > 0:
            return (packets_to_send * self.packet_size) / bandwidth  # 修改为考虑数据包数量
        else:
            return float('inf')

    def calculate_queueing_delay(self, node):
        λ = self.arrival_rate_matrix[node]
        μ = self.service_rate_matrix[node]
        ρ = λ / μ
        m = self.queue_capacity_matrix[node]
        if ρ < 1:
            L = (ρ / (1 - ρ)) - ((m + 1) * ρ ** (m + 1) / (1 - ρ ** (m + 1)))
            W = L / λ
            P_block = ρ ** m * (1 - ρ) / (1 - ρ ** (m + 1))
        else:
            L = float('inf')
            W = float('inf')
            P_block = 1
        return W, P_block

    def calculate_total_delay(self, node, next_node):
        propagation_delay = self.calculate_propagation_delay(node, next_node)
        transmission_delay = self.calculate_transmission_delay(node, next_node, len(self.queues[node]))
        queueing_delay, _ = self.calculate_queueing_delay(next_node)
        total_delay = propagation_delay + transmission_delay + queueing_delay
        return total_delay

    def transfer_packets(self):
        for node in self.channels:
            for neighbour, packets in list(self.channels[node].items()):
                while packets:
                    packet = packets.popleft()
                    self.queues[neighbour].append(packet)

    def observe(self, agent_id):
        node = int(agent_id.split('_')[1])
        obs_size = self.observation_spaces[agent_id].shape[0]
        arrival_rate = self.arrival_rate_matrix[node]
        service_rate = self.service_rate_matrix[node]
        W, P_block = self.calculate_queueing_delay(node)
        if len(self.queues[node]) > 0:
            packet = self.queues[node][0]
            obs = [packet.destination, len(self.queues[node]), arrival_rate, service_rate, W, P_block] + [
                len(self.queues[n]) for n in self.neighbours[node]]
        else:
            obs = [0, len(self.queues[node]), arrival_rate, service_rate, W, P_block] + [len(self.queues[n]) for n in
                                                                                         self.neighbours[node]]
        padded_obs = np.zeros(obs_size, dtype=np.float32)
        padded_obs[:len(obs)] = obs
        return padded_obs

    def print_info(self):
        total_propagation_delay = 0
        total_transmission_delay = 0
        total_queueing_delay = 0
        packet_count = 0

        for packet in self.successfully_transmitted_packets:
            total_propagation_delay += packet.propagation_delay
            total_transmission_delay += packet.transmission_delay
            total_queueing_delay += packet.queueing_delay
            packet_count += 1

        if packet_count > 0:
            total_delay = total_propagation_delay + total_transmission_delay + total_queueing_delay

            print(f"Episode {episode_counter}:")
            print(f"Total Propagation Delay: {total_propagation_delay:.4f}")
            print(f"Total Transmission Delay: {total_transmission_delay:.4f}")
            print(f"Total Queueing Delay: {total_queueing_delay:.4f}")
            print(f"Total Delay (Propagation + Transmission + Queueing): {total_delay:.4f}")
        else:
            print("No packets processed in this episode.")

        total_energy_consumption = sum(self.cumulative_energy.values())
        print(f"Total Cumulative Energy Consumption: {total_energy_consumption:.2f}")

    def render(self, mode='human'):
        if mode == 'human':
            for node, queue in self.queues.items():
                print(f"Node {node}, Queue length: {len(queue)}")

    def close(self):
        pass
