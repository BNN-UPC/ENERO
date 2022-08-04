import numpy as np
import gym
import gc
import os
import gym_graph
import random
import criticPPO as critic
import actorPPOmiddR as actor
import tensorflow as tf
from collections import deque
#import time as tt
import argparse
import pickle
import heapq
from keras import backend as K

# Use BtAsiaPac, EliBackbone and Goodnet for training

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# In this experiment we learn how to pick the best action(middlepoint) by marking for each middlepoint
# the action in the topology edges. Rewards are given per time-step.
# We also remove the SP paths that can create a loop with the source node!

ENV_NAME = 'GraphEnv-v16'

# Indicates how many time-steps has an episode
EPISODE_LENGTH = 100 # We are not using it now
SEED = 9
MINI_BATCH_SIZE = 55
experiment_letter = "_B_NEW"
take_critic_demands = True # True if we want to take the demands from the most critical links, True if we want to take the largest
percentage_demands = 15 # Percentage of demands that will be used in the optimization
str_perctg_demands = str(percentage_demands)
percentage_demands /= 100

EVALUATION_EPISODES = 20 # As the demand selection is deterministic, it doesn't make sense to evaluate multiple times over the same TM
PPO_EPOCHS = 8
num_samples_top1 = int(np.ceil(percentage_demands*380))#*5
num_samples_top2 = int(np.ceil(percentage_demands*506))#*4
num_samples_top3 = int(np.ceil(percentage_demands*272))#*6

BUFF_SIZE = num_samples_top1+num_samples_top2+num_samples_top3 # Experience buffer size. Careful to don't have more samples from one TM!

# The DECAY_STEPS must be a multiple of args.e (episode_iters)
DECAY_STEPS = 60 # The second value is to indicate every how many PPO EPISODES we decay the lr
DECAY_RATE = 0.96

CRITIC_DISCOUNT = 0.8

# if agent struggles to explore the environment, increase BETA
# if the agent instead is very random in its actions, not allowing it to take good decisions, you should lower it
ENTROPY_BETA = 0.01
ENTROPY_STEP = 60

clipping_val = 0.1
gamma = 0.99
lmbda = 0.95

max_grad_norm = 0.5

differentiation_str = "Enero_3top_"+str_perctg_demands+experiment_letter
checkpoint_dir = "./models"+differentiation_str

os.environ['PYTHONHASHSEED']=str(SEED)
np.random.seed(SEED)
random.seed(SEED)

tf.random.set_seed(1)

#train_dir = "./TensorBoard/"+differentiation_str
#summary_writer = tf.summary.create_file_writer(train_dir)
global_step = 0
NUM_ACTIONS = 100 # For now we have dynamic action space. This means that we consider all nodes as actions but removing the repeated paths

hidden_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_actor = tf.keras.initializers.Orthogonal(gain=np.sqrt(0.01), seed=SEED)
hidden_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(2), seed=SEED)
kernel_init_critic = tf.keras.initializers.Orthogonal(gain=np.sqrt(1), seed=SEED)

hparams = {
    'l2': 0.0001,
    'link_state_dim': 20,
    'readout_units': 20,
    'learning_rate': 0.0002,
    'T': 5,
}

def old_cummax(alist, extractor):
    with tf.name_scope('cummax'):
        maxes = [tf.reduce_max(extractor(v)) + 1 for v in alist]
        cummaxes = [tf.zeros_like(maxes[0])]
        for i in range(len(maxes) - 1):
            cummaxes.append(tf.math.add_n(maxes[0:i + 1]))
    return cummaxes

def decayed_learning_rate(step):
    lr = hparams['learning_rate']*(DECAY_RATE ** (step / DECAY_STEPS))
    if lr<10e-5:
        lr = 10e-5
    return lr

class PPOActorCritic:
    def __init__(self):
        self.memory = deque(maxlen=BUFF_SIZE)
        self.inds = np.arange(BUFF_SIZE)
        self.listQValues = None
        self.softMaxQValues = None
        self.global_step = global_step

        self.action = None
        self.softMaxQValues = None
        self.listQValues = None

        self.utilization_feature = None
        self.bw_allocated_feature = None

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=hparams['learning_rate'], beta_1=0.9, epsilon=1e-05)
        self.actor = actor.myModel(hparams, hidden_init_actor, kernel_init_actor)
        self.actor.build()

        self.critic = critic.myModel(hparams, hidden_init_critic, kernel_init_critic)
        self.critic.build()
    
    def pred_action_distrib_sp(self, env, source, destination):
        # List of graph features that are used in the cummax() call
        list_k_features = list()

        # We get the K-middlepoints between source-destination
        middlePointList = env.src_dst_k_middlepoints[str(source) +':'+ str(destination)]
        itMidd = 0
        
        # 2. Allocate (S,D, linkDemand) demand using the K shortest paths
        while itMidd < len(middlePointList):
            env.mark_action_sp(source, middlePointList[itMidd], source, destination)
            # If we allocated to a middlepoint that is not the final destination
            if middlePointList[itMidd]!=destination:
                env.mark_action_sp(middlePointList[itMidd], destination, source, destination)

            features = self.get_graph_features(env, source, destination)
            list_k_features.append(features)

            # We desmark the bw_allocated
            env.edge_state[:,2] = 0
            itMidd = itMidd + 1

        vs = [v for v in list_k_features]

        # We compute the graphs_ids to later perform the unsorted_segment_sum for each graph and obtain the 
        # link hidden states for each graph.
        graph_ids = [tf.fill([tf.shape(vs[it]['link_state'])[0]], it) for it in range(len(list_k_features))]
        first_offset = old_cummax(vs, lambda v: v['first'])
        second_offset = old_cummax(vs, lambda v: v['second'])

        tensor = ({
            'graph_id': tf.concat([v for v in graph_ids], axis=0),
            'link_state': tf.concat([v['link_state'] for v in vs], axis=0),
            'first': tf.concat([v['first'] + m for v, m in zip(vs, first_offset)], axis=0),
            'second': tf.concat([v['second'] + m for v, m in zip(vs, second_offset)], axis=0),
            'num_edges': tf.math.add_n([v['num_edges'] for v in vs]),
            }
        )        

        # Predict qvalues for all graphs within tensors
        r = self.actor(tensor['link_state'], tensor['graph_id'], tensor['first'], tensor['second'], 
            tensor['num_edges'], training=False)
        self.listQValues = tf.reshape(r, (1, len(r)))
        self.softMaxQValues = tf.nn.softmax(self.listQValues)
        
        # Return action distribution
        return self.softMaxQValues.numpy()[0], tensor
    
    def get_graph_features(self, env, source, destination):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        self.bw_allocated_feature = env.edge_state[:,2]
        self.utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'bw_allocated': tf.convert_to_tensor(value=self.bw_allocated_feature, dtype=tf.float32),
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['bw_allocated'] = tf.reshape(sample['bw_allocated'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['utilization'], sample['capacity'], sample['bw_allocated']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 3]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state': link_state, 'first': sample['first'][0:sample['length']],
                'second': sample['second'][0:sample['length']], 'num_edges': sample['num_edges']}

        return inputs

    def critic_get_graph_features(self, env):
        """
        We iterate over the converted graph nodes and take the features. The capacity and bw allocated features
        are normalized on the fly.
        """
        self.utilization_feature = env.edge_state[:,0]

        sample = {
            'num_edges': env.numEdges,  
            'length': env.firstTrueSize,
            'capacity': env.link_capacity_feature,
            'utilization': tf.convert_to_tensor(value=np.divide(self.utilization_feature, env.edge_state[:,1]), dtype=tf.float32),
            'first': env.first,
            'second': env.second
        }

        sample['utilization'] = tf.reshape(sample['utilization'][0:sample['num_edges']], [sample['num_edges'], 1])
        sample['capacity'] = tf.reshape(sample['capacity'][0:sample['num_edges']], [sample['num_edges'], 1])

        hiddenStates = tf.concat([sample['utilization'], sample['capacity']], axis=1)
        paddings = tf.constant([[0, 0], [0, hparams['link_state_dim'] - 2]])
        link_state = tf.pad(tensor=hiddenStates, paddings=paddings, mode="CONSTANT")

        inputs = {'link_state_critic': link_state, 'first_critic': sample['first'][0:sample['length']],
                'second_critic': sample['second'][0:sample['length']], 'num_edges_critic': sample['num_edges']}

        return inputs
    
    def _write_tf_summary(self, actor_loss, critic_loss, final_entropy):
        with summary_writer.as_default():
            tf.summary.scalar(name="actor_loss", data=actor_loss, step=self.global_step)
            tf.summary.scalar(name="critic_loss", data=critic_loss, step=self.global_step)  
            tf.summary.scalar(name="entropy", data=-final_entropy, step=self.global_step)                      

            tf.summary.histogram(name='ACTOR/FirstLayer/kernel:0', data=self.actor.variables[0], step=self.global_step)
            tf.summary.histogram(name='ACTOR/FirstLayer/bias:0', data=self.actor.variables[1], step=self.global_step)
            tf.summary.histogram(name='ACTOR/kernel:0', data=self.actor.variables[2], step=self.global_step)
            tf.summary.histogram(name='ACTOR/recurrent_kernel:0', data=self.actor.variables[3], step=self.global_step)
            tf.summary.histogram(name='ACTOR/bias:0', data=self.actor.variables[4], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/kernel:0', data=self.actor.variables[5], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout1/bias:0', data=self.actor.variables[6], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/kernel:0', data=self.actor.variables[7], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout2/bias:0', data=self.actor.variables[8], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/kernel:0', data=self.actor.variables[9], step=self.global_step)
            tf.summary.histogram(name='ACTOR/Readout3/bias:0', data=self.actor.variables[10], step=self.global_step)
            
            tf.summary.histogram(name='CRITIC/FirstLayer/kernel:0', data=self.critic.variables[0], step=self.global_step)
            tf.summary.histogram(name='CRITIC/FirstLayer/bias:0', data=self.critic.variables[1], step=self.global_step)
            tf.summary.histogram(name='CRITIC/kernel:0', data=self.critic.variables[2], step=self.global_step)
            tf.summary.histogram(name='CRITIC/recurrent_kernel:0', data=self.critic.variables[3], step=self.global_step)
            tf.summary.histogram(name='CRITIC/bias:0', data=self.critic.variables[4], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/kernel:0', data=self.critic.variables[5], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout1/bias:0', data=self.critic.variables[6], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/kernel:0', data=self.critic.variables[7], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout2/bias:0', data=self.critic.variables[8], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/kernel:0', data=self.critic.variables[9], step=self.global_step)
            tf.summary.histogram(name='CRITIC/Readout3/bias:0', data=self.critic.variables[10], step=self.global_step)
            summary_writer.flush()
            self.global_step = self.global_step + 1
    
    @tf.function(experimental_relax_shapes=True)
    def _critic_step(self, ret, link_state_critic, first_critic, second_critic, num_edges_critic):
        ret = tf.stop_gradient(ret)

        value = self.critic(link_state_critic, first_critic, second_critic,
                    num_edges_critic, training=True)[0]
        critic_sample_loss = K.square(ret - value)
        return critic_sample_loss
    
    @tf.function(experimental_relax_shapes=True)
    def _actor_step(self, advantage, old_act, old_policy_probs, link_state, graph_id, \
                    first, second, num_edges):
        adv = tf.stop_gradient(advantage)
        old_act = tf.stop_gradient(old_act)
        old_policy_probs = tf.stop_gradient(old_policy_probs)

        r = self.actor(link_state, graph_id, first, second, num_edges, training=True)
        qvalues = tf.reshape(r, (1, len(r)))
        newpolicy_probs = tf.nn.softmax(qvalues)
        newpolicy_probs2 = tf.math.reduce_sum(old_act * newpolicy_probs[0])

        ratio = K.exp(K.log(newpolicy_probs2) - K.log(tf.math.reduce_sum(old_act*old_policy_probs)))
        surr1 = -ratio*adv
        surr2 = -K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * adv
        loss_sample = tf.maximum(surr1, surr2)

        entropy_sample = -tf.math.reduce_sum(K.log(newpolicy_probs) * newpolicy_probs[0])
        return loss_sample, entropy_sample

    def _train_step_combined(self, inds):
        entropies = []
        actor_losses = []
        critic_losses = []
        # Optimize weights
        with tf.GradientTape() as tape:
            for minibatch_ind in inds:
                sample = self.memory[minibatch_ind]

                # ACTOR
                loss_sample, entropy_sample = self._actor_step(sample["advantage"], sample["old_act"], sample["old_policy_probs"], \
                            sample["link_state"], sample["graph_id"], sample["first"], sample["second"], sample["num_edges"])
                actor_losses.append(loss_sample)
                entropies.append(entropy_sample)
                
                # CRITIC
                critic_sample_loss = self._critic_step(sample["return"], sample["link_state_critic"], sample["first_critic"], sample["second_critic"], sample["num_edges_critic"])
                critic_losses.append(critic_sample_loss)
        
            critic_loss = tf.math.reduce_mean(critic_losses)
            final_entropy = tf.math.reduce_mean(entropies)
            actor_loss = tf.math.reduce_mean(actor_losses) - ENTROPY_BETA * final_entropy
            total_loss = actor_loss + critic_loss
        
        grad = tape.gradient(total_loss, sources=self.actor.trainable_weights + self.critic.trainable_weights)
        #gradients = [tf.clip_by_value(gradient, -1., 1.) for gradient in grad]
        grad, _grad_norm = tf.clip_by_global_norm(grad, max_grad_norm)
        self.optimizer.apply_gradients(zip(grad, self.actor.trainable_weights + self.critic.trainable_weights))
        entropies.clear()
        actor_losses.clear()
        critic_losses.clear()
        return actor_loss, critic_loss, final_entropy

    def ppo_update(self, actions, actions_probs, tensors, critic_features, returns, advantages):

        for pos in range(0, int(BUFF_SIZE)):

            tensor = tensors[pos]
            critic_feature = critic_features[pos]
            action = actions[pos]
            ret_value = returns[pos]
            adv_value = advantages[pos]
            action_dist = actions_probs[pos]
            
            final_tensors = ({
                'graph_id': tensor['graph_id'],
                'link_state': tensor['link_state'],
                'first': tensor['first'],
                'second': tensor['second'],
                'num_edges': tensor['num_edges'],
                'link_state_critic': critic_feature['link_state_critic'],
                'old_act': tf.convert_to_tensor(action, dtype=tf.float32),
                'advantage': tf.convert_to_tensor(adv_value, dtype=tf.float32),
                'old_policy_probs': tf.convert_to_tensor(action_dist, dtype=tf.float32),
                'first_critic': critic_feature['first_critic'],
                'second_critic': critic_feature['second_critic'],
                'num_edges_critic': critic_feature['num_edges_critic'],
                'return': tf.convert_to_tensor(ret_value, dtype=tf.float32),
            })      

            self.memory.append(final_tensors)  

        for i in range(PPO_EPOCHS):
            np.random.shuffle(self.inds)

            for start in range(0, BUFF_SIZE, MINI_BATCH_SIZE):
                end = start + MINI_BATCH_SIZE
                actor_loss, critic_loss, final_entropy = self._train_step_combined(self.inds[start:end])
        
        self.memory.clear()
        # self._write_tf_summary(actor_loss, critic_loss, final_entropy)
        gc.collect()
        return actor_loss, critic_loss

def get_advantages(values, masks, rewards):
    returns = []
    gae = 0
    for i in reversed(range(len(rewards))):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])

    adv = np.array(returns) - values[:-1]
    # Normalize advantages to reduce variance
    return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

if __name__ == "__main__":
    # Parse logs and get best model
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-i', help='iters', type=int, required=True)
    parser.add_argument('-c', help='counter model', type=int, required=True)
    parser.add_argument('-e', help='episode iterations', type=int, required=True)
    parser.add_argument('-f1', help='dataset folder name topology 1', type=str, required=True, nargs='+')
    parser.add_argument('-f2', help='dataset folder name topology 2', type=str, required=True, nargs='+')
    parser.add_argument('-f3', help='dataset folder name topology 3', type=str, required=True, nargs='+')
    args = parser.parse_args()

    dataset_folder_name1 = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"+args.f1[0]
    dataset_folder_name2 = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"+args.f2[0]
    dataset_folder_name3 = "../Enero_datasets/dataset_sing_top/data/results_my_3_tops_unif_05-1/"+args.f3[0]

    # Get the environment and extract the number of actions.
    env_training1 = gym.make(ENV_NAME)
    env_training1.seed(SEED)
    env_training1.generate_environment(dataset_folder_name1+"/TRAIN", "BtAsiaPac", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training1.top_K_critical_demands = take_critic_demands

    env_training2 = gym.make(ENV_NAME)
    env_training2.seed(SEED)
    env_training2.generate_environment(dataset_folder_name2+"/TRAIN", "Garr199905", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training2.top_K_critical_demands = take_critic_demands

    env_training3 = gym.make(ENV_NAME)
    env_training3.seed(SEED)
    env_training3.generate_environment(dataset_folder_name3+"/TRAIN", "Goodnet", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_training3.top_K_critical_demands = take_critic_demands

    env_eval = gym.make(ENV_NAME)
    env_eval.seed(SEED)
    env_eval.generate_environment(dataset_folder_name1+"/EVALUATE", "BtAsiaPac", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval.top_K_critical_demands = take_critic_demands

    env_eval2 = gym.make(ENV_NAME)
    env_eval2.seed(SEED)
    env_eval2.generate_environment(dataset_folder_name2+"/EVALUATE", "Garr199905", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval2.top_K_critical_demands = take_critic_demands

    env_eval3 = gym.make(ENV_NAME)
    env_eval3.seed(SEED)
    env_eval3.generate_environment(dataset_folder_name3+"/EVALUATE", "Goodnet", EPISODE_LENGTH, NUM_ACTIONS, percentage_demands)
    env_eval3.top_K_critical_demands = take_critic_demands

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    fileLogs = open("./Logs/exp" + differentiation_str + "Logs.txt", "a")

    # Load maximum reward from previous iterations and the current lr
    if os.path.exists("./tmp/" + differentiation_str + "tmp.pckl"):
        f = open("./tmp/" + differentiation_str + "tmp.pckl", 'rb')
        max_reward, hparams['learning_rate'] = pickle.load(f)
        f.close()
    else:
        max_reward = -1000

    # Decay lr
    if args.i%DECAY_STEPS==0:
        hparams['learning_rate'] = decayed_learning_rate(args.i)

    if args.i>=ENTROPY_STEP:
        ENTROPY_BETA = ENTROPY_BETA/10

    agent = PPOActorCritic()

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
    checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)

    if args.i>0:
        # -1 because the current value is to store the model that we train in this iteration
        checkpoint_actor = tf.train.Checkpoint(model=agent.actor, optimizer=agent.optimizer)
        checkpoint_actor.restore(checkpoint_dir + "/ckpt_ACT-" + str(args.c-1))
        checkpoint_critic = tf.train.Checkpoint(model=agent.critic, optimizer=agent.optimizer)
        checkpoint_critic.restore(checkpoint_dir + "/ckpt_CRT-" + str(args.c-1))

    reward_id = 0
    evalMeanReward = 0
    counter_store_model = args.c

    rewards_test = np.zeros(EVALUATION_EPISODES*3)
    error_links = np.zeros(EVALUATION_EPISODES*3)
    max_link_uti = np.zeros(EVALUATION_EPISODES*3)
    min_link_uti = np.zeros(EVALUATION_EPISODES*3)
    uti_std = np.zeros(EVALUATION_EPISODES*3)

    training_tm_ids = set(range(100))

    for iters in range(args.e):
        states = []
        critic_features = []
        tensors = []
        actions = []
        values = []
        masks = []
        rewards = []
        actions_probs = []

        print("MIDDLEPOINT ROUTING(3 TOP Topologies Enero "+experiment_letter+") PPO EPISODE: ", args.i+iters)
        number_samples_reached = False
        tm_id = random.sample(training_tm_ids, 1)[0]
        while not number_samples_reached:
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 1
            ######

            demand, source, destination = env_training1.reset(tm_id)
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)

                # Predict probabilities over middlepoints
                action_dist, tensor = agent.pred_action_distrib_sp(env_training1, source, destination)

                features = agent.critic_get_graph_features(env_training1)

                q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                        features['num_edges_critic'], training=False)[0].numpy()[0]

                action = np.random.choice(len(action_dist), p=action_dist)
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                reward, done, _, new_demand, new_source, new_destination, _, _, _ = env_training1.step(action, demand, source, destination)
                mask = not done

                states.append((env_training1.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                # If we have enough samples
                if len(states) == num_samples_top1:
                    number_samples_reached = True
                    break

                if done:
                    break

        number_samples_reached = False
        tm_id = random.sample(training_tm_ids, 1)[0]
        while not number_samples_reached:
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 2
            ######

            demand, source, destination = env_training2.reset(tm_id)
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)
                # Predict probabilities over middlepoints
                action_dist, tensor = agent.pred_action_distrib_sp(env_training2, source, destination)
                features = agent.critic_get_graph_features(env_training2)

                q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                        features['num_edges_critic'], training=False)[0].numpy()[0]

                action = np.random.choice(len(action_dist), p=action_dist)
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                reward, done, _, new_demand, new_source, new_destination, _, _, _ = env_training2.step(action, demand, source, destination)
                mask = not done

                states.append((env_training2.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                # If we have enough samples
                if len(states) == num_samples_top1+num_samples_top2:
                    number_samples_reached = True
                    break

                if done:
                    break

        number_samples_reached = False
        tm_id = random.sample(training_tm_ids, 1)[0]
        while not number_samples_reached:
            ######
            # GENERATING EXPERIENCES ON TOPOLOGY 3
            ######

            demand, source, destination = env_training3.reset(tm_id)
            while 1:
                # Used to clean the TF cache
                tf.random.set_seed(1)
                # Predict probabilities over middlepoints
                action_dist, tensor = agent.pred_action_distrib_sp(env_training3, source, destination)
                features = agent.critic_get_graph_features(env_training3)

                q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                        features['num_edges_critic'], training=False)[0].numpy()[0]

                action = np.random.choice(len(action_dist), p=action_dist)
                action_onehot = tf.one_hot(action, depth=len(action_dist), dtype=tf.float32).numpy()

                # Allocate the traffic of the demand to the paths to middlepoint
                reward, done, _, new_demand, new_source, new_destination, _, _, _ = env_training3.step(action, demand, source, destination)
                mask = not done

                states.append((env_training3.edge_state, demand, source, destination))
                tensors.append(tensor)
                critic_features.append(features)
                actions.append(action_onehot)
                values.append(q_value)
                masks.append(mask)
                rewards.append(reward)
                actions_probs.append(action_dist)

                demand = new_demand
                source = new_source
                destination = new_destination

                # If we have enough samples
                if len(states) == num_samples_top1+num_samples_top2+num_samples_top3:
                    number_samples_reached = True
                    break

                if done:
                    break

        features = agent.critic_get_graph_features(env_training3)
        q_value = agent.critic(features['link_state_critic'], features['first_critic'], features['second_critic'],
                features['num_edges_critic'], training=False)[0].numpy()[0]       
        values.append(q_value)

        returns, advantages = get_advantages(values, masks, rewards)
        actor_loss, critic_loss = agent.ppo_update(actions, actions_probs, tensors, critic_features, returns, advantages)
        fileLogs.write("a," + str(actor_loss.numpy()) + ",\n")
        fileLogs.write("c," + str(critic_loss.numpy()) + ",\n")
        fileLogs.flush()

        # Evaluate on FIRST TOPOLOGY
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            demand, source, destination = env_eval.reset(tm_id)
            done = False
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval, source, destination)
                
                action = np.argmax(action_dist)
                reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_eval.step(action, demand, source, destination)
                rewardAddTest += reward
                if done:
                    break
            rewards_test[eps] = rewardAddTest
            error_links[eps] = error_eval_links
            max_link_uti[eps] = maxLinkUti[2]
            min_link_uti[eps] = minLinkUti
            uti_std[eps] = utiStd
        
        # Evaluate on SECOND TOPOLOGy
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            posi = EVALUATION_EPISODES+eps
            demand, source, destination = env_eval2.reset(tm_id)
            done = False
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval2, source, destination)
                
                action = np.argmax(action_dist)
                reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_eval2.step(action, demand, source, destination)
                rewardAddTest += reward
                if done:
                    break
            rewards_test[posi] = rewardAddTest
            error_links[posi] = error_eval_links
            max_link_uti[posi] = maxLinkUti[2]
            min_link_uti[posi] = minLinkUti
            uti_std[posi] = utiStd

        # Evaluate on THIRD TOPOLOGY
        for eps in range(EVALUATION_EPISODES):
            tm_id = eps
            posi = EVALUATION_EPISODES*2+eps
            demand, source, destination = env_eval3.reset(tm_id)
            done = False
            rewardAddTest = 0
            while 1:
                action_dist, _ = agent.pred_action_distrib_sp(env_eval3, source, destination)
                
                action = np.argmax(action_dist)
                reward, done, error_eval_links, demand, source, destination, maxLinkUti, minLinkUti, utiStd = env_eval3.step(action, demand, source, destination)
                rewardAddTest += reward
                if done:
                    break
            rewards_test[posi] = rewardAddTest
            error_links[posi] = error_eval_links
            max_link_uti[posi] = maxLinkUti[2]
            min_link_uti[posi] = minLinkUti
            uti_std[posi] = utiStd

        evalMeanReward = np.mean(rewards_test)
        fileLogs.write(";," + str(np.mean(uti_std)) + ",\n")
        fileLogs.write("+," + str(np.mean(error_links)) + ",\n")
        fileLogs.write("<," + str(np.amax(max_link_uti)) + ",\n")
        fileLogs.write(">," + str(np.amax(min_link_uti)) + ",\n")
        fileLogs.write("ENTR," + str(ENTROPY_BETA) + ",\n")
        #fileLogs.write("-," + str(agent.epsilon) + ",\n")
        fileLogs.write("REW," + str(evalMeanReward) + ",\n")
        fileLogs.write("lr," + str(hparams['learning_rate']) + ",\n")
  
        if evalMeanReward>max_reward:
            max_reward = evalMeanReward
            reward_id = counter_store_model
            fileLogs.write("MAX REWD: " + str(max_reward) + " REWD_ID: " + str(reward_id) +",\n")
        
        fileLogs.flush()
        
        # Store trained model
        # Storing the model and the tape.gradient make the memory increase
        checkpoint_actor.save(checkpoint_prefix+'_ACT')
        checkpoint_critic.save(checkpoint_prefix+'_CRT')
        counter_store_model = counter_store_model + 1
        K.clear_session()
        gc.collect()

    f = open("./tmp/" + differentiation_str + "tmp.pckl", 'wb')
    pickle.dump((max_reward, hparams['learning_rate']), f)
    f.close()

