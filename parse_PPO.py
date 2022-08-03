import argparse
import numpy as np
import os
import matplotlib.pyplot as plt
from operator import add, sub
from scipy.signal import savgol_filter

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed

def read_max_load_link(standard_out_file):
    pre_optim_max_load_link, post_optim_max_load_link = 0, 0
    with open(standard_out_file) as fd:
        while (True):
            line = fd.readline()
            if line.startswith("pre-optimization"):
                camps = line.split(" ")
                pre_optim_max_load_link = float(camps[-1].split('\n')[0])
            elif line.startswith("post-optimization"):
                camps = line.split(" ")
                post_optim_max_load_link = float(camps[-1].split('\n')[0])
                break
        return (pre_optim_max_load_link, post_optim_max_load_link)

if __name__ == "__main__":
    # python parse_PPO.py -d ./Logs/expSP_3top_15_B_NEWLogs.txt
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-d', help='data file', type=str, required=True, nargs='+')
    args = parser.parse_args()

    aux = args.d[0].split(".")
    aux = aux[1].split("exp")
    differentiation_str = str(aux[1].split("Logs")[0])

    actor_loss = []
    critic_loss = []
    avg_std = []
    max_link_uti = []
    min_link_uti = []
    defo_max_uti = []
    error_links = []
    avg_rewards = []
    learning_rate = []
    cummulative_rewards = []

    if not os.path.exists("./Images"):
        os.makedirs("./Images")

    if not os.path.exists("./Images/TRAINING/"+differentiation_str):
        os.makedirs("./Images/TRAINING/"+differentiation_str)
    
    path_to_dir = "./Images/TRAINING/"+differentiation_str+"/"

    model_id = 0
    # Load best model
    with open(args.d[0]) as fp:
        for line in reversed(list(fp)):
            arrayLine = line.split(":")
            if arrayLine[0]=='MAX REWD':
                model_id = int(arrayLine[2].split(",")[0])
                break
    
    print("Model with maximum reward: ", model_id)

    with open(args.d[0]) as fp:
        for line in fp:
            arrayLine = line.split(",")
            if arrayLine[0]=="<":
                max_link_uti.append(float(arrayLine[1]))
            elif arrayLine[0]==">":
                min_link_uti.append(float(arrayLine[1]))
            elif arrayLine[0]=="a":
                actor_loss.append(float(arrayLine[1]))
            elif arrayLine[0]=="lr":
                learning_rate.append(float(arrayLine[1]))
            elif arrayLine[0]==";":
                avg_std.append(float(arrayLine[1]))
            elif arrayLine[0]=="+":
                error_links.append(float(arrayLine[1]))
            elif arrayLine[0]=="REW":
                if float(arrayLine[1])<-3000:
                    avg_rewards.append(-3000)
                else:
                    avg_rewards.append(float(arrayLine[1]))
            elif arrayLine[0]=="c":
                critic_loss.append(float(arrayLine[1]))

        plt.plot(actor_loss)
        plt.xlabel("Training Episode")
        plt.ylabel("ACTOR Loss")
        plt.savefig(path_to_dir+"ACTORLoss" + differentiation_str)
        plt.close()

        plt.plot(critic_loss)
        plt.xlabel("Training Episode")
        plt.ylabel("CRITIC Loss (MSE)")
        plt.yscale("log")
        plt.savefig(path_to_dir+"CRITICLoss" + differentiation_str)
        plt.close()

        plt.plot(max_link_uti, label="DRL Max Link Uti")
        plt.plot(defo_max_uti, label="DEFO Max Link Uti", c="tab:red")
        
        print("DRL MAX reward: ", np.amax(avg_rewards))
        plt.xlabel("Episodes")
        lgd = plt.legend(loc="lower left", bbox_to_anchor=(0.07, -0.22), ncol=2, fancybox=True, shadow=True)
        plt.title("GNN+DQN Testing score")
        plt.ylabel("Maximum link utilization")
        #plt.yscale('log')
        plt.savefig(path_to_dir+"MaxLinkUti" + differentiation_str, bbox_extra_artists=(lgd,), bbox_inches='tight')
        plt.close()

        plt.plot(min_link_uti)
        plt.xlabel("Episodes")
        plt.title("GNN+DQN Testing score")
        plt.ylabel("Minimum link utilization")
        plt.savefig(path_to_dir+"MinLinkUti" + differentiation_str)
        plt.close()

        plt.plot(avg_rewards)
        plt.xlabel("Episodes")
        plt.title("GNN+DQN Testing score")
        plt.ylabel("Average reward")
        plt.savefig(path_to_dir+"AvgReward" + differentiation_str)
        plt.close()

        plt.plot(learning_rate)
        plt.xlabel("Episodes")
        plt.title("GNN+DQN Testing score")
        plt.ylabel("Learning rate")
        plt.savefig(path_to_dir+"Lr_" + differentiation_str)
        plt.close()

        plt.plot(error_links)
        plt.xlabel("Episodes")
        plt.title("GNN+DQN Testing score")
        plt.ylabel("Error link (sum_total_TM/num_links")
        plt.savefig(path_to_dir+"ErrorLinks" + differentiation_str)
        plt.close()

        plt.plot(avg_std)
        plt.xlabel("Episodes")
        plt.title("GNN+DQN Testing score")
        plt.ylabel("Avg std of link utilization")
        plt.savefig(path_to_dir+"AvgStdUti" + differentiation_str)
        plt.close()

