import os
import time as tt
import resource
import subprocess

max_iters = 3000 # Total number of training episodes
episode_iters = 20 # How many training episodes to execute before the training script is called again

# NOTICE: The training script trains and stores the models every 20 episode_iters. When the batch of episode_iters
# finishes, the model is stored and in the next itreation is loaded again to start the training were it was left before.
# This is to avoid some memory leak issue existing with TF

if __name__ == "__main__":

    if not os.path.exists("./Logs"):
        os.makedirs("./Logs")

    if not os.path.exists("./tmp"):
        os.makedirs("./tmp")

    iters = 0
    counter_store_model = 1

    dataset_folder_name1 = "NEW_BtAsiaPac"
    dataset_folder_name2 = "NEW_Garr199905"
    dataset_folder_name3 = "NEW_Goodnet"

    while iters < max_iters:
        processes = []
        subprocess.call(['python train_Enero_15demands_3top_script.py -i '+str(iters)+ ' -c '+str(counter_store_model)+' -e '+str(episode_iters)+ ' -f1 '+dataset_folder_name1+' -f2 '+dataset_folder_name2+' -f3 '+dataset_folder_name3], shell=True)

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        counter_store_model = counter_store_model + episode_iters
        iters = iters + episode_iters


