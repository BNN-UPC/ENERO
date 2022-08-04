import os
import numpy as np
import argparse


if __name__ == "__main__":
    # python convert_dataset.py -d results_single_top -name Garr199905
    parser = argparse.ArgumentParser(description='Parse file and create plots')

    parser.add_argument('-f1', help='File where the topologies are stored', type=str, required=True, nargs='+')
    parser.add_argument('-name', help='Topology name', type=str, required=True, nargs='+')

    args = parser.parse_args()
    data_dir = args.f1[0]
    topology_name = args.name[0]
    new_dir = "../Enero_datasets/dataset_sing_top/data/"+data_dir+"/NEW_"+topology_name+'/'
    new_dir_train = new_dir+"TRAIN"
    new_dir_eval = new_dir+"EVALUATE"
    new_dir_all = new_dir+"ALL"

    new_dir_TM_train = new_dir_train+"/TM/"
    new_dir_TM_eval = new_dir_eval+"/TM/"
    new_dir_TM_all = new_dir_all+"/TM/"

    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    else:
        os.system("rm -rf %s" % (new_dir))
        os.makedirs(new_dir)

    os.makedirs(new_dir_train)
    os.makedirs(new_dir_eval)
    os.makedirs(new_dir_all)

    os.makedirs(new_dir_TM_train)
    os.makedirs(new_dir_TM_eval)
    os.makedirs(new_dir_TM_all)

    # aux_top_name = "./data/"+data_dir+"/"+topology_name+"/graph_"+topology_name+".txt"
    aux_top_name = "../Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo/"+topology_name+"/"+topology_name+".graph"

    os.system("cp %s %s.graph" % (aux_top_name, new_dir_train+'/'+topology_name))
    os.system("cp %s %s.graph" % (aux_top_name, new_dir_eval+'/'+topology_name))
    os.system("cp %s %s.graph" % (aux_top_name, new_dir_all+'/'+topology_name))

    num_TMs_train = 0
    nums_TMs_eval = 0
    nums_TMs_all = 0

    for subdir, dirs, files in os.walk("../Enero_datasets/results-1-link_capacity-unif-05-1/results_zoo/"+topology_name+"/TM/"):
        for file in files:
            #if file.startswith('TM-'):
            if file.endswith('.demands'):
                dst_dir = new_dir_TM_eval
                tm_iter = nums_TMs_eval
                # Compute handcrafted 70% of the total samples
                if num_TMs_train<100:
                    dst_dir = new_dir_TM_train
                    tm_iter = num_TMs_train
                os.system("cp %s %s.%s.demands" % (subdir+'/'+file, dst_dir+'/'+topology_name, tm_iter))
                
                if num_TMs_train<100:
                    num_TMs_train += 1
                else:
                    nums_TMs_eval += 1
                
                dst_dir = new_dir_TM_all
                tm_iter = nums_TMs_all
                os.system("cp %s %s.%s.demands" % (subdir+'/'+file, dst_dir+'/'+topology_name, tm_iter))
                nums_TMs_all += 1
                