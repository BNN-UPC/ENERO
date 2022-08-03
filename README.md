# ENERO: Efficient real-time WAN routing optimization with Deep Reinforcement Learning

#### Link to paper: [here](https://www.sciencedirect.com/science/article/pii/S1389128622002717)

#### Paul Almasan, Shihan Xiao, Xiangle Cheng, Xiang Shi, Pere Barlet-Ros, Albert Cabellos-Aparicio

## Abstract
Wide Area Networks (WAN) are a key infrastructure in today’s society. During the last years, WANs have seen a considerable increase in network’s traffic and network applications, imposing new requirements on existing network technologies (e.g., low latency and high throughput). Consequently, Internet Service Providers (ISP) are under pressure to ensure the customer’s Quality of Service and fulfill Service Level Agreements. Network operators leverage Traffic Engineering (TE) techniques to efficiently manage the network’s resources. However, WAN’s traffic can drastically change during time and the connectivity can be affected due to external factors (e.g., link failures). Therefore, TE solutions must be able to adapt to dynamic scenarios in real-time.

In this paper we propose Enero, an efficient real-time TE solution based on a two-stage optimization process. In the first one, Enero leverages Deep Reinforcement Learning (DRL) to optimize the routing configuration by generating a long-term TE strategy. To enable efficient operation over dynamic network scenarios (e.g., when link failures occur), we integrated a Graph Neural Network into the DRL agent. In the second stage, Enero uses a Local Search algorithm to improve DRL’s solution without adding computational overhead to the optimization process. The experimental results indicate that Enero is able to operate in real-world dynamic network topologies in 4.5 s on average for topologies up to 100 links.

# Instructions to set up the Environment

1. First, create the virtual environment and activate the environment.
```ruby
virtualenv -p python3 myenv
source myenv/bin/activate
```

2. Then, we install all the required packages.
```ruby
pip install -r requirements.txt
```

3. Register custom gym environment.
```ruby
pip install -e gym-graph/
```

# Instructions to prepare the datasets

The source code already provides the data and the trained model used in the paper. Therefore, we can start by evaluating the model executing the following script. However, the following steps can be used to prepare the data to train on other topologies.

1. Download the dataset from https://drive.google.com/file/d/1HQyRxj8F2J8ScQxeEvSvUV_RuVtehCBh/view?usp=sharing and unzip it. Then, copy the directory "results_zoo" to a new directory called "results-1-link_capacity-unif-05-1-zoo" located one level outside of this code's directory (i.e., do "cd .." in the terminal).

2. Copy the topology datasets from "results-1-link_capacity-unif-05-1-zoo" to "./ENERO/dataset/data/my_training_topologies/". In the paper, we copy the "EliBackbone", "Janetbackbone" and "HurricaneElectric" datasets.

3. Execute the following command to split the data between training and evaluation.
```ruby
python convert_dataset_Albert.py -d my_training_topologies -name Aarnet
```

# Instructions to obtain the Figures from the paper

1. First we unzip the "dataset_sing_top.zip" in "../", which is one directory level outside of where this readme is located. Then, we execute the following command:

```ruby
python figures_5_and_6.py -d SP_3top_15_B_NEW 
```

2. Then, we execute the following (one per topology):
```ruby
python figure_7.py -d SP_3top_15_B_NEW -p ../dataset_sing_top/evalRes_NEW_EliBackbone/EVALUATE/ -t EliBackbone

python figure_7.py -d SP_3top_15_B_NEW -p ../dataset_sing_top/evalRes_NEW_HurricaneElectric/EVALUATE/ -t HurricaneElectric

python figure_7.py -d SP_3top_15_B_NEW -p ../dataset_sing_top/evalRes_NEW_Janetbackbone/EVALUATE/ -t Janetbackbone
```

3. Next, we generate the link failure Figures (one per topology):
```ruby
python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../dataset_sing_top/LinkFailure/rwds-LinkFailure_HurricaneElectric

python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../dataset_sing_top/LinkFailure/rwds-LinkFailure_Janetbackbone

python figure_8.py -d SP_3top_15_B_NEW -num_topologies 20 -f ../dataset_sing_top/LinkFailure/rwds-LinkFailure_EliBackbone
```

4. Finally, we generate the generalization figure. For this figure, we need to unzip "DEFOResults.zip" and "rwds-results-1-link_capacity-unif-05-1-zoo.zip" in "../", which is one directory level outside.

```ruby
python figure_9.py -d SP_3top_15_B_NEW -p ../rwds-results-1-link_capacity-unif-05-1-zoo
```


Please cite the corresponding article if you use the code from this repository:

```
@article{ALMASAN2022109166,
title = {ENERO: Efficient real-time WAN routing optimization with Deep Reinforcement Learning},
journal = {Computer Networks},
volume = {214},
pages = {109166},
year = {2022},
issn = {1389-1286},
doi = {https://doi.org/10.1016/j.comnet.2022.109166},
url = {https://www.sciencedirect.com/science/article/pii/S1389128622002717},
author = {Paul Almasan and Shihan Xiao and Xiangle Cheng and Xiang Shi and Pere Barlet-Ros and Albert Cabellos-Aparicio},
keywords = {Routing, Optimization, Deep Reinforcement Learning, Graph Neural Networks},
abstract = {Wide Area Networks (WAN) are a key infrastructure in today’s society. During the last years, WANs have seen a considerable increase in network’s traffic and network applications, imposing new requirements on existing network technologies (e.g., low latency and high throughput). Consequently, Internet Service Providers (ISP) are under pressure to ensure the customer’s Quality of Service and fulfill Service Level Agreements. Network operators leverage Traffic Engineering (TE) techniques to efficiently manage the network’s resources. However, WAN’s traffic can drastically change during time and the connectivity can be affected due to external factors (e.g., link failures). Therefore, TE solutions must be able to adapt to dynamic scenarios in real-time. In this paper we propose Enero, an efficient real-time TE solution based on a two-stage optimization process. In the first one, Enero leverages Deep Reinforcement Learning (DRL) to optimize the routing configuration by generating a long-term TE strategy. To enable efficient operation over dynamic network scenarios (e.g., when link failures occur), we integrated a Graph Neural Network into the DRL agent. In the second stage, Enero uses a Local Search algorithm to improve DRL’s solution without adding computational overhead to the optimization process. The experimental results indicate that Enero is able to operate in real-world dynamic network topologies in 4.5 s on average for topologies up to 100 links.}
}
```
