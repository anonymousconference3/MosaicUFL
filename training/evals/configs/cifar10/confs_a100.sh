worker_ips:
    - 10.70.150.150:[0,1]

exp_path: /scratch/ek59/ny4733/tmp/iDLSys_client_selection_2023-master/training


#python_path: /home/datadisk/zikang/anaconda3/envs/oort/bin/

auth:
    ssh_user: "zikang"
    ssh_private_key: ~/.ssh/id_rsa
    additional_auth:
        - ip: 10.70.150.150
          username: "zikang"
          password: "zikang123"

# cmd to run before we can indeed run oort (in order)
setup_commands:
    # - source /home/datadisk/zikang/anaconda3/bin/activate oort     
    # - export NCCL_SOCKET_IFNAME='enp94s0f0'         # Run "ifconfig" to ensure the right NIC for nccl if you have multiple NICs --enp1s0

# ========== Additional job configuration ========== 
# Default parameters are specified in argParser.py, wherein more description of the parameter can be found



job_conf:
    - log_path: /scratch/ek59/ny4733/tmp/iDLSys_client_selection_2023-master/training/evals/log # Path of log files
    - job_name: cifar10                   # Generate logs under this folder: log_path/job_name/time_stamp
    - total_worker: 72                    # Number of participants per round, we use K=100 in our paper, large K will be much slower
    - data_set: cifar10                     # Dataset: openImg, google_speech, stackoverflow
    - data_dir: /scratch/ek59/ny4733/tmp/iDLSys_client_selection_2023-master/training/fedil/data    # Path of the dataset
    - data_mapfile: /scratch/ek59/ny4733/tmp/iDLSys_client_selection_2023-master/cilentDataMap/cifar10/data_to_client_map_100.pkl           # Allocation of data to each client, turn to iid setting if not provided
    - client_path: /scratch/ek59/ny4733/tmp/iDLSys_client_selection_2023-master/FLPerf/benchmark/dataset/data/device_info/client_behave_trace.pkl     # Path of the client trace
    - sample_mode: random                            # Client selection: random, oort
    - model: shufflenet_v2_x2_0                            # Models: shufflenet_v2_x2_0, mobilenet_v2, resnet34, albert-base-v2
    - gradient_policy: yogi                 # Commenting out this line will turn to "Fedprox"
    - round_penalty: 2.0                    # Penalty factor in our paper (\alpha), \alpha -> 0 turns to (Oort w/o sys)
    - eval_interval: 1                     # How many rounds to run a testing on the testing set
    - epochs: 2000                       # Number of rounds to run this training. We use 1000 in our paper, while it may converge w/ ~400 rounds
    - pacer_delta: 10
    - num_clients: 100
    - seed: 1234
    - all_update: 0 # aggregate all the update model from all client
    - cos: 0
    - load_model: 0
    - load_client: 0
    - load_path: "/scratch/ek59/ny4733/tmp/iDLSys_client_selection_2023-master/training/evals/model/9_0.1l_s_change300_record_cos_dd_model_weights_cifar10.pkl"
    - label_rate: 0.01
    - batch_size: 10
    - l_distance: 0 #1: aggregate using distance
    - loss_record: 0
    - server_update: 1
    - server_update_interval: 10
    - record_name: 72_0.1l_s_change300_record_cos_dd
    - cos_limit: 0.1
    - change_limit: 1
    - record_cos: 1
