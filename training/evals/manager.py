
# import yaml
# import torch
# import sys
# import time
# import random
# import os, subprocess
# import pickle, datetime

# def load_yaml_conf(yaml_file):
#     with open(yaml_file) as fin:
#         data = yaml.load(fin, Loader=yaml.FullLoader)
#     return data

# def process_cmd(yaml_file):
#     print("CUDA Available:", torch.cuda.is_available())
#     print("CUDA Device Count:", torch.cuda.device_count())

#     # 打印当前 PyTorch 版本。
#     print(torch.__version__)

#     # 打印当前安装的 CUDA 版本。
#     print(torch.version.cuda)

#     # 打印 cuDNN 库的版本。
#     print(torch.backends.cudnn.version())

#     # 打印 NCCL 库的版本，NCCL 用于 GPU 间的通信。
#     print(torch.cuda.nccl.version())

#     # 检查是否有 CUDA 设备可用。
#     if not torch.cuda.is_available():
#         # 如果没有可用的 CUDA 设备，打印相应的信息。
#         print("CUDA is not available.")
#     else:
#         # 如果有 CUDA 设备可用，获取 CUDA 设备的数量。
#         device_count = torch.cuda.device_count()
#         # 打印可用的 CUDA 设备数量。
#         print(f"Number of available CUDA devices: {device_count}")

#         # 列出所有可用的 CUDA 设备及其编号。
#         for i in range(device_count):
#             # 获取每个 CUDA 设备的名称。
#             device_name = torch.cuda.get_device_name(i)
#             # 打印设备编号和设备名称。
#             print(f"Device {i}: {device_name}")

#     # 加载 YAML 配置文件，返回一个字典。
#     yaml_conf = load_yaml_conf(yaml_file)

#     # 从配置中提取参数服务器的 IP 地址。
#     ps_ip = yaml_conf['ps_ip']

#     # 初始化工作节点 IP 地址列表和 GPU 数量列表。
#     worker_ips, total_gpus = [], []
#     # 初始化命令脚本列表（尽管此行未在代码中使用）。
#     cmd_script_list = []

#     # 遍历配置中的工作节点信息。
#     for ip_gpu in yaml_conf['worker_ips']:
#         # 分割每个节点的 IP 地址和 GPU 数量。
#         ip, num_gpu = ip_gpu.strip().split(':')
#         # 将解析出的 IP 地址添加到工作节点列表。
#         worker_ips.append(ip)
#         # 将解析出的 GPU 数量转换为整数并添加到 GPU 数量列表。
#         total_gpus.append(int(num_gpu))

#     # 生成当前时间的时间戳，用于日志文件名和其他记录目的。
#     time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')
#     # 初始化一个集合，用于记录所有运行中的虚拟机 IP。
#     running_vms = set()
#     # 设置作业名称。
#     job_name = 'kuiper_job'
#     # 设置日志路径。
#     log_path = './logs'
#     # 构造提交作业的用户信息，如果提供了用户名，将其添加到命令中。
#     submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""


#     # 初始化作业配置字典。
#     job_conf = {'time_stamp': time_stamp,
#                 'total_worker': sum(total_gpus),
#                 'ps_ip': ps_ip,
#                 'ps_port': random.randint(1000, 60000),
#                 'manager_port': random.randint(1000, 60000)
#                 }

#     # 更新作业配置字典，根据配置文件中的额外设置。
#     for conf in yaml_conf['job_conf']:
#         job_conf.update(conf)

#     # 初始化配置脚本和设置命令字符串。
#     conf_script = ''
#     setup_cmd = ''
#     # 如果配置文件中定义了初始化命令，构造一个完整的初始化命令字符串。
#     if yaml_conf['setup_commands'] is not None:
#         setup_cmd += (yaml_conf['setup_commands'][0] + ' && ')
#         for item in yaml_conf['setup_commands'][1:]:
#             setup_cmd += (item + ' && ')

#     # 初始化命令后缀（此行在代码中未使用）。
#     cmd_sufix = f" "

#     # 打印完整的设置命令，用于调试。
#     print(setup_cmd)
#     # 构造并更新每个配置项的命令脚本。
#     for conf_name in job_conf:
#         conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
#         # 特殊处理作业名称和日志路径配置。
#         if conf_name == "job_name":
#             job_name = job_conf[conf_name]
#         if conf_name == "log_path":
#             log_path = os.path.join(job_conf[conf_name], 'log', job_name, time_stamp)

#     # 构造学习者配置字符串，表示每个 GPU 的序号。
#     learner_conf = '-'.join([str(_) for _ in list(range(1, sum(total_gpus)+1))])
#     # 将任务提交到参数服务器。
#     running_vms.add(ps_ip)
#     ps_cmd = f" export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO python {yaml_conf['exp_path']}/param_server.py {conf_script} --this_rank=0 --learner={learner_conf} "
#     # 创建一个新的日志文件用于记录参数服务器的输出。
#     print(f"{job_name}_logging")
#     with open(f"{job_name}_logging", 'wb') as fout:
#         pass
        
#     # 打印启动参数服务器的消息，并通过 SSH 提交参数服务器启动命令。
#     print(f"Starting aggregator on {ps_ip}...")
#     with open(f"{job_name}_logging", 'a') as fout:

#         cmd = """
#         server: call param_server.py
#         The setup_cmd and ps_cmd: sshpass -p Aa19980824 ssh -o StrictHostKeyChecking=no 
#         zikang@10.70.150.150 "source /home/datadisk/zikang/anaconda3/bin/activate oort &&   export NCCL_IB_DISABLE=1
#         ; export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO 
#         python /home/datadisk/zikang/iDLSys_client_selection_2023-master/training/param_server.py  
#         --time_stamp=0626_233702 --total_worker=100 --ps_ip=10.70.150.150 --ps_port=18001 --manager_port=12372 
#         --log_path=/home/datadisk/zikang/iDLSys_client_selection_2023-master/training/evals/log 
#         --job_name=cifar10 --data_set=cifar10 --data_dir=/home/datadisk/zikang/iDLSys_client_selection_2023-master/FLPerf/FLPerf/benchmark/dataset/data/data 
#         --data_mapfile=/home/datadisk/zikang/iDLSys_client_selection_2023-master/cilentDataMap/cifar10/data_to_client_map.pkl 
#         --client_path=/home/datadisk/zikang/iDLSys_client_selection_2023-master/FLPerf/benchmark/dataset/data/device_info/client_behave_trace.pkl 
#         --sample_mode=random --model=shufflenet_v2_x2_0 --gradient_policy=yogi --round_penalty=2.0 --eval_interval=20 --epochs=3 --pacer_delta=10 --this_rank=0 --learner=1 "
#         """
#         #print("The setup_cmd and ps_cmd: " + f'sshpass -p Aa19980824 ssh -o StrictHostKeyChecking=no {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"')
#         subprocess.Popen(f'sshpass -p Aa19980824 ssh -o StrictHostKeyChecking=no {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"',
#                         shell=True, stdout=fout, stderr=fout)

#     # 等待一定时间，确保参数服务器已经启动（此行在代码中已注释）。
#     # time.sleep(2)
#     # 将任务提交到每个工作节点。
#     # server和client processes 同时进行
#     # 0用来当服务器了
#     rank_id = 1
#     for worker, gpu in zip(worker_ips, total_gpus):
#         running_vms.add(worker)
#         print(f"Starting workers on {worker} ...")
#         for _  in range(gpu):
#             time.sleep(1)

#             worker_cmd = f"export NCCL_IB_DISABLE=1; export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO python {yaml_conf['exp_path']}/learner.py {conf_script} --this_rank={rank_id} --learner={learner_conf} "
#             rank_id += 1

#             with open(f"{job_name}_logging", 'a') as fout:
#                 cmd = """
#                 client : run learner.py
#                 The worker_cmd: sshpass -p Aa19980824 ssh -o StrictHostKeyChecking=no 
#                 zikang@10.70.150.150  "source /home/datadisk/zikang/anaconda3/bin/activate oort &&  export NCCL_IB_DISABLE=1; 
#                 export NCCL_P2P_DISABLE=1; NCCL_DEBUG=INFO 
#                 python /home/datadisk/zikang/iDLSys_client_selection_2023-master/training/learner.py  
#                 --time_stamp=0626_233702 --total_worker=100 --ps_ip=10.70.150.150 --ps_port=18001 --manager_port=12372 
#                 --log_path=/home/datadisk/zikang/iDLSys_client_selection_2023-master/training/evals/log
#                 --job_name=cifar10 --data_set=cifar10 --data_dir=/home/datadisk/zikang/iDLSys_client_selection_2023-master/FLPerf/FLPerf/benchmark/dataset/data/data 
#                 --data_mapfile=/home/datadisk/zikang/iDLSys_client_selection_2023-master/cilentDataMap/cifar10/data_to_client_map.pkl 
#                 --client_path=/home/datadisk/zikang/iDLSys_client_selection_2023-master/FLPerf/benchmark/dataset/data/device_info/client_behave_trace.pkl 
#                 --sample_mode=random --model=shufflenet_v2_x2_0 --gradient_policy=yogi --round_penalty=2.0 --eval_interval=20 --epochs=3 --pacer_delta=10 --this_rank=1 --learner=1 "
#                 """
#                 #print("The worker_cmd: " +f'sshpass -p Aa19980824 ssh -o StrictHostKeyChecking=no {submit_user}{worker}  "{setup_cmd} {worker_cmd}"')
#                 subprocess.Popen(f'sshpass -p Aa19980824 ssh -o StrictHostKeyChecking=no {submit_user}{worker}  "{setup_cmd} {worker_cmd}"',
#                                 shell=True, stdout=fout, stderr=fout)
#     # 保存正在运行的工作节点的地址信息。
#     current_path = os.path.dirname(os.path.abspath(__file__))
#     job_name = os.path.join(current_path, job_name)
#     with open(job_name, 'wb') as fout:
#         job_meta = {'user':submit_user, 'vms': running_vms}
#         pickle.dump(job_meta, fout)

#     # 打印作业提交状态的消息。
#     print(f"Submitted job, please check your logs ({log_path}) for status")


# def terminate(job_name):

#     current_path = os.path.dirname(os.path.abspath(__file__))
#     job_meta_path = os.path.join(current_path, job_name)

#     if not os.path.isfile(job_meta_path):
#         print(f"Fail to terminate {job_name}, as it does not exist")

#     with open(job_meta_path, 'rb') as fin:
#         job_meta = pickle.load(fin)

#     for vm_ip in job_meta['vms']:
#         # os.system(f'scp shutdown.py {job_meta["user"]}{vm_ip}:~/')
#         print(f"Shutting down job on {vm_ip}")
#         os.system(f"sshpass -p Aa19980824 ssh -o StrictHostKeyChecking=no {job_meta['user']}{vm_ip} 'python {current_path}/shutdown.py {job_name}'")
#         # os.system(f"ssh {job_meta['user']}{vm_ip} 'python {current_path}/shutdown.py {job_name}'")


# if sys.argv[1] == 'submit':
#     process_cmd(sys.argv[2])
# elif sys.argv[1] == 'stop':
#     terminate(sys.argv[2])
# else:
#     print("Unknown cmds ...")



# Submit job to the remote cluster

import yaml
import sys
import time
import random
import os, subprocess
import pickle, datetime

import socket

def load_yaml_conf(yaml_file):
    with open(yaml_file) as fin:
        data = yaml.load(fin, Loader=yaml.FullLoader)
    return data

def process_cmd(yaml_file):
    current_path = os.path.dirname(os.path.abspath(__file__))

    config_path=os.path.join(current_path,yaml_file)
    print(config_path)
    yaml_conf = load_yaml_conf(config_path)
    
    # yaml_conf = load_yaml_conf(yaml_file)
    # ps_ip = yaml_conf['ps_ip']
    ps_ip=socket.gethostname()
    worker_ips, total_gpus = [], []

    for ip_gpu in yaml_conf['worker_ips']:
        ip, gpu_list = ip_gpu.strip().split(':')
        ip=socket.gethostname()
        worker_ips.append(ip)
        total_gpus.append(eval(gpu_list))
    
    running_vms = set()
    subprocess_list=set()
    submit_user = f"{yaml_conf['auth']['ssh_user']}@" if len(yaml_conf['auth']['ssh_user']) else ""


    total_gpu_processes =  sum([sum(x) for x in total_gpus])
    learner_conf = '-'.join([str(_) for _ in list(range(1, total_gpu_processes+1))])

    conf_script = ''
    setup_cmd = ''
    if yaml_conf['setup_commands'] is not None:
        for item in yaml_conf['setup_commands']:
            setup_cmd += (item + ' && ')


    time_stamp = datetime.datetime.fromtimestamp(time.time()).strftime('%m%d_%H%M%S')+'_'+str(random.randint(1,60000))
    job_conf = {'time_stamp':time_stamp,
                'total_worker': total_gpu_processes,
                'ps_ip':ps_ip,
                'ps_port':random.randint(1000, 60000),
                'manager_port':random.randint(1000, 60000),
                }
    
    for conf in yaml_conf['job_conf']:
        job_conf.update(conf)

    job_name = job_conf['job_name']
    
    # if len(sys.argv)>3:
    #     job_conf['sample_mode'] = sys.argv[3]
    # if len(sys.argv)>4:
    #     # job_conf['load_model'] = True
    #     job_conf['load_time_stamp'] = sys.argv[4]
    #     job_conf['load_epoch'] = sys.argv[5]
    #     job_conf["model_path"]=os.path.join(job_conf["log_path"], 'logs', job_name, job_conf['load_time_stamp'])
    
    for conf_name in job_conf:
        conf_script = conf_script + f' --{conf_name}={job_conf[conf_name]}'
    if len(sys.argv)>3:
        conf_script = conf_script + f" --data_dir={sys.argv[3] +'/'+ job_name}"
        print(conf_script)
    log_file_name=os.path.join(current_path,f"{job_name}_logging") 
    # =========== Submit job to parameter server ============
    running_vms.add(ps_ip)
    #ps_cmd = f" {yaml_conf['python_path']}/python {yaml_conf['exp_path']}/param_server.py {conf_script} --this_rank=0 --learner={learner_conf} --gpu_device=0"
    ps_cmd = f"{sys.executable} {yaml_conf['exp_path']}/param_server.py {conf_script} --this_rank=0 --learner={learner_conf} --gpu_device=0"
    print(conf_script)

    print(f"Starting time_stamp on {time_stamp}...")

    with open(log_file_name, 'wb') as fout:
        pass
    
    print(f"Starting aggregator on {ps_ip}...")
    with open(log_file_name, 'a') as fout:
        # p=subprocess.Popen(f'ssh -tt {submit_user}{ps_ip} "{setup_cmd} {ps_cmd}"', shell=True, stdout=fout, stderr=fout)
        
        # p=subprocess.Popen(f'{ps_cmd}', shell=True, stdout=fout, stderr=fout)
        cmd_sequence=f'{ps_cmd}'
        cmd_sequence=cmd_sequence.split()
        p = subprocess.Popen(cmd_sequence,stdout=fout, stderr=fout)

        subprocess_list.add(p)
        time.sleep(30)

    # =========== Submit job to each worker ============
    rank_id = 1
    for worker, gpu in zip(worker_ips, total_gpus):
        running_vms.add(worker)
        print(f"Starting workers on {worker} ...")
        for gpu_device  in range(len(gpu)):
            for _  in range(gpu[gpu_device]):
                #worker_cmd = f" {yaml_conf['python_path']}/python {yaml_conf['exp_path']}/learner.py {conf_script} --this_rank={rank_id} --learner={learner_conf} --gpu_device={gpu_device}"
                worker_cmd = f"{sys.executable} {yaml_conf['exp_path']}/learner.py {conf_script} --this_rank={rank_id} --learner={learner_conf} --gpu_device={gpu_device}"
                rank_id += 1

                with open(log_file_name, 'a') as fout:
                    # p=subprocess.Popen(f'ssh -tt {submit_user}{worker} "{setup_cmd} {worker_cmd}"', shell=True, stdout=fout, stderr=fout)
                    
                    # p=subprocess.Popen(f'{worker_cmd}', shell=True, stdout=fout, stderr=fout)

                    cmd_sequence=f'{worker_cmd}'
                    cmd_sequence=cmd_sequence.split()
                    p = subprocess.Popen(cmd_sequence,stdout=fout, stderr=fout)

                    subprocess_list.add(p)

    exit_codes = [p.wait() for p in subprocess_list]

    # dump the address of running workers
    # job_name = os.path.join(current_path, f"/job_info/{job_name}_{time_stamp}")
    # with open(job_name, 'wb') as fout:
    #     job_meta = {'user':submit_user, 'vms': running_vms}
    #     pickle.dump(job_meta, fout)

    print(f"Submitted job, please check your logs ({log_file_name}) for status")

def terminate(job_name):

    current_path = os.path.dirname(os.path.abspath(__file__))
    job_meta_path = os.path.join(current_path, job_name)

    if not os.path.isfile(job_meta_path):
        print(f"Fail to terminate {job_name}, as it does not exist")

    with open(job_meta_path, 'rb') as fin:
        job_meta = pickle.load(fin)

    for vm_ip in job_meta['vms']:
        # os.system(f'scp shutdown.py {job_meta["user"]}{vm_ip}:~/')
        print(f"Shutting down job on {vm_ip}")
        os.system(f"ssh {job_meta['user']}{vm_ip} '/mnt/home/lichenni/anaconda3/envs/oort/bin/python {current_path}/shutdown.py {job_name}'")
try:
    if len(sys.argv)==1:
        # process_cmd('configs/har/conf_test.yml')
        # process_cmd('configs/openimage/conf_test.yml')
        process_cmd('configs/speech/conf_test.yml')
        # process_cmd('configs/stackoverflow/conf_test.yml')
    elif sys.argv[1] == 'submit':
        process_cmd(sys.argv[2])
    elif sys.argv[1] == 'stop':
        terminate(sys.argv[2])
    else:
        print("Unknown cmds ...")
except Exception as e:
    print(f"====Error {e}")
