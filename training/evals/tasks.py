from invoke import task
import os


@task
def clean(c):
    c.run("rm -rf ~/pymp-*")

@task
def cifar10(c):
    c.run("python3 manager.py submit configs/cifar10/conf.yml")

@task
def openImg(c):
    c.run("python3 manager.py submit configs/openimage/conf.yml")

@task
def speech(c):
    c.run("python3 manager.py submit configs/speech/conf.yml")

@task
def blog(c):
    c.run("python3 manager.py submit configs/stackoverflow/conf.yml")

@task 
def kill(c):
    c.run("nvidia-smi | grep python | awk '{print $5}' | xargs kill -9")

@task
def killall(c):
    c.run("nvidia-smi | grep -E 'python|/usr/lib/xorg/Xorg|/usr/bin/gnome-shell' | awk '{print $5}' | xargs sudo kill -9 ")

@task
def killzombie(c):
    command = """pids=$(fuser -v /dev/nvidia* 2>/dev/null | grep -oP '\\d+')
for pid in $pids; do
    kill -9 $pid
done"""
    c.run(command)



@task
def plot(c, path_random, path_oort):
    current_directory = os.getcwd()
    path_random = current_directory + "/log/logs/cifar10/" + path_random + "/aggregator/training_perf"
    path_oort = current_directory + "/log/logs/cifar10/" + path_oort + "/aggregator/training_perf"
    c.run(f"python plot_perf.py {path_random} {path_oort}")


@task
def pl(c,td):
    c.run(f"python plot.py {td}")