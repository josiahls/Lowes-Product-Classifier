# Student Behavior Tracker

The goal of this application is to give behavior scores on students across a span of time.
We used a 360 degree camera for collecting the video for this application.
We used active learning for the classroom setting. 

The final goal is to be able to track students, and quickly recognize students that may be under-performing.

### Contents

3. [Usage](#usage)

### Usage
#### Client Execution
text

#### Server Execution
These instructions are made with the UNCC server in mind. It also is intended that you use Linux, or Cmder
1. Log into the server via: `ssh [username]@hpc.uncc.edu` using your username that you were approved for.
    1.   You will have to input your password and possibly a 2 step verification. Note, if you are a student 
    using your student id, you will be inputting your current student login password.
2. Typing `ls` should show the current directory that you are allowed to work in.
    1. If you are using the UNCC server, you should see `master_data` and `workspace`
4. Be sure to execute:
    1. `mkdir weights`
    2. `wget https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5`
3. Jobs can be executed via: `qsub face.sh`
    1. If you get the error: `qsub:  script is written in DOS/Windows text format` you can resolve this using 
    `dos2unix face.sh`
4. `qstat` shows all current jobs while `qstat -u [username]` shows the job for the current user
5. If you `cd /[projectname]/log` then use `nano [log file name]` you can view the log output of the server.
6. If you would like to code the project locally, and if you are using linux Ubuntu 16.04, then you can mount the file 
system on the server via: `sshfs [username]@hpc.uncc.edu: [local_folder_to_mount_to] -o nonempty`
7. If you want to add more videos you can do so without a file transfer application by using: `scp -v ./[path to the video you want to send, you can also use * to send all videos or files in the directory] 
[username]@hpc.uncc.edu:/users/[username]/[path to the folder you want to store the videos]`
8. If you need to delete a job, then execute `qdel [JOBID]`


### Note:
* If you are installing tensorflow for python3.7 use pip3 install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.10.1-py3-none-any.whl
* If you want to visualize tensorflow models you can call `tensorboard --logdir=./run_logs/[a specific folder if you want, otherwise leave blank] --host=127.0.0.1`
* python download_gdrive.py GoogleFileID /path/for/this/file/to/download/file.zip
