# Lowes Product Classifier

## About
The schedule can be found here: [Schedule](https://docs.google.com/spreadsheets/d/191MEJAiZgz2XXsHFm9EH_HufX-Z5-7UkNJ-RQGngFo8/edit?usp=sharing)

The project details the testing of several machine learning models for product classification.
The following models were used:
 - InceptionResNetV2
 - NASNetMobile
 - VGG19
 - SqueezeNet
 - MobileNet
 
 The current project uses a custom dataset of approx 1200+ images of light bulbs.
 The images can be found at our google drive. Please contact Josiah for access.

## Contents
* [Links](#links)



## Setup
Go to terminal and execute: `python setup.py install`

## Running: Client
<<<<<<< HEAD

=======
1. `python3 lowes-product-classifier/josiah_testing/demo_v2/RunDemo.py`


## Running: Server
These instructions are made with the UNCC server in mind. It also is intended that you use Linux, or Cmder
1. Log into the server via: `ssh [username]@hpc.uncc.edu` using your username that you were approved for.
    1.   You will have to input your password and possibly a 2 step verification. Note, if you are a student 
    using your student id, you will be inputting your current student login password.
2. Typing `ls` should show the current directory that you are allowed to work in.
    1. If you are using the UNCC server, you should see `master_data` and `workspace`
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
9. If you need to view the required modules for your script you can call command `module avail`

## Links


* [Markup Instructions](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)
* [Python Package Setup](https://packaging.python.org/tutorials/packaging-projects/)
* [Bayesian Inference Tensorflow](https://github.com/CamDavidsonPilon/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/blob/master/Chapter1_Introduction/Ch1_Introduction_TFP.ipynb)
## Notes:
