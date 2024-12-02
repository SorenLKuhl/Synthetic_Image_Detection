#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q gpuv100
##BSUB -q hpc

### -- set the job Name -- 
#BSUB -J SyntheticImageDetectionModel

### -- ask for number of cores (default: 1) -- 
#BSUB -n 4 

### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"

### -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 5GB

### -- set walltime limit: hh:mm -- 
#BSUB -W 06:00 

### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address

### -- send notification at start -- 
##BSUB -B 

### -- send notification at completion -- 
##BSUB -N 

### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o output/Output_%J.out 
#BSUB -e output/Output_%J.err 

# here follow the commands you want to execute with input.in as the input file
path=$(awk -F "=" '/workpath/ {print $2}' config.ini)
venv_path=".venv/bin/activate"
py_path="real_and_fake_faces.py"
source "$path$venv_path"
python "$path$py_path"
# source "/zhome/aa/0/169729/Desktop/Deep_Learning/Synthetic_Image_Detection/.venv/bin/activate"
# python "/zhome/aa/0/169729/Desktop/Deep_Learning/Synthetic_Image_Detection/real_and_fake_faces.py"
