## Connect to AWS:
`ssh -i /Users/.../xxx.pem ubuntu@yyy.amazonaws.com`

## Transfer a regular file from local to AWS:
`scp -i /Users/.../xxx.pem /Users/.../zzz.file`

## Download a regular file from AWS to local:
`scp -i /Users/.../xxx.pem ubuntu@yyy.amazonaws.com:~/SampleFile.txt ~/SampleFile2.txt`

## Transfer a local directory to AWS:
`scp -r -i /Users/.../xxx.pem /Users/.../zzz ubuntu@yyy.amazonaws.com:~/cs224w`

## Download a directory/folder from AWS to local:
`scp -r -i /Users/.../xxx.pem buntu@yyy.amazonaws.com:~/.../zzz/ /Users/.../model`

## Inspect and save changes in Terminal
view content: `vi file`
start changing: `i`
strop changing: `esc`
save and exit: `:wq` (type the colon first)

## Use of screen in AWS EC2
Type `screen` to start a new screen
Type `screen -ls` to enter screen mode, make sure training has started then quit
Type `control+A` then `D` to go back to main screen
Type `screen -r 2354.pts-8.ip-172-31-22-94`  to reconnect to the environment
Type `screen -X -S 1740.pts-0.ip-172-31-22-94 kill` to kill a certain screen
Type `screen -d -r 1548.pts-0.ip-172-31-22-94` if a screen exists but cannot be resumed, very likely it is still attached, so `-d` detaches it first.

## Activate source
Don't forget to `source activate tensorflow_p36` before training!

## Connect to Google Cloud
`gcloud compute ssh xxx@instance`

## Transfer a regular file from local to Google Cloud
`gcloud compute scp /Users/.../xxx.npz root@instance:/.../yyy/`

## Download a directory from Google Cloud

`gcloud compute scp --recurse root@instance:~/.../xxx/ /Users/.../yyy/`

## Activate source environment
`source activate p36`

## Set up the Conda environment

Type `sh /home/shared/setup.sh` then `source ~/.bashrc` to set up
Type `conda --version` to check current version of conda
Type `conda info --envs`
Type `conda create -n p36 python=3.6 anaconda` to build an Anaconda environment
Type `source activate p36` to activate the environment and to install localized packages
Type `source deactivate` to exit the current environment
