## Connect to AWS:
`ssh -i /Users/shuyiyin/Desktop/cs230_sxy_NE.pem ubuntu@ec2-18-212-78-189.compute-1.amazonaws.com`

## Transfer a regular file from local to AWS:
`scp -i /Users/shuyiyin/Desktop/cs230_sxy_NE.pem /Users/shuyiyin/Documents/Stanford/Autumn2018/CS224W-Analysis-of-Networks/project/FCRNN/exec.sh ubuntu@ec2-18-212-78-189.compute-1.amazonaws.com:~/cs224w/FCRNN/`

## Download a regular file from AWS to local:
`scp -i /path/my-key-pair.pem ubuntu@ec2-18-212-222-134.compute-1.amazonaws.com:~/SampleFile.txt ~/SampleFile2.txt`

## Transfer a local directory to AWS:
`scp -r -i /Users/shuyiyin/Desktop/cs230_sxy_NE.pem /Users/shuyiyin/Documents/Stanford/Autumn2018/CS224W-Analysis-of-Networks/project/FCRNN ubuntu@ec2-18-212-222-134.compute-1.amazonaws.com:~/cs224w`

## Download a directory/folder from AWS to local:
`scp -r -i /Users/shuyiyin/Desktop/cs230_sxy_NE.pem ubuntu@ec2-18-212-222-134.compute-1.amazonaws.com:~/cs224w/FCRNN ~/SampleFile.txt`

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
