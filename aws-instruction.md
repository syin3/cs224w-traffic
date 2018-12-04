## Connect to AWS:
`ssh -i /Users/shuyiyin/Desktop/cs230_sxy_NE.pem ubuntu@ec2-18-212-222-134.compute-1.amazonaws.com`

## Transfer a regular file from local to AWS:
`scp -i /Users/shuyiyin/Desktop/cs230_sxy_NE.pem /Users/shuyiyin/Documents/Stanford/Autumn2018/CS224W-Analysis-of-Networks/project/FCRNN ubuntu@ec2-18-212-222-134.compute-1.amazonaws.com:~/cs224w`

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
Type `screen` to enter screen mode, make sure training has started then quit
Type `control+A` then `D` to go back to main screen
Type `screen -r`  to reconnect to the environment\

##
Don't forget to `source activate tensorflow_p36` before training!
