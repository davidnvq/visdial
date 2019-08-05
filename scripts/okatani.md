# Current solution
1. Change dir to save_ckpt



# Notes at Okatani Lab
0. Access servers
```bash
ssh quang@k2
ssh quang@yagi03
```

1. To check the GPU server usages:

```bash
qstat -f -u "*"
```

2. The GPU directories I can access now:

```
quang@yagi02:/media/local_workspace/quang
quang@yagi03:/mnt/local_workspace/quang
quang@yagi10:/mnt/quang
quang@yagi11:/media/local_workspace/quang
quang@yagi13:/media/local_workspace/quang
quang@yagi15:/media/local_workspace/quang
quang@yagi16:/media/local_workspace/quang
quang@yagi17:/mnt/quang 
quang@yagi20:/media/local_workspace/quang
quang@yagi19:/media/local_workspace/quang
```

3. Script `execute.sh` to run qsub on GPU server:

```
#!/usr/bin/env sh
#$-pe gpu 1
#$-l gpu=4 # Using 4 GPUs
#$-j y
#$-cwd
#$-V
#$-o ./log/train.log
#$-q main.q@yagi03.vision.is.tohoku
/home/quanguet/anaconda3/bin/python file.py >> ./log/output.txt
```

4. Some command on k2 with `qsub`:
```
qstat -f -u "*"
qsub execute.sh
qdel JOB_ID
```
```
qlogin -pe gpu 1 -l gpu=1 -q main.q@yagi13
qlogin -pe gpu 8 -l gpu=8 -q main.q@yagi19
qlogin -pe gpu 4 -l gpu=4 -q main.q@yagi11
```

5. Check where is the location of package


# Copy results or Tensorboard
```
scp -i /home/quang/.ssh/key_local_computer -r {log_dir} quanguet@172.16.12.170:/home/quanguet/checkpoints/visdial
tensorboard --logdir /path/to/save_dir --port 8008 and visit localhost:8008
```
```
yagi20: 
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/quang/anaconda3/bin/python train.py --config attn_disc_lstm --project_zip lf_misc_bilstm.zip
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/quang/anaconda3/bin/python /home/quang/repos/visdial/train.py --config attn_gen_lstm 
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/quang/anaconda3/bin/python /home/quang/repos/visdial/train.py --config attn_disc_lstm 
CUDA_VISIBLE_DEVICES=0,1,2,3 /home/quang/anaconda3/bin/python /home/quang/repos/visdial_version1/train.py --config attn_misc_lstm 
yagi19  
```
CUDA_VISIBLE_DEVICES=0 python /home/quang/repos/visdial_v5/train.py \
--config attn_misc_lstm \

# Upload to COMET 
```python
COMET_API_KEY=2z9VHjswAJWF1TV6x4WcFMVss \comet upload ...
```

# Check which job is running
ps -ef | grep `pid`
