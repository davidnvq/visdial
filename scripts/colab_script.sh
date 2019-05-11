#!/usr/bin/env bash

!pip install comet-ml --quiet
!pip install tensorboardX --quiet

!cd /content
!rm -rf /content/visdial
!rm -rf /content/sample_data
!git clone https://github.com/quanguet/visdial.git /content/visdial --quiet

!bash /content/visdial/scripts/colab_train_attn_disc.sh