import torch
import argparse
from visdial.utils.model_utils import evaluate
parser = argparse.ArgumentParser()
parser.add_argument("--cpath", default="/home/quang/checkpoints/s09/config.json")
parser.add_argument("--wpath", default='/home/quang/checkpoints/s09/checkpoint_11.pth')
parser.add_argument("--ckpt_name", default="no_ft_ckpt_11")
parser.add_argument("--split", default="val")
parser.add_argument("--decoder_type", default='disc')
parser.add_argument("--device", default="cuda:0")

# For reproducibility.
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# =============================================================================
#   INPUT ARGUMENTS AND CONFIG
# =============================================================================



args = parser.parse_args()

evaluate(cpath=args.cpath, wpath=args.wpath, device='cuda',
		 split=args.split, decoder_type=args.decoder_type,
		 ckpt_name=args.ckpt_name)