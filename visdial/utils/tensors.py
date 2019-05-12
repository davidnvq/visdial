def move_to_cuda(batch, device):
	for key in batch:
		batch[key] = batch[key].to(device)
	return batch