import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torchvision

from networks import encoder as ENC
from networks import decoder as DEC
from networks import layers as LYR
from util.upb_dataset import *
from util.vis import *
from util.warp import *
from flow_rigid import *

import cv2
import argparse
import os

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=12)
parser.add_argument('--num_layers', type=int, default=18)
parser.add_argument('--num_input_images', type=int, default=2)
parser.add_argument('--num_output_channels', type=int, default=2)
parser.add_argument('--num_vis', type=int, default=4)
parser.add_argument('--scales', type=list, default=[0, 1, 2, 3])
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--log_int', type=int, default=100)
parser.add_argument('--vis_int', type=int, default=1000)
parser.add_argument('--num_epochs', type=int, default=20)
parser.add_argument('--scheduler_step_size', type=int, default=15)
parser.add_argument('--log_dir', type=str, default='./logs')
parser.add_argument('--checkpoint_dir', type=str, default='./snapshots')
parser.add_argument('--model_name', type=str, default='default')
parser.add_argument('--load_checkpoint', action='store_true')
parser.add_argument('--dataset', type=str, help="name of the dataset")
args = parser.parse_args()

# create directories
if not os.path.exists(args.log_dir):
	os.mkdir(args.log_dir)

if not os.path.exists(args.checkpoint_dir):
	os.mkdir(args.checkpoint_dir)
	os.makedirs(os.path.join(args.checkpoint_dir, "imgs"))
	os.makedirs(os.path.join(args.checkpoint_dir, "checkpoints"))

# define summary writer
writer = SummaryWriter(log_dir=args.log_dir)

# define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define encoder
encoder = ENC.ResnetEncoder(
	num_layers=args.num_layers, 
	pretrained=True, 
	num_input_images=args.num_input_images
)
encoder.encoder.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
encoder = encoder.to(device)

# define decoder
decoder = DEC.FlowDecoder(
	num_ch_enc=encoder.num_ch_enc, 
	scales=args.scales, 
	num_output_channels=args.num_output_channels, 
	use_skips=True
).to(device)


# define masking network
encoder_mask = ENC.ResnetEncoder(
	num_layers=args.num_layers, 
	pretrained=True, 
	num_input_images=args.num_input_images
)
encoder_mask.encoder.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)
encoder_mask.to(device) 

decoder_mask = DEC.FlowDecoder(
	num_ch_enc=encoder_mask.num_ch_enc,
	scales=args.scales,
	num_output_channels=1,
	use_skips=True
).to(device)

# define ssim
ssim = LYR.SSIM().to(device)

# define rigid flow
rigid_flow = RigidFlow()

# define optimizer
params = list(encoder.parameters())
params += list(decoder.parameters())
params += list(encoder_mask.parameters())
params += list(decoder_mask.parameters())
optimizer = optim.Adam(params, args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, args.scheduler_step_size, 0.1)

# define dataloader
with open(os.path.join("splits", args.dataset, "train_files.txt")) as fin:
	train_filenames = fin.readlines()

with open(os.path.join("splits", args.dataset, "test_files.txt")) as fin:
	test_filenames = fin.readlines()

train_dataset = UPBRAWDataset(
	data_path="./dataset",
	filenames=train_filenames,
	height=args.height,
	width=args.width,
	frame_idxs=[-1, 0],
	num_scales=4,
	is_train=True,
	img_ext="png"
)

test_dataset = UPBRAWDataset(
	data_path="./dataset",
	filenames = test_filenames ,
	height=args.height,
	width=args.width,
	frame_idxs=[-1, 0],
	num_scales=4,
	is_train=False,
	img_ext="png"
)

train_dataloader = DataLoader(
	train_dataset,
	batch_size=args.batch_size,
	shuffle=True,
	num_workers=4,
	drop_last=True,
	pin_memory=True
)

test_dataloader = DataLoader(
	test_dataset,
	batch_size=args.batch_size,
	shuffle=True,
	num_workers=1,
	drop_last=True,
	pin_memory=True
)
test_iter = iter(test_dataloader)

def save_checkpoint(epoch: int, rloss: float):
	state = {
		'epoch': epoch,
		'encoder': encoder.state_dict(),
		'decoder': decoder.state_dict(),
		'encoder_mask': encoder_mask.state_dict(),
		'decoder_mask': decoder_mask.state_dict(),
		'optimizer': optimizer.state_dict(),
		'scheduler': scheduler,
		'rloss': rloss
	}
	path = os.path.join(args.checkpoint_dir, 'checkpoints', args.model_name + ("_%d.pth" % (epoch)))
	torch.save(state, path)


def load_checkpoint():
	path = os.path.join(args.checkpoint_dir, 'checkpoints', args.model_name)
	state = torch.load(path)

	encoder.load_state_dict(state['encoder'])
	decoder.load_state_dict(state['decoder'])
	encoder_mask.load_state_dict(state['encoder_mask'])
	decoder_mask.load_state_dict(state['decoder_mask'])

	optimizer.load_state_dict(state['optimizer'])
	scheduler = scheduler
	return state['epoch'], state['rloss']


def get_rigid_flow(img1: torch.tensor, img2: torch.tensor):
	B, W, H = img1.shape[0], rigid_flow.WIDTH, rigid_flow.HEIGHT
	img1 = F.interpolate(img1, (H, W))
	img2 = F.interpolate(img2, (H, W))

	# get pix coords and then flow
	pix_coords = rigid_flow.get_pix_coords(img1, img2, B)
	rflow = rigid_flow.get_flow(pix_coords, B)
	rflow = rflow.transpose(2, 3).transpose(1, 2)
	rflow = F.interpolate(rflow, (args.height, args.width))
	return rflow.float()


def test_sample():
	global test_iter

	encoder.eval()
	encoder.eval()

	try:
		test_batch = next(test_iter)
	except StopIteration:
		test_iter = iter(test_dataloader)
		test_batch = next(test_iter)

	imgs1 = [data[('color_aug', -1, i)].to(device) for i in range(4)]
	imgs2 = [data[('color_aug', 0, i)].to(device) for i in range(4)]
	rflow = get_rigid_flow(imgs1[0], imgs2[0])
	
	# compute warped image using rigid flow
	wimg2_r = warp(imgs1[0], rflow)
	input = torch.cat((imgs1[0], imgs2[0], wimg2_r), dim=1)


	# compute reprojection loss
	# for the warped image with rigid flow
	ssim_loss = ssim(wimg2_r, imgs2[0]).mean(1, True)
	l1_loss = torch.abs(wimg2_r - imgs2[0]).mean(1, True)
	reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

	with torch.no_grad():
		enc_output = encoder(input, rflow, reprojection_loss)
		dec_output = decoder(enc_output)

	# compute warped image using rigid and dynamic flow
	dflow = dec_output[('flow', 0)]
	flow = dflow + rflow
	wimg2_dr = warp(imgs1[0], flow)

	# compute reprojection loss
	# for the warped image using the rigid and dynamic flow
	ssim_loss = ssim(wimg2_dr, imgs2[0]).mean(1, True)
	l1_loss = torch.abs(wimg2_dr - imgs2[0]).mean(1, True)
	reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss
	
	# compute mask
	with torch.no_grad():
		input = torch.cat([imgs1[0], imgs2[0], wimg2_dr], dim=1)
		enc_mask_output = encoder_mask(input, flow, reprojection_loss)
		dec_mask_output = decoder_mask(enc_mask_output)
		mask = torch.sigmoid(dec_mask_output[('flow', 0)])
		mask = mask.repeat(1, 3, 1, 1)

	# color flow
	flow = flow.cpu() 
	colors = []
	for j in range(args.batch_size):
		color_flow = flow[j].numpy().transpose(1, 2, 0)
		color_flow = flow_to_color(color_flow).transpose(2, 0, 1)
		color_flow = torch.tensor(color_flow).unsqueeze(0).float() / 255
		colors.append(color_flow)
	colors = torch.cat(colors, dim=0)

	img1 = imgs1[0][:args.num_vis].cpu()
	img2 = imgs2[0][:args.num_vis].cpu() 
	wimg2_dr = wimg2_dr[:args.num_vis].cpu()
	mask = mask[:args.num_vis].cpu()
	colors = colors[:args.num_vis]

	imgs = torch.cat([img1, img2, mask * wimg2_dr,  0.5 * (wimg2_dr + img2), 0.5 * (img1 + img2), colors, mask], dim=3)
	imgs = torchvision.utils.make_grid(imgs, nrow=1, normalize=False)
	imgs = (255 * imgs.numpy().transpose(1, 2, 0)).astype(np.uint8)
	cv2.imwrite("./snapshots/imgs/%d.%d.png" % (epoch, i), imgs[..., ::-1])

	encoder.train()
	decoder.train()

if __name__ == "__main__":
	rloss = None
	start_epoch = 0

	if args.load_checkpoint:
		start_epoch, rloss = load_checkpoint()

	for epoch in range(start_epoch, args.num_epochs):
		for i, data in enumerate(train_dataloader):
			# zero grad         
			optimizer.zero_grad()

			# extract data
			imgs1 = [data[('color_aug', -1, i)].to(device) for i in range(4)]
			imgs2 = [data[('color_aug', 0, i)].to(device) for i in range(4)]
			rflow = get_rigid_flow(imgs1[0], imgs2[0])

			# compute warped imaged using rigid flow
			wimg2_r = warp(imgs1[0], rflow)
			input = torch.cat((imgs1[0], imgs2[0], wimg2_r), dim=1)

			# compute reprojection loss
			# for the warped image with rigid flow
			ssim_loss = ssim(wimg2_r, imgs2[0]).mean(1, True)
			l1_loss = torch.abs(wimg2_r - imgs2[0]).mean(1, True)
			reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

			# compute dynamic flow
			enc_output = encoder(input, rflow, reprojection_loss)
			dec_output = decoder(enc_output)

			loss = 0
			for j in args.scales:
				img1 = imgs1[j]
				img2 = imgs2[j]

				# compute warped image using rigid + dynamic flow
				scaled_dflow = dec_output[('flow', j)]
				scaled_rflow = F.interpolate(rflow, (args.height//2**j, args.width//2**j))
				scaled_flow = scaled_dflow + scaled_rflow
				wimg2_dr = warp(img1, scaled_flow)

				# compute reprojection loss
				# for the warped image using the rigid and dynamic flow
				ssim_loss = ssim(wimg2_dr, img2).mean(1, True)
				l1_loss = torch.abs(wimg2_dr - img2).mean(1, True)
				reprojection_loss = 0.85 * ssim_loss + 0.15 * l1_loss

				# smooth loss
				smooth_loss = get_smooth_loss(scaled_dflow[:, :1, :, :], img2)
				smooth_loss += get_smooth_loss(scaled_dflow[:, 1:, :, :], img2)

				# compute masks
				if j == 0:
					input = torch.cat((img1, img2, wimg2_dr), dim=1)
					encoder_mask_output = encoder_mask(input, scaled_flow, reprojection_loss)
					decoder_mask_output = decoder_mask(encoder_mask_output) 

				# compute maks loss
				mask = decoder_mask_output[('flow', j)]
				mask = torch.sigmoid(mask)
				weighting_loss = nn.BCELoss()(mask, torch.ones(mask.shape).cuda())

				# mask reprojection loss
				reprojection_loss = mask * reprojection_loss

				# total loss
				loss += (reprojection_loss.mean() + 0.2 * weighting_loss +  0.01 * smooth_loss) / 2**j

			# backward step
			loss.backward()
			optimizer.step()

			# compute running loss
			rloss = loss.item() if rloss is None else 0.99 * rloss + 0.01 * loss.item()

			# log interval
			if i % args.log_int == 0:
				it = epoch * (len(train_dataset) // args.batch_size) + i
				writer.add_scalar("Loss", rloss, it)
				print("Epoch: %d, Batch: %d, Loss: %.4f" % (epoch, i, rloss))

			# visualization interval
			if i % args.vis_int == 0:
				test_sample()

		# scheduler step
		scheduler.step()

		# save model
		save_checkpoint(epoch, rloss)

	# export scalar data to JSON for external processing
	writer.close()
