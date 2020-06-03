import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import torchvision

import PIL.Image as pil
import cv2
import argparse
import matplotlib.pyplot as plt

from networks import encoder as ENC
from networks import decoder as DEC
from networks import layers as LYR
from flow_rigid import *
from util.vis import *
from util.warp import *

# define parser
parser = argparse.ArgumentParser()
parser.add_argument('--num_layers', type=int, default=18)
parser.add_argument('--num_input_images', type=int, default=2)
parser.add_argument('--num_output_channels', type=int, default=2)
parser.add_argument('--scales', type=list, default=[0, 1, 2, 3])
parser.add_argument('--height', type=int, default=256)
parser.add_argument('--width', type=int, default=512)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--video_path', type=str, default='../../upb_dataset/faf0f08990fd4d27.mov')
parser.add_argument('--save_gif', type=str, default='./teaser/faf0f08990fd4d27.gif')
parser.add_argument('--checkpoint_dir', type=str, default='./snapshots')
parser.add_argument('--model_name', type=str, default='default_complex.pth')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# define rigid flow object
rigid_flow = RigidFlow()

# define ssim
ssim = LYR.SSIM().to(device)

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

if not os.path.exists("./teaser"):
    os.mkdir("teaser")

def load_checkpoint():
    global encoder
    global decoder

    path = os.path.join(args.checkpoint_dir, 'checkpoints', args.model_name)
    state = torch.load(path)
    encoder.load_state_dict(state['encoder'])
    decoder.load_state_dict(state['decoder'])
    encoder.eval()
    decoder.eval()


def procees_frame(frame: np.array):
    frame = frame[...,::-1].astype(np.float)
    if frame.max() > 1:
        frame /= 255.

    frame = torch.tensor(frame.transpose(2, 0, 1)).unsqueeze(0)
    frame = frame.to(device)
    frame = F.interpolate(frame, (args.height, args.width))
    return frame.float()


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


def pipeline(prev_frame: np.array, frame: np.array):
    prev_frame = procees_frame(prev_frame)
    frame = procees_frame(frame)

    # get rigid flow
    rflow = get_rigid_flow(prev_frame, frame)

    # warped image
    wframe = warp(prev_frame, rflow)

    # compute error map
    ssim_loss = ssim(wframe, frame).mean(1, True)
    l1_loss = torch.abs(wframe - frame).mean(1, True)
    err_map = 0.85 * ssim_loss + 0.15 * l1_loss

    # get dynamic flow correction
    input = torch.cat([prev_frame, frame, wframe], dim=1)
    with torch.no_grad():
        enc_output = encoder(input, rflow, err_map)
        dec_output = decoder(enc_output)
        dflow = dec_output[('flow', 0)]

    flow = dflow + rflow
    flow = flow.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    color_flow = flow_to_color(flow)
    rflow = rflow.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    color_rflow = flow_to_color(rflow)
    return np.concatenate([color_flow, color_rflow], axis=1)[..., ::-1]

def test_video():
    all_frames = []
    cap = cv2.VideoCapture(args.video_path)
    ret, prev_frame = cap.read()
    prev_frame = prev_frame[:320, ...]

    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame = frame[:320]
        flow = pipeline(prev_frame, frame)
        prev_frame = frame

        frame = cv2.resize(frame, (512, 256))
        flow = cv2.resize(flow, (1024, 256))
        frame_flow = np.concatenate([frame, flow], axis=1)

        # Display the resulting frame
        all_frames.append(frame_flow)
        cv2.imshow('frame', frame_flow)
        if cv2.waitKey(0) & 0xFF == ord('q'):
           break

    # save gif
    all_frames = [pil.fromarray(frame[...,::-1]) for frame in all_frames]
    all_frames[0].save(
        args.save_gif, 
        save_all=True, 
        append_images=all_frames[1:],
        duration=175
    )

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    load_checkpoint()
    test_video()
