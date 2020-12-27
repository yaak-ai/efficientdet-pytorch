import argparse
from collections import namedtuple
from pathlib import Path

import numpy as np
import cv2
from tqdm import tqdm

import torch
from skvideo.io import ffprobe, vreader, FFmpegWriter

from effdet import create_model
from effdet import drawing

METADATA = namedtuple("metadata", "codec fps nb_frames width height")

# H265 vcodec
FFMPEG_VCODEC = "libx265"
# H265 encode speed
FFMPEG_PRESET = "ultrafast"
# Video tag for quicktime player
FFMPEG_VTAG = "hvc1"

IMAGENET_DEFAULT_MEAN = torch.tensor(
    np.array((0.485, 0.456, 0.406)) * 255, dtype=torch.float32
)
IMAGENET_DEFAULT_STD = torch.tensor(
    np.array((0.229, 0.224, 0.225)) * 255, dtype=torch.float32
)


def preprocess(frame, input_tensor):

    """
    Model specfic frame pre-processing
    1. Tesor from numpy array
    2. Mean subtraction
    3. HWC -> CHW
    4. CHN -> NCHW
    """
    size = input_tensor.shape[:2]

    # Throwing all pre-precessing onto device if possible
    frame = cv2.resize(frame, size, interpolation=cv2.INTER_LINEAR) + 0.0
    frame = torch.from_numpy(frame)
    frame -= IMAGENET_DEFAULT_MEAN
    frame /= IMAGENET_DEFAULT_STD
    frame = frame.permute([2, 0, 1])

    return frame.unsqueeze(0)


def redact(model_name, weights_path, source_image, dst_image, threshold, batch_size=1):

    torch.backends.cudnn.benchmark = True

    model = create_model(
        model_name,
        bench_task="predict",
        num_classes=2,
        pretrained=True,
        redundant_bias=None,
        checkpoint_path=weights_path,
        checkpoint_ema=False,
    )

    model_config = model.config
    class_names = ["Vehicle registration plate", "Human face"]

    input_tensor = torch.zeros(
        (model_config["image_size"][0], model_config["image_size"][1], 3),
        dtype=torch.float32,
    )

    param_count = sum([m.numel() for m in model.parameters()])
    print("Model %s created, param count: %d" % (model_name, param_count))

    model = model
    model.eval()

    for params in model.parameters():
        params.requires_grad = False

    frame = cv2.imread(source_image)[:, :, (2, 1, 0)]

    frame_tensor = preprocess(frame, input_tensor).float()

    with torch.no_grad():
        objs = model(frame_tensor, img_info=None)

    obj = objs.cpu().numpy()[0]

    (h, w) = frame.shape[:2]
    size = model_config["image_size"][0]
    # 512 is the expected network image input size
    scaling = np.array([w / size, h / size, w / size, h / size])
    obj = [b for b in obj if b[4] > threshold[int(b[5] - 1)]]
    boxes = [b[:4] * scaling for b in obj]
    boxes = [list(map(int, b)) for b in boxes]
    for box in boxes:
        w, h = box[2] - box[0], box[3] - box[1]
        print(f"{box[0]}, {box[1]} — W x H : {w} x {h}")

    frame = drawing.draw_rectangle(frame, boxes)
    cv2.imwrite(dst_image, frame[:, :, (2, 1, 0)])

    cv2.imshow("img", frame[:, :, (2, 1, 0)])
    cv2.waitKey(0)


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run PIIL model on Yaak Drive Data")
    parser.add_argument("-m", "--model", dest="model", help="PII Model path")
    parser.add_argument("-w", "--weights", dest="weights", help="PII Model config")
    parser.add_argument("-s", "--source_image", dest="image", help="Image Image path")
    parser.add_argument(
        "-d", "--dest_image", dest="image_out", help="Output Image path"
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        help="Batch size",
        default=1,
        type=int,
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="threshold",
        default=[0.1, 0.2],
        type=float,
        nargs=2,
        help="Model Threshold",
    )

    args = parser.parse_args()

    redact(
        args.model,
        args.weights,
        args.image,
        args.image_out,
        args.threshold,
        args.batch_size,
    )
