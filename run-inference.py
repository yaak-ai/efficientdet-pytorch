import time
import argparse
from queue import Queue
from threading import Thread
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


def seek_fn(
    video_file_path, queue_frames, queue_tensor, input_tensor, frame_count, batch_size
):
    """
    Iterator over the video frames
    Seek and you shall find it — Harsi, Circa 2020
    """

    # Opena a video reader
    print(f"Opened file {video_file_path}")
    src_reader = vreader(video_file_path.as_posix())

    pbar = tqdm(src_reader, total=frame_count, ascii=True, unit="frames")

    tensor_batch = []
    frame_batch = []
    for frame in pbar:
        if len(frame_batch) < batch_size:
            frame_batch.append(frame.copy())
            tensor_batch.append(preprocess(frame, input_tensor))
            continue
        queue_frames.put(frame_batch)
        queue_tensor.put(torch.cat(tensor_batch, dim=0).float().cuda())
        frame_batch, tensor_batch = [frame.copy()], [preprocess(frame, input_tensor)]

    # last batch
    queue_frames.put(frame_batch)
    queue_tensor.put(torch.cat(tensor_batch, dim=0).float().cuda())
    #
    src_reader.close()
    # TODO : Harsimrat — gehacked here lolz
    queue_frames.put(None)
    queue_tensor.put(None)


def get_metadata(video_file_path):

    metadata = ffprobe(video_file_path.as_posix())
    codec = metadata["video"]["@codec_name"]
    # Get a frame count H265 videos don't have "nb_frames" key in ffprobe
    nb_frames = int(metadata["video"]["@nb_frames"]) if codec == "h264" else -1
    fps = metadata["video"]["@r_frame_rate"].split("/")[0]
    height = int(metadata["video"]["@height"])
    width = int(metadata["video"]["@width"])

    m = METADATA(codec=codec, fps=fps, nb_frames=nb_frames, width=width, height=height)

    return m


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


def inference_fn(model, model_frame_queue, detections_queue, img_info):

    """
    Run inference after fetching image tensor from model_frame_queue
    write back detections_queue. Frame is expected to be pre-processed
    """

    with torch.no_grad():

        while True:
            frame = model_frame_queue.get()
            if frame is None:
                break
            output = model(frame, img_info=None)
            detections_queue.put(output)
    # TODO : Harsi gehacked here again for eof sig
    detections_queue.put(None)


def redact_fn(queue_frame, queue_detection, video_file_path, fps, threshold):

    """
    Fetches image from frame_queue, detections from detections_queue
    Blurs/Draws prediction and writes on stream
    """

    frame_count = 0
    obj_count = 0
    print(f"Opening video writer for {video_file_path}")

    class_names = ["Vehicle registration plate", "Human face"]

    dst_writer = FFmpegWriter(
        video_file_path.as_posix(),
        outputdict={
            "-vcodec": FFMPEG_VCODEC,
            "-r": fps,
            "-preset": FFMPEG_PRESET,
            "-vtag": FFMPEG_VTAG,
        },
    )

    while True:
        frames = queue_frame.get()
        objs = queue_detection.get()
        if frames is None:
            break

        objs = objs.cpu().numpy()
        for frame, obj in zip(frames, objs):
            (h, w) = frame.shape[:2]
            # 512 is the expected network image input size
            scaling = np.array([w / 512, h / 512, w / 512, h / 512])
            obj = [b for b in obj if b[4] > threshold[int(b[5] - 1)]]
            boxes = [b[:4] * scaling for b in obj]
            scores = [b[4] for b in obj]
            classes = [class_names[int(b[5] - 1)] for b in obj]
            boxes = [list(map(int, b)) for b in boxes]
            detections = list(zip(classes, scores, boxes))
            frame = drawing.redact_regions(frame, detections)
            frame = drawing.draw_rectangle(frame, boxes)
            dst_writer.writeFrame(frame)
            frame_count += 1
            obj_count += len(detections)
            # [x_min, y_min, x_max, y_max, score, class]

    dst_writer.close()
    print(f"Found {obj_count} redacted objects in {frame_count} frames")


def redact(
    model_name,
    weights_path,
    source_vid_path,
    dst_vid_path,
    results_json,
    threshold,
    batch_size=1,
):

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

    param_count = sum([m.numel() for m in model.parameters()])
    print("Model %s created, param count: %d" % (model_name, param_count))

    model = model.cuda()
    model.eval()

    for params in model.parameters():
        params.requires_grad = False

    queue_frames = Queue()
    queue_tensor = Queue(maxsize=1)
    queue_detection = Queue(maxsize=1)

    metadata = get_metadata(Path(source_vid_path))
    video_scale = min(
        model_config["image_size"][0] / metadata.height,
        model_config["image_size"][1] / metadata.width,
    )

    img_info = {
        "img_idx": 0,
        "img_size": torch.tensor([[metadata.width, metadata.height]]).cuda(),
        "img_scale": torch.tensor([[1.0 / video_scale]]).cuda(),
    }

    print(
        f"{source_vid_path} {metadata.codec} {metadata.fps} fps {metadata.nb_frames} frames"
    )

    print(img_info)

    if metadata.nb_frames == 0:
        print(f"Empty file ? {source_vid_path}")
        return

    print(f"Redacted video WIDTHxHEIGHT {metadata.width}x{metadata.height}")

    # caches tensor for re-use for the whole video
    input_tensor = torch.zeros(
        (model_config["image_size"][0], model_config["image_size"][1], 3),
        dtype=torch.float32,
    ).cuda()

    t0 = Thread(
        target=seek_fn,
        args=(
            Path(source_vid_path),
            queue_frames,
            queue_tensor,
            input_tensor,
            metadata.nb_frames,
            batch_size,
        ),
    )
    t0.start()
    t1 = Thread(
        target=inference_fn,
        args=(model, queue_tensor, queue_detection, img_info),
    )
    t1.start()
    t2 = Thread(
        target=redact_fn,
        args=(
            queue_frames,
            queue_detection,
            Path(dst_vid_path),
            metadata.fps,
            threshold,
        ),
    )
    t2.start()
    t2.join()


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Run PIIL model on Yaak Drive Data")
    parser.add_argument("-m", "--model", dest="model", help="PII Model path")
    parser.add_argument("-w", "--weights", dest="weights", help="PII Model config")
    parser.add_argument("-s", "--source_vid", dest="video", help="Video path")
    parser.add_argument("-r", "--results-json", dest="json", help="Json path")
    parser.add_argument("-d", "--dest_vid", dest="video_out", help="Output Video path")
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
        args.video,
        args.video_out,
        args.json,
        args.threshold,
        args.batch_size,
    )
