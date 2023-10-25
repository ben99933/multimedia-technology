import numpy as np
import gc
import cv2
from tqdm import tqdm
import math
from math import log
from decimal import Decimal as double

width: int = 1024
height: int = 1024
fps = 24
second = 30
total_frames = fps * second


def iteration(frame):
    return (int)(30 + 4.7 * 10 * math.sqrt(100 * (frame / total_frames)))
    # return 50


def mandelbrot_set(iteration: int, resolution: tuple, zoom_factor: float) -> np.ndarray:
    def build_grid() -> np.ndarray:
        lowerBound = np.float64(-2 / zoom_factor)
        upperBound = np.float64(2 / zoom_factor)
        # 這個座標點是在網路找的 不然要調到瘋掉
        # 然後我不知道為啥他的座標是倒過來的
        offsetX = np.float64(0)
        offsetY = np.float64(-1.419751266)
        xline = np.linspace(np.float64(lowerBound + offsetX), np.float64(upperBound + offsetX), resolution[0])
        yline = np.linspace(np.float64(lowerBound + offsetY), np.float64(upperBound + offsetY), resolution[1])
        x, y = np.meshgrid(yline, xline)
        return x + 1j * y

    c = build_grid()
    # print(c.shape)
    # z = (0 + 0j) * np.ones((resolution[0], resolution[1]))
    z = np.zeros(c.shape, dtype=np.complex128)
    background = np.zeros((resolution[0], resolution[1], 3))
    lastmask = np.ones((resolution[0], resolution[1], 3))
    frame = np.zeros((resolution[0], resolution[1], 3))

    mask = None

    for i in range(iteration+1):
        mask = (abs(z) <= 2).astype(np.float32)
        z = z * mask
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mask = lastmask * mask

        framei = (lastmask - mask) * np.array([255, 30+225*(i/iteration), 30+225*(i/iteration)])
        background += framei
        lastmask = mask

        frame = background.astype(np.uint8)

        z = z * z + c

        # free mem
        del mask, framei

        if (i % 10 == 0 or i == iteration - 1): gc.collect()

    return frame


def main():
    frames = []
    scale = 1
    # zoom_factor = 1.005
    zoom_factor = math.pow(6378723189, 1 / (fps * second))
    for i in tqdm(range(total_frames), position=0, leave=True):
        scale = scale * zoom_factor  # per frame scale 2.206%
        frame = mandelbrot_set(iteration=iteration(i), resolution=(width, height), zoom_factor=scale)
        frames.append(frame)

    # make video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter("mandelbrot_animation.mp4", fourcc, fps, (width, height))

    for frame in frames:
        video_writer.write(frame)
    video_writer.release()


if __name__ == "__main__":
    main()
