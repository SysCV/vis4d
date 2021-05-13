from sort_graph import *
import torch
from PIL import Image, ImageDraw, ImageColor
import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    torch.set_printoptions(precision=2)
    frames: List[torch.FloatTensor] = []
    det_bbox = torch.Tensor([10.0, 10.0, 20.0, 30.0])
    for i in range(10):
        frames.append(det_bbox)
        det_bbox = det_bbox + torch.Tensor([5.0, 0.0, 5.0, 0.0])
    tracks = []

    kf = KalmanFilter()

    kalman_state, covariance = kf.initiate(
        xyxy_to_xyah(frames[0].unsqueeze(0)).squeeze()
    )
    track_bbox = xyah_to_xyxy(kalman_state[:4].unsqueeze(0)).squeeze()
    tracks.append(
        dict(
            track_bbox=track_bbox,
            track_vel=kalman_state[4:],
            track_cov=covariance,
        )
    )

    for frame_id, det_bbox in enumerate(frames):
        if frame_id == 0:
            continue
        print("#" * 50)
        print("frame: ", frame_id)
        print("#" * 50)
        track_bbox = tracks[frame_id - 1]["track_bbox"]
        track_vel = tracks[frame_id - 1]["track_vel"]
        track_cov = tracks[frame_id - 1]["track_cov"]
        print("track bbox: ", track_bbox)
        print("detection bbox: ", det_bbox)

        image = Image.new("RGB", (100, 50), "white")
        draw = ImageDraw.Draw(image)
        # draw track bbox
        draw.rectangle(
            track_bbox.cpu().numpy().tolist(),
            outline=ImageColor.getrgb("black"),
        )
        # draw detection bbox
        draw.rectangle(
            det_bbox.cpu().numpy().tolist(),
            outline=ImageColor.getrgb("red"),
        )

        kalman_state = torch.cat(
            (xyxy_to_xyah(track_bbox.unsqueeze(0)).squeeze(), track_vel)
        )
        kalman_state, covariance = kf.predict(kalman_state, covariance)

        pred_bbox = xyah_to_xyxy(kalman_state[:4].unsqueeze(0)).squeeze()
        print("pred_bbox:  ", pred_bbox)
        # draw prediction bbox
        draw.rectangle(
            pred_bbox.cpu().numpy().tolist(),
            outline=ImageColor.getrgb("blue"),
        )
        update_state, update_covariance = kf.update(
            kalman_state,
            covariance,
            xyxy_to_xyah(det_bbox.unsqueeze(0)).squeeze(),
        )

        updated_bbox = xyah_to_xyxy(update_state[:4].unsqueeze(0)).squeeze()
        print("updated bbox: ", updated_bbox)
        # draw updated bbox
        draw.rectangle(
            updated_bbox.cpu().numpy().tolist(),
            outline=ImageColor.getrgb("green"),
        )
        tracks.append(
            dict(
                track_bbox=updated_bbox,
                track_vel=update_state[4:],
                track_cov=update_covariance,
            )
        )
        plt.imshow(np.asarray(image))
        plt.title(
            "black: track   red: detection   blue: pred   green: updated"
        )
        plt.savefig(
            "/home/daniel/Desktop/sorttest/" + str(frame_id) + ".png", dpi=300
        )
