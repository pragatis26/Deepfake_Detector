"""
utils/video_utils.py — Extract uniformly-spaced frames from a video file
"""

import os
import uuid
import cv2


def extract_frames(video_path: str, max_frames: int = 20, output_dir: str = 'static/uploads') -> list:
    """
    Extract up to `max_frames` evenly-spaced frames from a video.

    Args:
        video_path:  Path to the video file
        max_frames:  Maximum number of frames to extract
        output_dir:  Directory to save frame images

    Returns:
        List of file paths to saved frame images
    """
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[VideoUtils] Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)

    if total_frames <= 0:
        cap.release()
        return []

    # Pick evenly-spaced frame indices
    step = max(1, total_frames // max_frames)
    frame_indices = list(range(0, total_frames, step))[:max_frames]

    saved_paths = []
    prefix = uuid.uuid4().hex[:8]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        fname = f"frame_{prefix}_{idx:05d}.jpg"
        fpath = os.path.join(output_dir, fname)
        cv2.imwrite(fpath, frame)
        saved_paths.append(fpath)

    cap.release()
    print(f"[VideoUtils] Extracted {len(saved_paths)}/{max_frames} frames from {os.path.basename(video_path)}")
    return saved_paths


def get_video_info(video_path: str) -> dict:
    """Return basic metadata about a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}
    info = {
        'fps':          cap.get(cv2.CAP_PROP_FPS),
        'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        'width':        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'height':       int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'duration_sec': int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1))
    }
    cap.release()
    return info
