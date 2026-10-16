"""Shared camera-enumeration helper for the demo scripts.

OpenCV only exposes a numeric index for each camera, and that index's order
does not always match what you'd expect - on a laptop with a built-in
FaceTime/webcam and an external USB camera, the external one is often
--source 0, pushing the built-in camera to --source 1 or higher. --list-cameras
opens each index in turn and reports what it finds, plus (on macOS) the
camera names from System Settings, so you can identify the right index
before running a demo for real.
"""
import contextlib
import os
import platform
import subprocess

import cv2


@contextlib.contextmanager
def _quiet_stderr():
    """Silence the camera backend's native (non-Python) stderr warnings.

    Indices past the last real camera fail to open, and OpenCV's camera
    backends (AVFoundation on macOS, FFmpeg elsewhere) print that failure
    straight to the C-level stderr fd - below what cv2's own Python logging
    controls. Redirecting fd 2 is the only way to quiet it.
    """
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        saved = os.dup(2)
        os.dup2(devnull, 2)
    except OSError:
        yield
        return
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(devnull)
        os.close(saved)


def list_cameras(max_index=6):
    """Probe camera indices 0..max_index-1 and report what opens.

    Opening every index is the only portable way to discover cameras with
    OpenCV - there is no cross-platform "list devices" call.
    """
    found = []
    with _quiet_stderr():
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            opened = cap.isOpened()
            width = height = 0
            if opened:
                ok, _ = cap.read()
                opened = opened and ok
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            if opened:
                found.append({"index": i, "width": width, "height": height})
    return found


def macos_camera_names():
    """Camera names from System Settings, in the order macOS lists them.

    This is best-effort context, not a guaranteed match to OpenCV's index
    order - macOS and OpenCV enumerate devices independently.
    """
    if platform.system() != "Darwin":
        return []
    try:
        out = subprocess.run(
            ["system_profiler", "SPCameraDataType"],
            capture_output=True, text=True, timeout=5,
        ).stdout
    except Exception:
        return []
    names = []
    for line in out.splitlines():
        line = line.strip()
        if line.endswith(":") and not line.lower().startswith(("model id", "unique id")):
            names.append(line[:-1])
    return names


def print_camera_list(max_index=6):
    cams = list_cameras(max_index)
    print(f"Cameras OpenCV can open (probed indices 0-{max_index - 1}):")
    if not cams:
        print("  none found. Check your OS camera permissions for this terminal, "
              "or try --list-cameras --max-index 10 in case a camera sits higher.")
    for c in cams:
        print(f"  --source {c['index']}  ->  {c['width']}x{c['height']}")

    names = macos_camera_names()
    if names:
        print()
        print("macOS System Settings lists these cameras, in its own order "
              "(for reference only - it does not have to match the indices above):")
        for n in names:
            print(f"  - {n}")

    print()
    print("Run a demo with the --source index that gave the resolution you "
          "expect for the camera you want, e.g. --source 1.")
