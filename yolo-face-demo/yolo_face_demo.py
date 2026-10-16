"""Draw face detections from a local YOLO checkpoint on a live webcam feed.

Author: Xu Chen
"""

import argparse
from pathlib import Path
import sys

from camera_utils import print_camera_list


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--list-cameras", action="store_true",
                         help="print which --source index opens which camera, then exit")
    parser.add_argument("--max-index", type=int, default=6,
                         help="highest camera index to probe for --list-cameras")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (0 to 1)")
    parser.add_argument("--check", action="store_true", help="Test model inference without opening the camera")
    args = parser.parse_args()
    if args.list_cameras:
        print_camera_list(args.max_index)
        return
    if not 0 <= args.conf <= 1:
        parser.error("--conf must be between 0 and 1")

    model_path = Path(__file__).resolve().parent / "yolov8n-face.pt"
    if not model_path.is_file():
        parser.error(f"Place an Ultralytics-compatible face checkpoint at {model_path}")

    import cv2
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    if set(str(name).lower() for name in model.names.values()) != {"face"}:
        raise RuntimeError("This demo requires a face-only checkpoint with class name 'face'.")
    if args.check:
        import numpy as np
        model.predict(np.zeros((480, 640, 3), dtype=np.uint8), conf=args.conf,
                      verbose=False, save=False, device="cpu")
        print("PASS: face model loaded and inference completed; camera was not opened.")
        return
    cap = cv2.VideoCapture(args.source)
    try:
        if not cap.isOpened():
            raise RuntimeError(
                "Could not open camera. Check camera permissions, close other camera "
                "apps, or run --list-cameras to find the right --source index."
            )

        print("Focus the camera window and press q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera.")

            results = model.predict(frame, conf=args.conf, verbose=False, save=False)
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    confidence = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"Face {confidence:.2f}",
                        (x1, max(y1 - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA,
                    )

            cv2.imshow("YOLO Face Detection - q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
