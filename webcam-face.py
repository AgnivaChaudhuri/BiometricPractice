#!/usr/bin/env python3
import argparse
import time
from typing import Any, Dict, List, Tuple

import cv2
from deepface import DeepFace


def as_list(result: Any) -> List[Dict[str, Any]]:
    # DeepFace sometimes returns a dict (single face) or list (multiple faces)
    if result is None:
        return []
    if isinstance(result, list):
        return result
    if isinstance(result, dict):
        return [result]
    return []


def draw_emotion_overlay(
    frame_bgr,
    faces: List[Dict[str, Any]],
) -> None:
    for face in faces:
        region = face.get("region", {}) or {}
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))

        dominant = face.get("dominant_emotion", None)
        emo_scores = face.get("emotion", {}) or {}
        conf = None
        if dominant in emo_scores:
            conf = emo_scores[dominant]

        label = dominant if dominant else "no-face"
        if conf is not None:
            label = f"{label} ({conf:.1f})"

        # Draw box if region looks valid
        if w > 0 and h > 0:
            cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame_bgr,
                label,
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        else:
            # Fallback label top-left if region missing
            cv2.putText(
                frame_bgr,
                label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )


def print_emotions(faces: List[Dict[str, Any]]) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    if not faces:
        print(f"[{ts}] no face detected")
        return

    for i, face in enumerate(faces):
        dominant = face.get("dominant_emotion", "unknown")
        emo_scores = face.get("emotion", {}) or {}
        top3 = sorted(emo_scores.items(), key=lambda kv: kv[1], reverse=True)[:3]
        top3_str = ", ".join([f"{k}:{v:.1f}" for k, v in top3])
        print(f"[{ts}] face#{i}: dominant={dominant} | top3=({top3_str})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Real-time emotion recognition from webcam using DeepFace.")
    ap.add_argument("--camera-index", type=int, default=0, help="Webcam index (default: 0).")
    ap.add_argument("--width", type=int, default=640, help="Capture width (best-effort).")
    ap.add_argument("--height", type=int, default=480, help="Capture height (best-effort).")
    ap.add_argument("--analyze-every", type=int, default=10, help="Run emotion analysis every N frames.")
    ap.add_argument(
        "--detector-backend",
        type=str,
        default="opencv",
        help="DeepFace detector backend: opencv, retinaface, mtcnn, mediapipe, etc.",
    )
    ap.add_argument("--enforce-detection", action="store_true", help="If set, raise error when no face is found.")
    ap.add_argument("--no-display", action="store_true", help="Disable video window display.")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    # Best-effort set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    frame_idx = 0
    last_faces: List[Dict[str, Any]] = []

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                print("Failed to read frame from camera.")
                break

            frame_idx += 1

            if frame_idx % max(1, args.analyze_every) == 0:
                try:
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    result = DeepFace.analyze(
                        img_path=frame_rgb,
                        actions=["emotion"],
                        detector_backend=args.detector_backend,
                        enforce_detection=args.enforce_detection,
                        silent=True,
                    )
                    last_faces = as_list(result)
                    print_emotions(last_faces)
                except Exception as e:
                    # If enforce_detection=False, DeepFace may still throw on some frames/backends.
                    print(f"[warn] analysis failed: {e}")

            if not args.no_display:
                vis = frame_bgr.copy()
                draw_emotion_overlay(vis, last_faces)
                cv2.imshow("Webcam Emotions (press 'q' to quit)", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

