# Card detector for the Memorama project
# Uses perspective warp + template matching to find and classify cards

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


@dataclass
class CardConfig:
    # holds info for one card type
    name: str
    template_path: str
    bgr_colour: tuple


@dataclass
class Detection:
    card_name: str
    mse: float
    quad_pts: np.ndarray = field(repr=False)
    warped_roi: np.ndarray = field(repr=False)


CARD_CATALOGUE = [
    # CardConfig("Star",     "data/templates/star.png",     (220, 50, 50)),
    CardConfig("Flower",   "data/templates/flower.png",   (180, 60, 180)),
    CardConfig("Leaves",   "data/templates/leaves.png",   (30, 160, 30)),
    CardConfig("Sign",     "data/templates/sign.png",     (0, 0, 220)),
    CardConfig("Moon",     "data/templates/moon.png",     (220, 220, 30)),
    CardConfig("Triangle", "data/templates/triangle.png", (220, 120, 0)),
    CardConfig("Octagon",  "data/templates/octagon.png",  (160, 30, 160)),
    CardConfig("Heart",    "data/templates/heart.png",    (30, 160, 160)),
]

# thresholds / config
TEMPLATE_SIZE = 200
BORDER_MARGIN = 15  # trim the card border after warp
MAX_MSE_THRESHOLD = 3500.0
MIN_QUAD_AREA = 2_000
POLY_EPSILON_FACTOR = 0.03
DUPLICATE_CENTROID_DIST = 20.0
BLUR_KERNEL_SIZE = 5
ADAPTIVE_BLOCK_SIZE = 31  # needs to be odd
ADAPTIVE_C = 4


class ShapeMatcher:

    def __init__(self, catalogue=CARD_CATALOGUE, max_mse=MAX_MSE_THRESHOLD,
                 min_quad_area=MIN_QUAD_AREA, debug_warps=False):
        self.catalogue = catalogue
        self.max_mse = max_mse
        self.min_quad_area = min_quad_area
        self.debug_warps = debug_warps

        self._cfg_map = {c.name: c for c in catalogue}
        self._templates = {}
        self._load_templates()

    def _load_templates(self):
        for cfg in self.catalogue:
            path = Path(cfg.template_path)
            if not path.exists():
                raise FileNotFoundError(f"Template not found: {path.resolve()}")

            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"cv2.imread failed for {path}")

            img = cv2.resize(img, (TEMPLATE_SIZE, TEMPLATE_SIZE),
                             interpolation=cv2.INTER_AREA)

            # binarize so it matches what we get from live frames
            _, binary = cv2.threshold(img, 0, 255,
                                      cv2.THRESH_BINARY | cv2.THRESH_OTSU)

            if cv2.countNonZero(binary) == 0 or cv2.countNonZero(binary) == binary.size:
                raise RuntimeError(
                    f"Template '{cfg.name}' came out blank after binarizing, check the file."
                )

            self._templates[cfg.name] = binary
            print(f"loaded template: {cfg.name}")

    def process_frame(self, frame):
        if frame is None or frame.size == 0:
            return frame, []

        annotated = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), 0)

        # invert so the dark card border shows up as white for contour finding
        thresh = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            ADAPTIVE_BLOCK_SIZE,
            ADAPTIVE_C,
        )

        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST,
                                        cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        seen_quads = []

        for cnt in contours:
            quad_pts = self._fit_quad(cnt)
            if quad_pts is None:
                continue

            if self._is_duplicate(quad_pts, seen_quads):
                continue
            seen_quads.append(quad_pts)

            roi = self._extract_warp(gray, quad_pts)
            if roi is None:
                continue

            if self.debug_warps:
                cv2.imshow("[DEBUG] warped ROI", roi)

            det = self._classify_roi(roi, quad_pts)
            if det is None:
                continue

            detections.append(det)
            self._annotate_detection(annotated, det)
            print(f"found: {det.card_name}  mse={det.mse:.1f}")

        return annotated, detections

    def _fit_quad(self, contour):
        if cv2.contourArea(contour) < self.min_quad_area:
            return None

        perimeter = cv2.arcLength(contour, closed=True)
        approx = cv2.approxPolyDP(contour, POLY_EPSILON_FACTOR * perimeter, closed=True)

        if len(approx) != 4:
            return None

        pts = approx.reshape(4, 2).astype(np.float32)
        return self._order_points(pts)

    @staticmethod
    def _order_points(pts):
        # sort corners into TL, TR, BR, BL order
        ordered = np.zeros((4, 2), dtype=np.float32)
        s = pts.sum(axis=1)
        d = np.diff(pts, axis=1).ravel()
        ordered[0] = pts[np.argmin(s)]
        ordered[2] = pts[np.argmax(s)]
        ordered[1] = pts[np.argmin(d)]
        ordered[3] = pts[np.argmax(d)]
        return ordered

    @staticmethod
    def _extract_warp(gray_frame, pts, size=TEMPLATE_SIZE):
        dst = np.array(
            [[0, 0], [size - 1, 0], [size - 1, size - 1], [0, size - 1]],
            dtype=np.float32,
        )

        H = cv2.getPerspectiveTransform(pts, dst)
        warped = cv2.warpPerspective(gray_frame, H, (size, size))

        # crop out the card border
        m = BORDER_MARGIN
        inner = warped[m: size - m, m: size - m]

        if inner.size == 0:
            print(f"[warn] margin too large (margin={m}, size={size}), skipping quad")
            return None

        inner_resized = cv2.resize(inner, (size, size), interpolation=cv2.INTER_AREA)

        _, binary = cv2.threshold(inner_resized, 0, 255,
                                   cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return binary

    @staticmethod
    def _is_duplicate(pts, seen, max_dist=DUPLICATE_CENTROID_DIST):
        cx, cy = pts.mean(axis=0)
        for prev in seen:
            px, py = prev.mean(axis=0)
            if (cx - px) ** 2 + (cy - py) ** 2 < max_dist ** 2:
                return True
        return False

    def _classify_roi(self, roi, quad_pts):
        # try all 4 rotations since we don't know which way the card is facing
        rotations = [roi]
        for _ in range(3):
            rotations.append(cv2.rotate(rotations[-1], cv2.ROTATE_90_CLOCKWISE))

        best_name = None
        best_mse = float("inf")

        for name, template in self._templates.items():
            t = template.astype(np.float32)
            for rot_roi in rotations:
                diff = rot_roi.astype(np.float32) - t
                mse = float(np.mean(diff ** 2))
                if mse < best_mse:
                    best_mse = mse
                    best_name = name

        if best_name is None or best_mse > self.max_mse:
            print(f"rejected quad — best was {best_name} with MSE={best_mse:.1f} (thresh={self.max_mse:.1f})")
            return None

        return Detection(
            card_name=best_name,
            mse=round(best_mse, 1),
            quad_pts=quad_pts,
            warped_roi=roi,
        )

    def _annotate_detection(self, frame, det):
        cfg = self._cfg_map.get(det.card_name)
        colour = cfg.bgr_colour if cfg else (200, 200, 200)

        pts_int = det.quad_pts.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts_int], isClosed=True, color=colour, thickness=3)

        for corner in det.quad_pts.astype(np.int32):
            cv2.circle(frame, tuple(corner), 5, colour, cv2.FILLED)

        tl = det.quad_pts[0].astype(np.int32)
        label = f"{det.card_name}  mse={det.mse:.0f}"
        (lw, lh), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.60, 2)

        lx = max(0, int(tl[0]))
        ly = max(lh + baseline + 6, int(tl[1]) - 10)

        cv2.rectangle(frame, (lx, ly - lh - baseline - 4), (lx + lw + 6, ly),
                      colour, cv2.FILLED)
        cv2.putText(frame, label, (lx + 3, ly - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (255, 255, 255), 2, cv2.LINE_AA)


def _run_webcam_demo(camera_index=0, debug_warps=False):
    matcher = ShapeMatcher(debug_warps=debug_warps)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"can't open camera {camera_index}")
        sys.exit(1)

    print(f"camera {camera_index} running — press q to quit, s to save snapshot, d to toggle debug")

    snapshot_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("frame grab failed, trying again...")
            continue

        annotated, detections = matcher.process_frame(frame)

        hud = f"Cards found: {len(detections)}  |  mse_thresh={matcher.max_mse:.0f}"
        cv2.putText(annotated, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (240, 240, 240), 1, cv2.LINE_AA)

        cv2.imshow("Kinova HRI — Card Detector", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            fname = f"debug_snapshot_{snapshot_counter:03d}.png"
            cv2.imwrite(fname, annotated)
            print(f"saved {fname}")
            snapshot_counter += 1
        elif key == ord("d"):
            matcher.debug_warps = not matcher.debug_warps
            print(f"debug warps: {'on' if matcher.debug_warps else 'off'}")
            if not matcher.debug_warps:
                cv2.destroyWindow("[DEBUG] warped ROI")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Memorama card detector demo")
    parser.add_argument("--camera", type=int, default=0, help="camera index")
    parser.add_argument("--debug", action="store_true", help="show warped ROI window")
    args = parser.parse_args()
    _run_webcam_demo(camera_index=args.camera, debug_warps=args.debug)