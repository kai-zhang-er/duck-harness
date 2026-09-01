"""Capture and annotate one simulated head-camera red-ball frame."""

from pathlib import Path

from PIL import Image

from duckharness.adapters import MujocoMicroduckAdapter
from duckharness.perception import RedBallDetector, draw_detection


ROOT = Path("/Volumes/ssd/Programs/DuckHarness")
MODEL_PATH = ROOT / "duckharness/assets/scenes/navigation_camera.xml"
POLICY_PATH = ROOT / "microduck/policies/alpha_walking.onnx"
OUTPUT_PATH = Path("debug/red_ball_detection.png")


robot = MujocoMicroduckAdapter(MODEL_PATH, POLICY_PATH)
frame = robot.get_camera_frame("head")
detection = RedBallDetector().detect(frame.rgb)
annotated = draw_detection(frame.rgb, detection)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
Image.fromarray(annotated).save(OUTPUT_PATH)
print(f"Detection: {detection}")
print(f"Saved annotated frame to {OUTPUT_PATH.resolve()}")
