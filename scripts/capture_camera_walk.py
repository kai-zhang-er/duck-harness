"""Capture first-person RGB frames while walking in the camera scene."""

from pathlib import Path

from PIL import Image

from duckharness.adapters import MujocoMicroduckAdapter


ROOT = Path("/Volumes/ssd/Programs/DuckHarness")
MODEL_PATH = ROOT / "duckharness/assets/scenes/navigation_camera.xml"
POLICY_PATH = ROOT / "microduck/policies/alpha_walking.onnx"
OUTPUT_DIR = Path("frames")


robot = MujocoMicroduckAdapter(MODEL_PATH, POLICY_PATH)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
robot.move(vx=0.30)

try:
    for step_idx in range(250):
        robot.step()
        if step_idx % 10 == 0:
            frame = robot.get_camera_frame("head")
            Image.fromarray(frame.rgb).save(
                OUTPUT_DIR / f"{step_idx // 10:03d}.png"
            )
finally:
    robot.stop()

print(f"Saved camera frames to {OUTPUT_DIR.resolve()}")
