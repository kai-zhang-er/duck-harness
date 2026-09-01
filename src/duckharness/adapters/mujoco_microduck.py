"""MuJoCo adapter for the official MicroDuck walking policy."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnxruntime as ort

from .base import CameraFrame, RobotState


# These values match the validated walking runtime in microduck_rl/scripts/
# infer_policy.py. The ONNX walking policies use the unified 61D contract.
DEFAULT_POSE = np.array(
    [
        0.0,
        -0.0873,
        -0.4579,
        -0.0049,
        0.4530,
        0.3491,
        0.3491,
        0.0,
        0.0,
        0.0,
        0.0873,
        0.4579,
        0.0049,
        -0.4530,
    ],
    dtype=np.float32,
)

OBSERVATION_SIZE = 61
ACTION_SIZE = 14
SIMULATION_TIMESTEP = 0.005
CONTROL_FREQUENCY = 50.0
VIRTUAL_CAMERA_PITCHES = {
    "head_forward": 0.0,
    "head_down_20": -math.radians(20.0),
    "head_down_35": -math.radians(35.0),
}


class MujocoMicroduckAdapter:
    """Drive an official MicroDuck MuJoCo scene through a walking ONNX policy.

    ``move`` only changes the command. ``step`` owns the control loop: it
    reads the current state, builds the policy observation, runs inference,
    applies the actuator targets, and advances MuJoCo at its fixed timestep.
    """

    def __init__(
        self,
        model_path: str | Path,
        walking_policy_path: str | Path,
        *,
        action_scale: float = 1.0,
        fallen_height: float = 0.06,
        simulation_timestep: float = SIMULATION_TIMESTEP,
        control_frequency: float = CONTROL_FREQUENCY,
        camera_name: str = "head_camera",
        image_width: int = 640,
        image_height: int = 480,
        session_options: ort.SessionOptions | None = None,
    ) -> None:
        if simulation_timestep <= 0.0:
            raise ValueError("simulation_timestep must be positive")
        if control_frequency <= 0.0:
            raise ValueError("control_frequency must be positive")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("image_width and image_height must be positive")

        self.model_path = Path(model_path)
        self.walking_policy_path = Path(walking_policy_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MuJoCo model not found: {self.model_path}")
        if not self.walking_policy_path.is_file():
            raise FileNotFoundError(
                f"Walking policy not found: {self.walking_policy_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.simulation_timestep = float(simulation_timestep)
        self.model.opt.timestep = self.simulation_timestep
        self.data = mujoco.MjData(self.model)

        self.control_frequency = float(control_frequency)
        self.control_timestep = 1.0 / self.control_frequency
        decimation_float = self.control_timestep / self.simulation_timestep
        self.decimation = round(decimation_float)
        if self.decimation < 1 or not math.isclose(
            decimation_float, self.decimation, rel_tol=0.0, abs_tol=1e-9
        ):
            raise ValueError(
                "control_timestep must be an integer multiple of "
                "simulation_timestep"
            )

        self.action_scale = float(action_scale)
        self.fallen_height = float(fallen_height)
        self.command = np.zeros(13, dtype=np.float32)
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)

        self._trunk_qpos_adr, self._trunk_qvel_adr = self._find_freejoint(
            "trunk_base_freejoint"
        )
        self._trunk_body_id = self._name_id(
            mujoco.mjtObj.mjOBJ_BODY, "trunk_base"
        )
        self._imu_ang_vel_id = self._name_id(
            mujoco.mjtObj.mjOBJ_SENSOR, "imu_ang_vel"
        )
        self._joint_qpos_indices = []
        self._joint_qvel_indices = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                raise ValueError(f"Actuator {actuator_id} has no joint transmission")
            self._joint_qpos_indices.append(int(self.model.jnt_qposadr[joint_id]))
            self._joint_qvel_indices.append(int(self.model.jnt_dofadr[joint_id]))

        if self.model.nu != ACTION_SIZE:
            raise ValueError(
                f"Expected {ACTION_SIZE} actuators for MicroDuck, got {self.model.nu}"
            )
        self.default_pose = DEFAULT_POSE.copy()

        self.camera_name = camera_name
        camera_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        if camera_id < 0:
            raise ValueError(f"Required MuJoCo camera {camera_name!r} not found")
        self._camera_id = int(camera_id)
        self._head_camera_base_local_rotation: np.ndarray | None = None
        self.image_width = int(image_width)
        self.image_height = int(image_height)
        # Renderer creation needs a graphics connection on macOS. Keep it
        # lazy so headless physics/control users do not need a display.
        self._renderer: mujoco.Renderer | None = None

        self.policy = ort.InferenceSession(
            str(self.walking_policy_path), sess_options=session_options
        )
        policy_input = self.policy.get_inputs()[0]
        policy_output = self.policy.get_outputs()[0]
        self._input_name = policy_input.name
        self._output_name = policy_output.name
        self._validate_policy_shape(policy_input.shape, policy_output.shape)

        self._reset_to_default_pose()
        if self.camera_name == "head_camera":
            self._orient_head_camera_forward()

    def move(self, vx: float, vy: float = 0.0, vyaw: float = 0.0) -> None:
        """Set a body-frame velocity command without advancing simulation."""

        values = np.asarray([vx, vy, vyaw], dtype=np.float32)
        if not np.all(np.isfinite(values)):
            raise ValueError("velocity commands must be finite")
        self.command[:3] = values

    def stop(self) -> None:
        """Set all walking velocity commands to zero."""

        self.move(0.0, 0.0, 0.0)

    @property
    def control_dt(self) -> float:
        """Duration of one policy/control period in seconds."""

        return self.control_timestep

    @property
    def sim_time(self) -> float:
        """Current MuJoCo simulation time in seconds."""

        return float(self.data.time)

    def step(self) -> None:
        """Advance one 50 Hz control period.

        One policy action is computed from the current observation, then held
        across ``decimation`` MuJoCo steps at the fixed simulation timestep.
        """

        self._apply_policy_action()
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)

    def state(self) -> RobotState:
        """Return ground-truth base pose and world-frame linear velocity."""

        position = self.data.qpos[
            self._trunk_qpos_adr : self._trunk_qpos_adr + 3
        ].astype(np.float64)
        quaternion = self.data.qpos[
            self._trunk_qpos_adr + 3 : self._trunk_qpos_adr + 7
        ]
        linear_velocity = self.data.qvel[
            self._trunk_qvel_adr : self._trunk_qvel_adr + 3
        ].astype(np.float64)

        w, x, y, z = quaternion
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )

        return RobotState(
            position=tuple(float(value) for value in position),
            yaw=float(yaw),
            linear_velocity=tuple(float(value) for value in linear_velocity),
            fallen=bool(position[2] < self.fallen_height),
        )

    def get_camera_frame(self, camera: str = "head") -> CameraFrame:
        """Render an RGB frame without advancing physics.

        ``head`` and ``head_forward`` are aliases for the default camera.
        ``head_down_20`` and ``head_down_35`` are virtual views that keep the
        same head-mounted camera position but tilt its optical axis downward.
        A concrete MuJoCo camera name is also accepted for simulation-specific
        cameras. Selecting a view never advances physics.
        """

        if camera == "head":
            concrete_name = self.camera_name
            frame_name = concrete_name
            if self.camera_name == "head_camera":
                self._set_head_camera_pitch(0.0)
        elif camera in VIRTUAL_CAMERA_PITCHES:
            if self.camera_name != "head_camera":
                raise ValueError(
                    "virtual head camera views require camera_name='head_camera'"
                )
            concrete_name = self.camera_name
            frame_name = camera
            self._set_head_camera_pitch(VIRTUAL_CAMERA_PITCHES[camera])
        else:
            concrete_name = camera
            frame_name = concrete_name
            camera_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_CAMERA,
                concrete_name,
            )
            if camera_id < 0:
                raise ValueError(f"MuJoCo camera {camera!r} not found")
            if concrete_name == self.camera_name and self.camera_name == "head_camera":
                self._set_head_camera_pitch(0.0)

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model,
                height=self.image_height,
                width=self.image_width,
            )
        self._renderer.update_scene(self.data, camera=concrete_name)
        rgb = self._renderer.render().copy()
        return CameraFrame(
            rgb=rgb,
            timestamp=float(self.data.time),
            camera_name=frame_name,
        )

    def get_rgb(self, camera: str = "head") -> np.ndarray:
        """Return a copied RGB array as a convenience API."""

        return self.get_camera_frame(camera).rgb

    def _apply_policy_action(self) -> None:
        observation = self._observation().reshape(1, OBSERVATION_SIZE)
        action = self.policy.run(
            [self._output_name], {self._input_name: observation}
        )[0]
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.size != ACTION_SIZE:
            raise ValueError(
                f"Walking policy returned {action.size} actions; "
                f"expected {ACTION_SIZE}"
            )
        self.last_action = action.copy()
        self.data.ctrl[:] = self.default_pose + action * self.action_scale

    def _observation(self) -> np.ndarray:
        """Build the official 61D walking observation."""

        base_ang_vel = self._sensor_data(self._imu_ang_vel_id)
        gravity = self._projected_gravity()
        joint_pos = (
            self.data.qpos[self._joint_qpos_indices].astype(np.float32)
            - self.default_pose
        )
        joint_vel = self.data.qvel[self._joint_qvel_indices].astype(np.float32)
        return np.concatenate(
            [base_ang_vel, gravity, joint_pos, joint_vel, self.last_action, self.command]
        ).astype(np.float32)

    def _projected_gravity(self) -> np.ndarray:
        quaternion = self.data.xquat[self._trunk_body_id].astype(np.float32)
        world_gravity = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        return self._quat_rotate_inverse(quaternion, world_gravity)

    def _sensor_data(self, sensor_id: int) -> np.ndarray:
        address = int(self.model.sensor_adr[sensor_id])
        return self.data.sensordata[address : address + 3].copy().astype(np.float32)

    @staticmethod
    def _quat_rotate_inverse(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
        w = quaternion[0]
        xyz = quaternion[1:4]
        cross_term = np.cross(xyz, vector) * 2.0
        return vector - w * cross_term + np.cross(xyz, cross_term)

    def _reset_to_default_pose(self) -> None:
        self.data.qpos[self._trunk_qpos_adr : self._trunk_qpos_adr + 3] = [
            0.0,
            0.0,
            0.125,
        ]
        self.data.qpos[self._trunk_qpos_adr + 3 : self._trunk_qpos_adr + 7] = [
            1.0,
            0.0,
            0.0,
            0.0,
        ]
        for index, qpos_index in enumerate(self._joint_qpos_indices):
            self.data.qpos[qpos_index] = self.default_pose[index]
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.default_pose
        mujoco.mj_forward(self.model, self.data)

    def _orient_head_camera_forward(self) -> None:
        """Aim the existing head-mounted camera along the robot's +X axis.

        The upstream camera is correctly attached to the moving head body but
        its authored optical axis points backward for this deployment scene.
        Reorienting only the camera's local frame preserves the attachment and
        makes the stable ``head`` API a forward-facing first-person view.
        """

        camera_id = self._camera_id
        parent_id = int(self.model.cam_bodyid[camera_id])
        parent_rotation = self.data.xmat[parent_id].reshape(3, 3)

        # MuJoCo cameras look along local -Z and use local +Y as image up.
        # Express a forward-facing camera in the head body's local frame once;
        # retaining this local transform lets the camera follow the head and
        # trunk when the robot turns.
        desired_world_rotation = np.array(
            [
                [0.0, 0.0, -1.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        self._head_camera_base_local_rotation = (
            parent_rotation.T @ desired_world_rotation
        )
        self._set_head_camera_pitch(0.0)

    def _set_head_camera_pitch(self, pitch: float) -> None:
        """Set a virtual pitch offset on the head camera without moving joints."""

        if self._head_camera_base_local_rotation is None:
            raise RuntimeError("head camera forward pose has not been initialized")

        camera_id = self._camera_id
        cos_pitch = math.cos(pitch)
        sin_pitch = math.sin(pitch)
        pitch_rotation = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, cos_pitch, -sin_pitch],
                [0.0, sin_pitch, cos_pitch],
            ],
            dtype=np.float64,
        )
        local_rotation = self._head_camera_base_local_rotation @ pitch_rotation
        local_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(local_quaternion, local_rotation.reshape(-1))
        self.model.cam_quat[camera_id] = local_quaternion
        mujoco.mj_forward(self.model, self.data)

    def _find_freejoint(self, name: str) -> tuple[int, int]:
        joint_id = self._name_id(mujoco.mjtObj.mjOBJ_JOINT, name)
        if self.model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {name!r} is not a free joint")
        return (
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def _name_id(self, object_type: Any, name: str) -> int:
        object_id = mujoco.mj_name2id(self.model, object_type, name)
        if object_id < 0:
            raise ValueError(f"Required MuJoCo {object_type.name} {name!r} not found")
        return int(object_id)

    @staticmethod
    def _validate_policy_shape(input_shape: Any, output_shape: Any) -> None:
        input_size = input_shape[-1] if input_shape else None
        if isinstance(input_size, int) and input_size != OBSERVATION_SIZE:
            raise ValueError(
                f"Walking policy expects {input_size} observations; "
                f"this adapter emits {OBSERVATION_SIZE}."
            )
        output_size = output_shape[-1] if output_shape else None
        if isinstance(output_size, int) and output_size != ACTION_SIZE:
            raise ValueError(
                f"Walking policy returns {output_size} actions; "
                f"this adapter expects {ACTION_SIZE}."
            )
