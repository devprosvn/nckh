#!/usr/bin/env python3
"""
Evaluate a trained SAC agent in CARLA, record each run to MP4, 
and give the user a live top‑down spectator view.

Dependencies (all pure‑Python):
  • moviepy>=1.0.3
  • numpy
Make sure CARLA is already running, e.g.:
  CarlaUE4.exe -benchmark -quality-level=Low -nosound
"""
import argparse
import pathlib
import queue
import subprocess
import sys
import time
from collections import deque

import paddle
# paddle.set_device("gpu:0")
paddle.set_device("cpu")

import logging
logger = logging.getLogger(__name__)

import moviepy.editor as mvp
import numpy as np
from env_utils import LocalEnv
from parl.utils import logger, tensorboard
# from torch_base import TorchModel, TorchSAC, TorchAgent
from paddle_base import PaddleModel, PaddleSAC, PaddleAgent
from env_config import EnvConfig

# ──────────────────────────────────────────────────────────
EVAL_EPISODES = 3
GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
ACTOR_LR = 3e-4
CRITIC_LR = 3e-4


def _setup_recorder(world, vehicle, h=720, w=1280, fov=90):
    """
    Spawn an RGB camera ~50 m above the ego vehicle, looking straight down.
    Return the camera actor and a thread‑safe queue that will receive numpy frames.
    """
    import carla  # imported late to avoid a hard dependency when unit‑testing

    bp_lib = world.get_blueprint_library()
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(w))
    cam_bp.set_attribute("image_size_y", str(h))
    cam_bp.set_attribute("fov", str(fov))

    top = carla.Transform(
        carla.Location(x=0.0, y=0.0, z=50),
        carla.Rotation(pitch=-90, yaw=0, roll=0),
    )
    camera = world.spawn_actor(cam_bp, top, attach_to=vehicle)

    frame_q: "queue.Queue[np.ndarray]" = queue.Queue(maxsize=256)

    def _on_image(image: "carla.Image"):
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = arr.reshape((image.height, image.width, 4))[:, :, :3]  # drop alpha
        try:
            frame_q.put_nowait(arr)
        except queue.Full:
            pass  # drop frames if consumer is too slow

    camera.listen(_on_image)
    return camera, frame_q


def _update_spectator(world, vehicle):
    """Place CARLA's spectator directly above the vehicle each tick."""
    import carla

    spectator: carla.Actor = world.get_spectator()
    veh_tf = vehicle.get_transform()
    loc = veh_tf.location + carla.Location(z=50.0)
    rot = carla.Rotation(pitch=-90.0, yaw=veh_tf.rotation.yaw)
    spectator.set_transform(carla.Transform(loc, rot))


def _frames_to_video(frames, out_file: pathlib.Path, fps: int):
    """Write a list/iterator of RGB numpy frames to MP4 using MoviePy."""
    clip = mvp.ImageSequenceClip(list(frames), fps=fps)
    # libx264 gives small files and broad player support
    clip.write_videofile(out_file.as_posix(), codec="libx264", audio=False)
    clip.close()


def run_episode(agent, env, episode_idx, video_dir, fps):
    world = env.unwrapped.world
    world.tick()  # Ensure world state is updated
    actors = world.get_actors().filter('vehicle.tesla.model3')
    if not actors:
        logger.error("No vehicle.tesla.model3 actors found. Available actors: %s",
                     [a.type_id for a in world.get_actors()])
        raise RuntimeError("No ego vehicle found")
    vehicle = actors[0]
    camera, frame_q = _setup_recorder(world, vehicle)
    episode_reward = 0.0
    obs = env.reset()
    done = False
    steps = 0
    frames = deque()
    try:
        while not done and steps < env._max_episode_steps:
            steps += 1
            action = agent.predict(obs)
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            while not frame_q.empty():
                frames.append(frame_q.get_nowait())
            _update_spectator(world, vehicle)
    finally:
        camera.stop()
        camera.destroy()
    while not frame_q.empty():
        frames.append(frame_q.get_nowait())
    if frames:
        out_path = video_dir / f"episode_{episode_idx:03d}.mp4"
        _frames_to_video(frames, out_path, fps=fps)
        logger.info("Saved video %s (%d frames)", out_path, len(frames))
    return episode_reward


def main(args):
    logger.info("-----------------Carla_SAC-------------------")
    logger.set_dir("./{}_eval".format(args.env))

    # env for eval
    eval_env_params = EnvConfig["test_env_params"]
    eval_env = LocalEnv(args.env, eval_env_params)

    obs_dim = eval_env.obs_dim
    action_dim = eval_env.action_dim

    # Initialize model, algorithm, agent
    if args.framework == "torch":
        # TorchModel, TorchSAC, TorchAgent assumed available
        from torch_base import TorchModel, TorchSAC, TorchAgent  # noqa: WPS433

        CarlaModel, SAC, CarlaAgent = TorchModel, TorchSAC, TorchAgent
    else:
        CarlaModel, SAC, CarlaAgent = PaddleModel, PaddleSAC, PaddleAgent

    model = CarlaModel(obs_dim, action_dim)
    algorithm = SAC(
        model,
        gamma=GAMMA,
        tau=TAU,
        alpha=ALPHA,
        actor_lr=ACTOR_LR,
        critic_lr=CRITIC_LR,
    )
    agent = CarlaAgent(algorithm)
    agent.restore(f"./{args.restore_model}")

    video_dir = pathlib.Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)

    for ep in range(args.eval_episodes):
        reward = run_episode(agent, eval_env, ep, video_dir, args.fps)
        tensorboard.add_scalar("eval/episode_reward", reward, ep)
        logger.info("Episode %d reward: %.2f", ep, reward)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="carla-v0")
    p.add_argument(
        "--framework",
        default="paddle",
        choices=["paddle", "torch"],
        help="deep learning framework",
    )
    p.add_argument("--eval_episodes", default=EVAL_EPISODES, type=int)
    p.add_argument("--restore_model", default="model.ckpt")
    p.add_argument("--video_dir", default="videos", help="where MP4s are saved")
    p.add_argument("--fps", default=20, type=int, help="recording frame‑rate")
    args = p.parse_args()
    main(args)