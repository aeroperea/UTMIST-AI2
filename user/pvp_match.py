# anchors: config, helpers, build_agents, run

# --- imports ---
from environment.environment import RenderMode, CameraResolution
from environment.agent import run_real_time_match
from user.train_agent import UserInputAgent, BasedAgent, ConstantAgent, ClockworkAgent, SB3Agent, RecurrentPPOAgent
from user.my_agent import SubmittedAgent
from typing import Optional, Dict, Any, List
import pygame
pygame.init()

# anchor: config
# primary checkpoint + optional alternate
path: str = "/home/aero/ML/RL/UTMIST-AI2/checkpoints/FusedFeatureExtractor7N_NewRewards/FusedFeatureExtractor7N_NewRewards_7499880_steps.zip"
pathAlt: Optional[str] = None

# primary architecture; set any to None to use defaults baked into SubmittedAgent
arch: Dict[str, Optional[Any]] = {
    "pi_shape": [256, 256, 128, 128],            # e.g., [256, 256, 128]
    "vf_shape": [256, 256, 128, 128],            # e.g., [256, 256, 128]
    "extractor_n_blocks": 7,  # e.g., 6
}

# optional alternate architecture
archAlt: Optional[Dict[str, Optional[Any]]] = {
    "pi_shape": [256, 256, 128, 128],            # e.g., [256, 256, 128]
    "vf_shape": [256, 256, 128, 128],            # e.g., [256, 256, 128]
    "extractor_n_blocks": 7,  # e.g., 6
}

# single boolean to flip both checkpoint sides and architectures
flipSides: bool = False

# anchor: helpers
def _final_paths_and_arch() -> Dict[str, Any]:
    p1 = path
    p2 = pathAlt if pathAlt is not None else path

    a1 = arch
    a2 = archAlt if archAlt is not None else arch

    if flipSides:
        p1, p2 = p2, p1
        a1, a2 = a2, a1
    return {"path1": p1, "path2": p2, "arch1": a1, "arch2": a2}

def _prune(d: Dict[str, Optional[Any]]) -> Dict[str, Any]:
    # drop None so SubmittedAgent uses its defaults
    return {k: v for k, v in d.items() if v is not None}

# anchor: build_agents
resolved = _final_paths_and_arch()
my_agent = SubmittedAgent(file_path=resolved["path1"], **_prune(resolved["arch1"]))
# opponent = UserInputAgent()
opponent = SubmittedAgent(file_path=resolved["path2"], **_prune(resolved["arch2"]))

# anchor: run
match_time = 99999
run_real_time_match(
    agent_1=my_agent,
    agent_2=opponent,
    max_timesteps=30 * 999990000,
    resolution=CameraResolution.LOW,
)
