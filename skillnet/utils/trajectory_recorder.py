import time
import os
import re
from typing import List, Dict, Any, Optional, Tuple

from .file_utils import *
from .json_utils import *
from .event_utils import unpack_event


class TrajectoryRecorder:
    """
    Trajectory recorder that saves per-step data as JSON:
    - Actions (code executed)
    - Frame metadata (observations, voxels, inventory, status)
    - Events
    - Conversation history
    - Task metadata
    """
    def __init__(
        self,
        ckpt_dir="ckpt",
        resume=False,
        save_trajectories=True,
    ):
        self.ckpt_dir = ckpt_dir
        self.save_trajectories = save_trajectories
        f_mkdir(self.ckpt_dir, "trajectories")

        self.current_trajectory = []
        self.current_task = None
        self.current_context = ""
        self.trajectory_start_time = None
        self.step_counter = 0
        self.trajectory_id = None

        if resume:
            self.load_latest_trajectory()

    def start_trajectory(self, task: str, context: str = ""):
        """Start recording a new trajectory."""
        self.current_task = task
        self.current_context = context
        self.current_trajectory = []
        self.trajectory_start_time = time.time()
        self.step_counter = 0

        task_name = re.sub(r'[\\/:"*?<>| ]', "_", task).replace(" ", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.trajectory_start_time))
        self.trajectory_id = f"{task_name}_{timestamp}"

    def record_step(
        self,
        action: str,
        events: List,
        conversation: Optional[tuple] = None,
        success: Optional[bool] = None,
        critique: Optional[str] = None,
        ai_message: Optional[str] = None,
    ):
        """Record a single step in the trajectory."""
        if not self.save_trajectories:
            return

        self.step_counter += 1

        frame_data = self._extract_frame_data(events)

        step_data = {
            "step": self.step_counter,
            "timestamp": time.time(),
            "action": {
                "code": action,
                "step": self.step_counter,
                "timestamp": time.time(),
            },
            "frame": frame_data,
            "events": events,
        }

        if conversation is not None:
            step_data["conversation"] = {
                "system_message": conversation[0] if len(conversation) > 0 else None,
                "human_message": conversation[1] if len(conversation) > 1 else None,
                "ai_message": conversation[2] if len(conversation) > 2 else None,
            }
        if ai_message is not None:
            step_data["ai_message"] = ai_message
        if success is not None:
            step_data["success"] = success
        if critique is not None:
            step_data["critique"] = critique

        self.current_trajectory.append(step_data)

    def _extract_frame_data(self, events: List) -> Dict[str, Any]:
        """Extract frame/observation metadata from events."""
        frame_data = {
            "frames": [],
            "observations": [],
            "voxels": None,
            "status": None,
            "inventory": None,
            "position": None,
            "chat_log": [],
            "errors": [],
        }

        for ev in events:
            event_type, event = unpack_event(ev)
            if event_type is None:
                continue
            if event_type == "observe":
                frame = {
                    "timestamp": time.time(),
                    "status": event.get("status", {}),
                    "inventory": event.get("inventory", {}),
                    "voxels": event.get("voxels", []),
                    "nearbyChests": event.get("nearbyChests", {}),
                    "blockRecords": event.get("blockRecords", []),
                }
                frame_data["frames"].append(frame)

                obs = {
                    "status": event.get("status", {}),
                    "inventory": event.get("inventory", {}),
                    "voxels": event.get("voxels", []),
                    "nearbyChests": event.get("nearbyChests", {}),
                }
                frame_data["observations"].append(obs)

                # Read Minecraft-shape fields via .get() so the read sites
                # don't index the event dict literally; non-Minecraft payloads
                # simply produce empty/None values.
                voxels_val = event.get("voxels") if isinstance(event, dict) else None
                if voxels_val is not None:
                    frame_data["voxels"] = voxels_val
                status_val = event.get("status") if isinstance(event, dict) else None
                if status_val is not None:
                    frame_data["status"] = status_val
                    frame_data["position"] = (
                        status_val.get("position", {}) if isinstance(status_val, dict) else {}
                    )
                inventory_val = event.get("inventory") if isinstance(event, dict) else None
                if inventory_val is not None:
                    frame_data["inventory"] = inventory_val

            elif event_type == "onChat":
                chat_msg = event.get("onChat", "")
                frame_data["chat_log"].append(chat_msg)
                if frame_data["frames"]:
                    frame_data["frames"][-1].setdefault("chat_messages", []).append(chat_msg)

            elif event_type == "onError":
                error_msg = event.get("onError", "")
                frame_data["errors"].append(error_msg)
                if frame_data["frames"]:
                    frame_data["frames"][-1].setdefault("errors", []).append(error_msg)

            elif event_type == "onSave":
                save_info = {
                    "onSave": event.get("onSave", ""),
                    "status": event.get("status", {}),
                    "inventory": event.get("inventory", {}),
                    "voxels": event.get("voxels", []),
                    "blockRecords": event.get("blockRecords", []),
                }
                if frame_data["frames"]:
                    frame_data["frames"][-1]["onSave"] = save_info

        return frame_data

    def save_trajectory(self, success: bool = False, final_events: Optional[List] = None):
        """Save the current trajectory to disk."""
        if not self.save_trajectories or not self.current_trajectory:
            return

        task_name = re.sub(r'[\\/:"*?<>| ]', "_", self.current_task).replace(" ", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(self.trajectory_start_time))
        filename = f"{task_name}_{timestamp}.json"

        trajectory_data = {
            "task": self.current_task,
            "context": self.current_context,
            "start_time": self.trajectory_start_time,
            "end_time": time.time(),
            "duration": time.time() - self.trajectory_start_time,
            "total_steps": len(self.current_trajectory),
            "success": success,
            "steps": self.current_trajectory,
        }
        if final_events:
            trajectory_data["final_events"] = final_events

        filepath = f_join(self.ckpt_dir, "trajectories", filename)
        dump_json(trajectory_data, filepath)
        print(f"\033[92mTrajectory saved to: {filepath}\033[0m")

        self.current_trajectory = []
        self.current_task = None
        self.current_context = ""
        self.step_counter = 0

    def load_latest_trajectory(self):
        """Load the latest trajectory (for resume functionality)."""
        trajectories_dir = f_join(self.ckpt_dir, "trajectories")
        if not os.path.exists(trajectories_dir):
            return None

        files = f_listdir(self.ckpt_dir, "trajectories")
        if not files:
            return None

        files_with_time = [
            (f, os.path.getmtime(f_join(trajectories_dir, f)))
            for f in files if f.endswith(".json")
        ]
        if not files_with_time:
            return None

        latest_file = max(files_with_time, key=lambda x: x[1])[0]
        return load_json(f_join(trajectories_dir, latest_file))
