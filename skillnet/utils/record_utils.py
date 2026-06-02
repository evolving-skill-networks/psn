import re
import time

from .file_utils import *
from .json_utils import *
from .event_utils import unpack_event


class EventRecorder:
    def __init__(
        self,
        ckpt_dir="ckpt",
        resume=False,
        init_position=None,
    ):
        self.ckpt_dir = ckpt_dir
        self.item_history = set()
        self.item_vs_time = {}
        self.item_vs_iter = {}
        self.biome_history = set()
        self.init_position = init_position
        self.position_history = [[0, 0]]
        self.elapsed_time = 0
        self.iteration = 0
        f_mkdir(self.ckpt_dir, "events")
        if resume:
            self.resume()

    def record(self, events, task):
        # Always increment iteration counter first (matches Original Voyager behavior
        # where every step() call counts as one iteration, regardless of event validity).
        task_safe = re.sub(r'[\\/:"*?<>| ]', "_", task)
        task_safe = task_safe.replace(" ", "_") + time.strftime(
            "_%Y%m%d_%H%M%S", time.localtime()
        )
        self.iteration += 1

        # Validate events for file recording (skip writing if invalid,
        # but iteration has already been counted above)
        if not events or not isinstance(events, list) or len(events) == 0:
            print(
                f"\033[96m****Recorder message: {self.iteration} iteration passed "
                f"(empty events, no file written)****\033[0m"
            )
            return

        first_event = events[0]
        if not isinstance(first_event, (list, tuple)) or len(first_event) < 2:
            print(
                f"\033[33m[Recorder] Warning: first event structure invalid, "
                f"skipping file write (iteration {self.iteration}).\033[0m"
            )
            return

        first_event_data = first_event[1] if len(first_event) > 1 else {}
        if not isinstance(first_event_data, dict):
            print(
                f"\033[33m[Recorder] Warning: first event data is not a dict, "
                f"skipping file write (iteration {self.iteration}).\033[0m"
            )
            return

        task = task_safe
        if not self.init_position:
            # Safely fetch position
            status = first_event_data.get("status", {})
            position = status.get("position", {}) if isinstance(status, dict) else {}
            if position and "x" in position and "z" in position:
                self.init_position = [
                    position["x"],
                    position["z"],
                ]
            else:
                print("\033[33m[Recorder] Warning: position not found in first event, using default\033[0m")
                self.init_position = [0, 0]
        for ev in events:
            event_type, event = unpack_event(ev)
            if event_type is None:
                continue
            if event_type == "observe":
                self.update_items(event)
                self.update_elapsed_time(event)
        print(
            f"\033[96m****Recorder message: {self.elapsed_time} ticks have elapsed****\033[0m\n"
            f"\033[96m****Recorder message: {self.iteration} iteration passed****\033[0m"
        )
        dump_json(events, f_join(self.ckpt_dir, "events", task))

    def resume(self, cutoff=None):
        self.item_history = set()
        self.item_vs_time = {}
        self.item_vs_iter = {}
        self.elapsed_time = 0
        self.position_history = [[0, 0]]

        def get_timestamp(string):
            timestamp = "_".join(string.split("_")[-2:])
            return time.mktime(time.strptime(timestamp, "%Y%m%d_%H%M%S"))

        records = f_listdir(self.ckpt_dir, "events")
        sorted_records = sorted(records, key=get_timestamp)
        for record in sorted_records:
            self.iteration += 1
            if cutoff and self.iteration > cutoff:
                break
            events = load_json(f_join(self.ckpt_dir, "events", record))
            
            # Defensive check
            if not events or not isinstance(events, list) or len(events) == 0:
                continue
            
            if not self.init_position:
                # Safely fetch position
                first_event = events[0]
                if isinstance(first_event, (list, tuple)) and len(first_event) > 1:
                    first_event_data = first_event[1]
                    if isinstance(first_event_data, dict):
                        status = first_event_data.get("status", {})
                        position = status.get("position", {}) if isinstance(status, dict) else {}
                        if position and "x" in position and "z" in position:
                            self.init_position = (position["x"], position["z"])
            
            for ev in events:
                event_type, event = unpack_event(ev)
                if event_type is None:
                    continue
                if event_type == "observe":
                    self.update_items(event)
                    self.update_position(event)
                    self.update_elapsed_time(event)

    def update_items(self, event):
        # Only process events that contain a status field (typically observe events)
        status = event.get("status") if isinstance(event, dict) else None
        if not isinstance(status, dict):
            return
        inventory = event.get("inventory", {})
        elapsed_time = status.get("elapsedTime", 0)
        biome = status.get("biome", "")
        items = set(inventory.keys())
        new_items = items - self.item_history
        self.item_history.update(items)
        self.biome_history.add(biome)
        if new_items:
            if self.elapsed_time + elapsed_time not in self.item_vs_time:
                self.item_vs_time[self.elapsed_time + elapsed_time] = []
            self.item_vs_time[self.elapsed_time + elapsed_time].extend(new_items)
            if self.iteration not in self.item_vs_iter:
                self.item_vs_iter[self.iteration] = []
            self.item_vs_iter[self.iteration].extend(new_items)

    def update_elapsed_time(self, event):
        # Only process events that contain a status field (typically observe events)
        status = event.get("status") if isinstance(event, dict) else None
        if not isinstance(status, dict):
            return
        self.elapsed_time += status.get("elapsedTime", 0)

    def update_position(self, event):
        # Only process events that contain a status field (typically observe events)
        status = event.get("status") if isinstance(event, dict) else None
        if not isinstance(status, dict):
            return
        pos = status.get("position") if isinstance(status.get("position"), dict) else None
        if pos is None:
            return
        position = [
            pos.get("x", 0) - self.init_position[0],
            pos.get("z", 0) - self.init_position[1],
        ]
        if self.position_history[-1] != position:
            self.position_history.append(position)
