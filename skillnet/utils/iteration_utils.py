"""
Iteration utilities for PSN

Provides tools for parsing iteration logs and preparing checkpoint data
for resuming from a specific iteration.
"""

import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Callable, Any, Set


def parse_iteration_log(
    ckpt_dir: str,
    max_iteration: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Parse the iteration_log.jsonl file.

    Args:
        ckpt_dir: checkpoint directory
        max_iteration: optional; only return records with iteration <= this value

    Returns:
        List of iteration records
    """
    log_path = os.path.join(ckpt_dir, "progress", "iteration_log.jsonl")

    if not os.path.exists(log_path):
        return []

    records = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                iteration = record.get("iteration", 0)

                if max_iteration is not None and iteration > max_iteration:
                    break

                records.append(record)
            except json.JSONDecodeError:
                continue

    return records


def get_iteration_cutoff_info(
    ckpt_dir: str,
    target_iteration: int
) -> Optional[Dict[str, Any]]:
    """
    Get truncation info for a specified iteration.

    Args:
        ckpt_dir: checkpoint directory
        target_iteration: target iteration number

    Returns:
        Dict with timestamp, inventory_snapshot, etc. or None if not found
    """
    records = parse_iteration_log(ckpt_dir, max_iteration=target_iteration)

    if not records:
        return None

    # Prefer exact match
    for record in reversed(records):
        if record.get("iteration") == target_iteration:
            return record

    # If no exact match, return the last record (most recent smaller iteration)
    return records[-1] if records else None


def truncate_jsonl_file(
    src_path: str,
    dest_path: str,
    filter_fn: Callable[[Dict], bool]
) -> int:
    """
    Truncate a JSONL file to a target path, copying only matching records.

    Args:
        src_path: source JSONL file path
        dest_path: target JSONL file path
        filter_fn: filter function; return True to keep

    Returns:
        number of records copied
    """
    if not os.path.exists(src_path):
        return 0

    # Ensure the destination directory exists
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    count = 0
    with open(src_path, 'r', encoding='utf-8') as src_f:
        with open(dest_path, 'w', encoding='utf-8') as dest_f:
            for line in src_f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    if filter_fn(record):
                        dest_f.write(json.dumps(record, ensure_ascii=False) + '\n')
                        count += 1
                except json.JSONDecodeError:
                    continue

    return count


def get_event_timestamp(filename: str) -> float:
    """
    Extract timestamp from an event filename.

    Filename format: {task_name}_{YYYYMMDD}_{HHMMSS}

    Args:
        filename: event filename (without extension)

    Returns:
        Unix timestamp
    """
    try:
        # Format: task_name_YYYYMMDD_HHMMSS
        parts = filename.split('_')
        if len(parts) >= 2:
            timestamp_str = "_".join(parts[-2:])
            return time.mktime(time.strptime(timestamp_str, "%Y%m%d_%H%M%S"))
    except:
        pass
    return 0


def get_event_files_before_timestamp(
    ckpt_dir: str,
    cutoff_timestamp: str
) -> List[str]:
    """
    Get the list of event files before a cutoff timestamp.

    Args:
        ckpt_dir: checkpoint directory
        cutoff_timestamp: ISO-format cutoff timestamp

    Returns:
        List of event file names to keep
    """
    events_dir = os.path.join(ckpt_dir, "events")
    if not os.path.exists(events_dir):
        return []

    # Parse cutoff timestamp
    try:
        cutoff_dt = datetime.fromisoformat(cutoff_timestamp.replace('Z', '+00:00'))
        cutoff_unix = cutoff_dt.timestamp()
    except:
        # On parse failure, return empty list
        return []

    files_to_keep = []
    for filename in os.listdir(events_dir):
        # Skip non-files
        file_path = os.path.join(events_dir, filename)
        if not os.path.isfile(file_path):
            continue

        # Extract filename (without extension)
        name_without_ext = os.path.splitext(filename)[0]
        file_timestamp = get_event_timestamp(name_without_ext)

        if file_timestamp > 0 and file_timestamp <= cutoff_unix:
            files_to_keep.append(filename)

    return files_to_keep


def rebuild_curriculum_from_iteration_log(
    records: List[Dict[str, Any]],
    dest_dir: str
) -> Dict[str, Any]:
    """
    Rebuild curriculum data from iteration_log records.

    Args:
        records: list of iteration log records
        dest_dir: target checkpoint directory

    Returns:
        Dict with completed_tasks and failed_tasks counts
    """
    completed_tasks = []
    failed_tasks = []

    for record in records:
        task = record.get("task", "")
        success = record.get("success", False)

        if success:
            if task not in completed_tasks:
                completed_tasks.append(task)
        else:
            failed_tasks.append(task)

    # Remove completed tasks from failed_tasks
    for task in completed_tasks:
        while task in failed_tasks:
            failed_tasks.remove(task)

    # Create directory and write files
    curriculum_dir = os.path.join(dest_dir, "curriculum")
    os.makedirs(curriculum_dir, exist_ok=True)

    with open(os.path.join(curriculum_dir, "completed_tasks.json"), 'w', encoding='utf-8') as f:
        json.dump(completed_tasks, f, ensure_ascii=False, indent=2)

    with open(os.path.join(curriculum_dir, "failed_tasks.json"), 'w', encoding='utf-8') as f:
        json.dump(failed_tasks, f, ensure_ascii=False, indent=2)

    # Create an empty qa_cache.json (vectordb is rebuilt on resume)
    with open(os.path.join(curriculum_dir, "qa_cache.json"), 'w', encoding='utf-8') as f:
        json.dump({}, f)

    return {
        "completed_count": len(completed_tasks),
        "failed_count": len(failed_tasks)
    }


def rebuild_psn_progress_from_inventory(
    inventory: Dict[str, int],
    dest_dir: str,
    domain_knowledge=None,
) -> Set[str]:
    """
    Rebuild PSN curriculum milestone state based on inventory.

    Args:
        inventory: inventory snapshot
        dest_dir: target checkpoint directory

    Returns:
        Set of completed milestone names
    """
    # Get milestone-item mapping from domain knowledge
    MILESTONE_ITEMS = domain_knowledge.get_milestone_item_mapping() if domain_knowledge else {}

    completed_milestones = set()

    # Determine completed milestones based on inventory
    for item, milestone in MILESTONE_ITEMS.items():
        if item in inventory and inventory[item] > 0:
            completed_milestones.add(milestone)

    # Create directory and write files
    psn_dir = os.path.join(dest_dir, "psn_curriculum")
    os.makedirs(psn_dir, exist_ok=True)

    progress_data = {
        "completed_milestones": list(completed_milestones),
        "current_goal": None,
        "achievements": []
    }

    with open(os.path.join(psn_dir, "progress.json"), 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, ensure_ascii=False, indent=2)

    return completed_milestones


def truncate_skill_graph(
    src_dir: str,
    dest_dir: str,
    cutoff_timestamp: str
) -> Dict[str, Any]:
    """
    Copy and truncate the skill_graph directory.

    Delete skills created after the cutoff timestamp and roll back node versions.

    Args:
        src_dir: source checkpoint directory
        dest_dir: target checkpoint directory
        cutoff_timestamp: ISO-format cutoff timestamp

    Returns:
        Dict with removed_nodes, rolled_back_nodes counts
    """
    src_skill_graph = os.path.join(src_dir, "skill_graph")
    dest_skill_graph = os.path.join(dest_dir, "skill_graph")

    if not os.path.exists(src_skill_graph):
        return {"removed_nodes": 0, "rolled_back_nodes": 0}

    # Parse cutoff timestamp
    try:
        cutoff_dt = datetime.fromisoformat(cutoff_timestamp.replace('Z', '+00:00'))
    except:
        # On parse failure, copy the whole directory
        shutil.copytree(src_skill_graph, dest_skill_graph)
        return {"removed_nodes": 0, "rolled_back_nodes": 0}

    result = {
        "removed_nodes": 0,
        "rolled_back_nodes": 0,
        "removed_node_names": [],
        "rolled_back_details": []
    }

    # Create the target directory structure
    os.makedirs(dest_skill_graph, exist_ok=True)

    # Copy directories other than graph.json (code, description, effects, metadata, preconditions)
    subdirs_to_copy = ["code", "description", "effects", "metadata", "preconditions", "optimizer"]
    for subdir in subdirs_to_copy:
        src_subdir = os.path.join(src_skill_graph, subdir)
        dest_subdir = os.path.join(dest_skill_graph, subdir)
        if os.path.exists(src_subdir):
            shutil.copytree(src_subdir, dest_subdir, dirs_exist_ok=True)

    # Process graph.json
    graph_dir = os.path.join(src_skill_graph, "graph")
    graph_json_path = os.path.join(graph_dir, "graph.json")

    if os.path.exists(graph_json_path):
        with open(graph_json_path, 'r', encoding='utf-8') as f:
            graph_data = json.load(f)

        # Process nodes
        nodes_to_remove = []
        nodes = graph_data.get("nodes", {})

        for node_name, node_data in list(nodes.items()):
            versions = node_data.get("versions", [])
            if not versions:
                continue

            # Get the creation time of the first version
            first_version = versions[0]
            created_at_str = first_version.get("created_at", "")

            try:
                created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))

                if created_at > cutoff_dt:
                    # Node created after cutoff; mark for deletion
                    nodes_to_remove.append(node_name)
                else:
                    # Check whether the version should be rolled back
                    valid_versions = []
                    for v in versions:
                        v_created_at_str = v.get("created_at", "")
                        try:
                            v_created_at = datetime.fromisoformat(v_created_at_str.replace('Z', '+00:00'))
                            if v_created_at <= cutoff_dt:
                                valid_versions.append(v)
                        except:
                            valid_versions.append(v)

                    if len(valid_versions) < len(versions):
                        # Needs rollback
                        node_data["versions"] = valid_versions
                        result["rolled_back_nodes"] += 1
                        result["rolled_back_details"].append({
                            "node": node_name,
                            "original_versions": len(versions),
                            "kept_versions": len(valid_versions)
                        })
            except:
                # Parse failed; keep the node
                pass

        # Delete nodes
        for node_name in nodes_to_remove:
            del nodes[node_name]
            result["removed_nodes"] += 1
            result["removed_node_names"].append(node_name)

            # Also delete related files
            for subdir in ["code", "description", "effects", "metadata", "preconditions"]:
                for ext in [".js", ".txt", ".json"]:
                    file_path = os.path.join(dest_skill_graph, subdir, f"{node_name}{ext}")
                    if os.path.exists(file_path):
                        os.remove(file_path)

        # Clean invalid references in edges
        edges = graph_data.get("edges", {})
        for parent in list(edges.keys()):
            if parent in nodes_to_remove:
                del edges[parent]
            else:
                edges[parent] = [child for child in edges[parent] if child not in nodes_to_remove]

        # Save the processed graph.json
        dest_graph_dir = os.path.join(dest_skill_graph, "graph")
        os.makedirs(dest_graph_dir, exist_ok=True)

        with open(os.path.join(dest_graph_dir, "graph.json"), 'w', encoding='utf-8') as f:
            json.dump(graph_data, f, ensure_ascii=False, indent=2)

    # Note: do not copy vectordb; let it auto-rebuild on resume
    # This keeps vectordb in sync with graph.json

    return result


def prepare_iteration_checkpoint(
    source_dir: str,
    dest_dir: str,
    target_iteration: int,
    domain_knowledge=None,
) -> Dict[str, Any]:
    """
    Prepare a new truncated checkpoint directory (core function).

    Args:
        source_dir: source checkpoint directory
        dest_dir: target checkpoint directory
        target_iteration: target iteration

    Returns:
        Dict with preparation statistics
    """
    result = {
        "source_dir": source_dir,
        "dest_dir": dest_dir,
        "target_iteration": target_iteration,
        "success": False,
        "error": None
    }

    try:
        # 1. Check the source directory
        if not os.path.exists(source_dir):
            result["error"] = f"Source directory does not exist: {source_dir}"
            return result

        # 2. Get cutoff info
        cutoff_info = get_iteration_cutoff_info(source_dir, target_iteration)
        if not cutoff_info:
            result["error"] = f"Cannot find iteration {target_iteration} in iteration_log"
            return result

        actual_iteration = cutoff_info.get("iteration", target_iteration)
        cutoff_timestamp = cutoff_info.get("timestamp", "")
        cutoff_inventory = cutoff_info.get("inventory_snapshot", {})

        result["actual_iteration"] = actual_iteration
        result["cutoff_timestamp"] = cutoff_timestamp

        print(f"\033[36m[Iteration Resume] Preparing checkpoint from iteration {actual_iteration}\033[0m")
        print(f"\033[36m[Iteration Resume] Cutoff timestamp: {cutoff_timestamp}\033[0m")

        # 3. If the target directory exists, remove it first
        if os.path.exists(dest_dir):
            shutil.rmtree(dest_dir)

        # 4. Create the target directory structure
        os.makedirs(dest_dir, exist_ok=True)

        # 5. Copy event files (only those with timestamps <= cutoff)
        events_to_keep = get_event_files_before_timestamp(source_dir, cutoff_timestamp)
        if events_to_keep:
            dest_events_dir = os.path.join(dest_dir, "events")
            os.makedirs(dest_events_dir, exist_ok=True)

            for filename in events_to_keep:
                src_path = os.path.join(source_dir, "events", filename)
                dest_path = os.path.join(dest_events_dir, filename)
                shutil.copy2(src_path, dest_path)

        result["events_copied"] = len(events_to_keep)
        print(f"\033[32m[Iteration Resume] Copied {len(events_to_keep)} event files\033[0m")

        # 6. Truncate and copy progress/*.jsonl
        iteration_filter = lambda r: r.get("iteration", 0) <= actual_iteration

        progress_files = ["iteration_log.jsonl", "skill_events.jsonl", "resource_stats.jsonl", "learning_dynamics.jsonl"]
        progress_stats = {}

        for filename in progress_files:
            src_path = os.path.join(source_dir, "progress", filename)
            dest_path = os.path.join(dest_dir, "progress", filename)

            if os.path.exists(src_path):
                count = truncate_jsonl_file(src_path, dest_path, iteration_filter)
                progress_stats[filename] = count

        result["progress_stats"] = progress_stats
        print(f"\033[32m[Iteration Resume] Truncated progress files: {progress_stats}\033[0m")

        # 7. Rebuild curriculum/*.json from iteration_log
        records = parse_iteration_log(source_dir, max_iteration=actual_iteration)
        curriculum_stats = rebuild_curriculum_from_iteration_log(records, dest_dir)
        result["curriculum_stats"] = curriculum_stats
        print(f"\033[32m[Iteration Resume] Rebuilt curriculum: {curriculum_stats}\033[0m")

        # 8. Copy and process skill_graph/
        skill_graph_stats = truncate_skill_graph(source_dir, dest_dir, cutoff_timestamp)
        result["skill_graph_stats"] = skill_graph_stats
        print(f"\033[32m[Iteration Resume] Truncated skill_graph: "
              f"removed {skill_graph_stats['removed_nodes']} nodes, "
              f"rolled back {skill_graph_stats['rolled_back_nodes']} nodes\033[0m")

        # 9. Rebuild psn_curriculum/progress.json from inventory
        psn_milestones = rebuild_psn_progress_from_inventory(
            cutoff_inventory, dest_dir, domain_knowledge=domain_knowledge
        )
        result["psn_milestones"] = list(psn_milestones)
        print(f"\033[32m[Iteration Resume] Rebuilt PSN milestones: {psn_milestones}\033[0m")

        # 10. Copy other required files
        other_dirs = ["action", "stats", "logs", "trajectories", "transactions"]
        for dirname in other_dirs:
            src_path = os.path.join(source_dir, dirname)
            dest_path = os.path.join(dest_dir, dirname)
            if os.path.exists(src_path) and os.path.isdir(src_path):
                shutil.copytree(src_path, dest_path, dirs_exist_ok=True)

        result["success"] = True
        print(f"\033[35m[Iteration Resume] Successfully created checkpoint at {dest_dir}\033[0m")

    except Exception as e:
        result["error"] = str(e)
        print(f"\033[31m[Iteration Resume] Error: {e}\033[0m")
        import traceback
        traceback.print_exc()

    return result
