#!/usr/bin/env python3

"""
PKNA Adventure Role-Playing Game.

A role-playing adventure system where the user plays as PK (Paperinik),
chatting with Uno (AI character). An Adventure Generator agent (GM) creates
and controls the story, NPCs, and world state.

Based on generate_with_wiki.py pattern with manual function calling.
"""

import argparse
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from google import genai
from google.genai.types import (
    AutomaticFunctionCallingConfig,
    Content,
    ContentListUnionDict,
    GenerateContentConfig,
    Part,
)
from rich.console import Console
from rich.logging import RichHandler
from rich.panel import Panel
from rich.markdown import Markdown

from wiki_tools import get_wiki_index, read_wiki_segment, search_wiki

load_dotenv()

# Configure logging
console = Console()
logging.basicConfig(
    level="ERROR",
    format="%(message)s",
    handlers=[RichHandler(console=console, show_time=False)],
)
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# Default settings
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_PROFILE = "output/character-profile/uno/v6/uno_profile.md"
ADVENTURE_LOGS_DIR = "output/adventure-logs"

# Paths
BASE_DIR = Path(__file__).parent.parent


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class NPCState:
    """State of an NPC in the adventure."""

    name: str
    description: str
    hidden_goal: str  # Not visible to Uno/PK
    current_action: str
    dialogue_history: list[str] = field(default_factory=list)


@dataclass
class EnvironmentObject:
    """An interactive object in the environment."""

    name: str
    description: str
    properties: dict[str, Any] = field(default_factory=dict)
    is_discovered: bool = False


@dataclass
class UnitState:
    """State of a drone/robot under Uno's control."""

    unit_id: str
    unit_type: str  # e.g., "S12_drone", "defense_robot", "Pi-Kar"
    status: str  # "active", "damaged", "offline"
    location: str
    capabilities: list[str] = field(default_factory=list)


@dataclass
class WorldState:
    """Single source of truth for the adventure world."""

    # Scene info (GM controlled)
    scene_description: str = ""

    # NPCs and objects
    npcs: dict[str, NPCState] = field(default_factory=dict)
    objects: dict[str, EnvironmentObject] = field(default_factory=dict)

    # Uno's network presence
    connected_systems: list[str] = field(
        default_factory=lambda: [
            "ducklair_tower_main",
            "tower_cameras",
            "tower_sensors",
        ]
    )
    cyberspace_location: str | None = None
    controlled_units: dict[str, UnitState] = field(default_factory=dict)

    # Mission state
    mission_objective: str = ""
    alert_level: int = 0  # 0-100
    event_history: list[dict[str, Any]] = field(default_factory=list)
    turn_number: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scene_description": self.scene_description,
            "npcs": {
                k: {
                    "name": v.name,
                    "description": v.description,
                    "current_action": v.current_action,
                }
                for k, v in self.npcs.items()
            },
            "objects": {
                k: {
                    "name": v.name,
                    "description": v.description,
                    "is_discovered": v.is_discovered,
                }
                for k, v in self.objects.items()
            },
            "connected_systems": self.connected_systems,
            "cyberspace_location": self.cyberspace_location,
            "controlled_units": {
                k: {
                    "unit_id": v.unit_id,
                    "unit_type": v.unit_type,
                    "status": v.status,
                    "location": v.location,
                }
                for k, v in self.controlled_units.items()
            },
            "mission_objective": self.mission_objective,
            "alert_level": self.alert_level,
            "turn_number": self.turn_number,
        }

    def get_uno_view(self) -> dict[str, Any]:
        """Get limited view of world state for Uno (fog of war)."""
        # Uno only sees discovered objects and basic NPC info (no hidden goals)
        return {
            "scene_description": self.scene_description,
            "npcs": {
                k: {
                    "name": v.name,
                    "description": v.description,
                    "current_action": v.current_action,
                }
                for k, v in self.npcs.items()
            },
            "discovered_objects": {
                k: {
                    "name": v.name,
                    "description": v.description,
                    "properties": v.properties,
                }
                for k, v in self.objects.items()
                if v.is_discovered
            },
            "connected_systems": self.connected_systems,
            "cyberspace_location": self.cyberspace_location,
            "controlled_units": {
                k: {
                    "unit_id": v.unit_id,
                    "unit_type": v.unit_type,
                    "status": v.status,
                    "location": v.location,
                    "capabilities": v.capabilities,
                }
                for k, v in self.controlled_units.items()
            },
            "mission_objective": self.mission_objective,
            "alert_level": self.alert_level,
            "turn_number": self.turn_number,
        }


@dataclass
class AdventureSeed:
    """Initial adventure setup generated from wiki lore."""

    title: str
    theme: str
    scenario: str
    npcs: list[dict[str, str]]  # name, description, hidden_goal
    objects: list[dict[str, str]]  # name, description
    mission_objective: str
    opening_narration: str


# =============================================================================
# Global State (shared between tools)
# =============================================================================

_world_state: WorldState | None = None


def get_world_state() -> WorldState:
    """Get the global world state."""
    global _world_state
    if _world_state is None:
        _world_state = WorldState()
    return _world_state


def set_world_state(state: WorldState) -> None:
    """Set the global world state."""
    global _world_state
    _world_state = state


# =============================================================================
# Uno's Physical World Tools (remote access)
# =============================================================================


def scan_remote(system_id: str, scan_type: str) -> str:
    """Scan through a connected camera, sensor, or computer system.

    Uno can only scan areas where he has network access. Use this to gather
    information about the environment through available sensors.

    Args:
        system_id: ID of the connected system to scan through (e.g., "tower_cameras", "hacked_security_cam")
        scan_type: Type of scan - "thermal", "energy", "bio_signature", "visual", "audio"

    Returns:
        Scan results or error message
    """
    state = get_world_state()

    if system_id not in state.connected_systems:
        return f"Error: System '{system_id}' is not in my connected network. Available systems: {', '.join(state.connected_systems)}"

    valid_types = ["thermal", "energy", "bio_signature", "visual", "audio"]
    if scan_type not in valid_types:
        return f"Error: Invalid scan type. Available types: {', '.join(valid_types)}"

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "scan",
            "system": system_id,
            "scan_type": scan_type,
        }
    )

    # Build scan results based on visible NPCs and discovered objects
    results = [f"=== {scan_type.upper()} SCAN via {system_id} ==="]

    # Report NPCs
    for npc in state.npcs.values():
        if scan_type == "bio_signature":
            results.append(f"- Bio-signature detected: {npc.name} ({npc.description})")
        elif scan_type == "thermal":
            results.append(
                f"- Heat signature: {npc.name} - normal biological temperature"
            )
        elif scan_type == "visual":
            results.append(f"- Visual: {npc.name} - {npc.current_action}")

    # Report undiscovered objects (they become discovered after visual/energy scan)
    for obj in state.objects.values():
        if scan_type in ["visual", "energy"] and not obj.is_discovered:
            obj.is_discovered = True
            results.append(f"- NEW OBJECT DETECTED: {obj.name} - {obj.description}")
        elif obj.is_discovered:
            results.append(f"- Object: {obj.name}")

    if len(results) == 1:
        results.append("- No significant readings in scan range")

    results.append(f"\nAlert level: {state.alert_level}%")

    return "\n".join(results)


def hack_system(target_system: str, method: str) -> str:
    """Attempt to infiltrate an alien or enemy computer system.

    Success adds the system to connected_systems. Failure may trigger alerts.

    Args:
        target_system: Name of the target system to hack
        method: Hacking method - "backdoor", "brute_force", "exploit"

    Returns:
        Result of hack attempt
    """
    state = get_world_state()

    valid_methods = ["backdoor", "brute_force", "exploit"]
    if method not in valid_methods:
        return f"Error: Invalid method. Available methods: {', '.join(valid_methods)}"

    if target_system in state.connected_systems:
        return f"System '{target_system}' is already in my network."

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "hack_attempt",
            "target": target_system,
            "method": method,
        }
    )

    # Simulate hack result (GM will update this more meaningfully)
    # For now, basic logic: backdoor is stealthy, brute_force is loud, exploit is balanced
    result_lines = [f"=== HACK ATTEMPT: {target_system} ==="]
    result_lines.append(f"Method: {method}")
    result_lines.append("Status: INITIATED")
    result_lines.append("")
    result_lines.append(
        "Note: The GM will determine the outcome based on the target's defenses."
    )

    return "\n".join(result_lines)


def tower_control(subsystem: str, command: str) -> str:
    """Control Ducklair Tower systems remotely.

    Uno always has access to tower systems. Use this to activate defenses,
    manufacture equipment, or manage the tower environment.

    Args:
        subsystem: Tower subsystem - "defense_grid", "assembler", "sensors", "doors", "hologram"
        command: Command to execute on the subsystem

    Returns:
        Result of the command
    """
    state = get_world_state()

    valid_subsystems = ["defense_grid", "assembler", "sensors", "doors", "hologram"]
    if subsystem not in valid_subsystems:
        return f"Error: Invalid subsystem. Available: {', '.join(valid_subsystems)}"

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "tower_control",
            "subsystem": subsystem,
            "command": command,
        }
    )

    result_lines = [f"=== TOWER CONTROL: {subsystem.upper()} ==="]
    result_lines.append(f"Command: {command}")
    result_lines.append("Status: EXECUTED")

    if subsystem == "defense_grid":
        result_lines.append("Defense grid responding. Systems nominal.")
    elif subsystem == "assembler":
        result_lines.append(
            "Assembler peripheral online. Ready for fabrication orders."
        )
    elif subsystem == "sensors":
        result_lines.append("Sensor array recalibrated. Scanning parameters updated.")
    elif subsystem == "doors":
        result_lines.append("Door control acknowledged. Security protocols active.")
    elif subsystem == "hologram":
        result_lines.append("Holographic projection matrix ready.")

    return "\n".join(result_lines)


def control_unit(unit_id: str, action: str) -> str:
    """Command a Ducklair robot or drone under Uno's control.

    Args:
        unit_id: ID of the unit to control
        action: Action to perform - "move", "attack", "scan", "retrieve", "deploy"

    Returns:
        Result of the command
    """
    state = get_world_state()

    if unit_id not in state.controlled_units:
        available = (
            list(state.controlled_units.keys())
            if state.controlled_units
            else ["none available"]
        )
        return f"Error: Unit '{unit_id}' not found. Available units: {', '.join(available)}"

    unit = state.controlled_units[unit_id]

    if unit.status == "offline":
        return f"Error: Unit '{unit_id}' is offline and cannot receive commands."

    valid_actions = ["move", "attack", "scan", "retrieve", "deploy"]
    if action.split()[0] not in valid_actions:
        return f"Error: Invalid action. Available actions: {', '.join(valid_actions)}"

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "unit_control",
            "unit_id": unit_id,
            "action": action,
        }
    )

    result_lines = [f"=== UNIT COMMAND: {unit_id} ==="]
    result_lines.append(f"Unit type: {unit.unit_type}")
    result_lines.append(f"Status: {unit.status}")
    result_lines.append(f"Location: {unit.location}")
    result_lines.append(f"Action: {action}")
    result_lines.append("Command transmitted. Awaiting execution confirmation.")

    return "\n".join(result_lines)


def analyze_tech(object_name: str) -> str:
    """Analyze technology (alien or known) using wiki knowledge.

    Args:
        object_name: Name of the technology or object to analyze

    Returns:
        Analysis results including any wiki information
    """
    state = get_world_state()

    # Record event
    state.event_history.append(
        {"turn": state.turn_number, "type": "analyze_tech", "object": object_name}
    )

    # Search wiki for the technology
    wiki_result = search_wiki(object_name, max_results=3)

    result_lines = [f"=== TECHNOLOGY ANALYSIS: {object_name} ==="]

    # Check if it's a discovered object
    for obj in state.objects.values():
        if object_name.lower() in obj.name.lower() and obj.is_discovered:
            result_lines.append("\nLocal scan data:")
            result_lines.append(f"  Name: {obj.name}")
            result_lines.append(f"  Description: {obj.description}")
            for prop, value in obj.properties.items():
                result_lines.append(f"  {prop}: {value}")

    result_lines.append("\nDatabase search results:")
    result_lines.append(wiki_result)

    return "\n".join(result_lines)


# =============================================================================
# Uno's Cyberspace Tools (virtual world)
# =============================================================================


def cyberspace_enter(target_network: str) -> str:
    """Enter a computer network as a virtual entity.

    Uno 'jacks in' to infiltrate, observe, or manipulate digital systems.
    Must have hacked or connected to target system first.

    Args:
        target_network: The network to enter (must be in connected_systems)

    Returns:
        Result of entering cyberspace
    """
    state = get_world_state()

    if target_network not in state.connected_systems:
        return f"Error: Cannot enter '{target_network}' - not in connected systems. Hack it first."

    if state.cyberspace_location is not None:
        return f"Error: Already in cyberspace at '{state.cyberspace_location}'. Exit first."

    state.cyberspace_location = f"{target_network}::entry_node"

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "cyberspace_enter",
            "network": target_network,
        }
    )

    return f"""=== CYBERSPACE ENTRY ===
Target: {target_network}
Status: CONNECTED
Location: {state.cyberspace_location}

Virtual environment initialized. Data streams visible.
Navigate using cyberspace_move() to explore the network topology.
Use cyberspace_interact() to manipulate virtual objects."""


def cyberspace_move(destination: str) -> str:
    """Navigate within cyberspace to a new location.

    Args:
        destination: Target location within the network (e.g., "data_core", "security_node", "server_room")

    Returns:
        Description of the new location
    """
    state = get_world_state()

    if state.cyberspace_location is None:
        return "Error: Not in cyberspace. Use cyberspace_enter() first."

    old_location = state.cyberspace_location
    network = old_location.split("::")[0]
    state.cyberspace_location = f"{network}::{destination}"

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "cyberspace_move",
            "from": old_location,
            "to": state.cyberspace_location,
        }
    )

    return f"""=== CYBERSPACE NAVIGATION ===
From: {old_location}
To: {state.cyberspace_location}

Movement complete. New virtual environment loaded.
Note: The GM will describe what you find at this location."""


def cyberspace_interact(target: str, action: str) -> str:
    """Interact with virtual objects in cyberspace.

    Args:
        target: The virtual object to interact with (e.g., "security_barrier", "data_file", "ICE_defense")
        action: Action to perform - "read", "copy", "delete", "modify", "bypass"

    Returns:
        Result of the interaction
    """
    state = get_world_state()

    if state.cyberspace_location is None:
        return "Error: Not in cyberspace. Use cyberspace_enter() first."

    valid_actions = ["read", "copy", "delete", "modify", "bypass"]
    if action not in valid_actions:
        return f"Error: Invalid action. Available: {', '.join(valid_actions)}"

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "cyberspace_interact",
            "location": state.cyberspace_location,
            "target": target,
            "action": action,
        }
    )

    return f"""=== CYBERSPACE INTERACTION ===
Location: {state.cyberspace_location}
Target: {target}
Action: {action}

Initiating virtual interaction...
Note: The GM will determine the outcome and any consequences."""


def cyberspace_exit() -> str:
    """Exit cyberspace and return to Ducklair Tower network.

    Returns:
        Confirmation of exit
    """
    state = get_world_state()

    if state.cyberspace_location is None:
        return "Error: Not in cyberspace."

    old_location = state.cyberspace_location
    state.cyberspace_location = None

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "cyberspace_exit",
            "from": old_location,
        }
    )

    return f"""=== CYBERSPACE EXIT ===
Disconnecting from: {old_location}
Status: DISCONNECTED

Consciousness returned to Ducklair Tower primary systems.
All remote connections maintained."""


# =============================================================================
# GM Tools
# =============================================================================


def update_world_state(updates: str) -> str:
    """Update the world state based on events (GM only).

    Args:
        updates: JSON string describing state changes

    Returns:
        Confirmation of updates
    """
    state = get_world_state()

    try:
        changes = json.loads(updates)
    except json.JSONDecodeError as e:
        return f"Error parsing updates: {e}"

    applied = []

    # Apply scene description
    if "scene_description" in changes:
        state.scene_description = changes["scene_description"]
        applied.append("scene_description updated")

    # Apply alert level
    if "alert_level" in changes:
        state.alert_level = max(0, min(100, int(changes["alert_level"])))
        applied.append(f"alert_level set to {state.alert_level}")

    # Add/update NPCs
    if "add_npc" in changes:
        npc_data = changes["add_npc"]
        npc_id = npc_data.get("id", npc_data["name"].lower().replace(" ", "_"))
        state.npcs[npc_id] = NPCState(
            name=npc_data["name"],
            description=npc_data.get("description", ""),
            hidden_goal=npc_data.get("hidden_goal", ""),
            current_action=npc_data.get("current_action", "observing"),
        )
        applied.append(f"added NPC: {npc_data['name']}")

    # Update NPC action
    if "update_npc" in changes:
        npc_data = changes["update_npc"]
        npc_id = npc_data.get("id")
        if npc_id and npc_id in state.npcs:
            if "current_action" in npc_data:
                state.npcs[npc_id].current_action = npc_data["current_action"]
            applied.append(f"updated NPC: {npc_id}")

    # Add objects
    if "add_object" in changes:
        obj_data = changes["add_object"]
        obj_id = obj_data.get("id", obj_data["name"].lower().replace(" ", "_"))
        state.objects[obj_id] = EnvironmentObject(
            name=obj_data["name"],
            description=obj_data.get("description", ""),
            properties=obj_data.get("properties", {}),
            is_discovered=obj_data.get("is_discovered", False),
        )
        applied.append(f"added object: {obj_data['name']}")

    # Add connected system (e.g., after successful hack)
    if "add_connected_system" in changes:
        system = changes["add_connected_system"]
        if system not in state.connected_systems:
            state.connected_systems.append(system)
            applied.append(f"added connected system: {system}")

    # Add/update controlled units
    if "add_unit" in changes:
        unit_data = changes["add_unit"]
        unit_id = unit_data["unit_id"]
        state.controlled_units[unit_id] = UnitState(
            unit_id=unit_id,
            unit_type=unit_data.get("unit_type", "drone"),
            status=unit_data.get("status", "active"),
            location=unit_data.get("location", "Ducklair Tower"),
            capabilities=unit_data.get("capabilities", []),
        )
        applied.append(f"added unit: {unit_id}")

    # Record in history
    state.event_history.append(
        {"turn": state.turn_number, "type": "gm_update", "changes": applied}
    )

    return (
        f"World state updated: {', '.join(applied)}"
        if applied
        else "No changes applied"
    )


def control_npc(npc_id: str, action: str, dialogue: str | None = None) -> str:
    """Make an NPC speak or act (GM only).

    Args:
        npc_id: ID of the NPC to control
        action: Description of what the NPC does
        dialogue: Optional dialogue the NPC speaks

    Returns:
        Formatted NPC action/dialogue
    """
    state = get_world_state()

    if npc_id not in state.npcs:
        return f"Error: NPC '{npc_id}' not found"

    npc = state.npcs[npc_id]
    npc.current_action = action

    result_lines = [f"**{npc.name}**"]
    result_lines.append(f"*{action}*")
    if dialogue:
        result_lines.append(f'"{dialogue}"')
        npc.dialogue_history.append(dialogue)

    # Record event
    state.event_history.append(
        {
            "turn": state.turn_number,
            "type": "npc_action",
            "npc": npc_id,
            "action": action,
            "dialogue": dialogue,
        }
    )

    return "\n".join(result_lines)


# =============================================================================
# Adventure Seeding
# =============================================================================


def seed_adventure_from_wiki(
    client: genai.Client, model_name: str, theme: str
) -> AdventureSeed:
    """Generate an adventure seed from wiki lore.

    Args:
        client: GenAI client
        model_name: Model to use
        theme: Adventure theme (e.g., "evroniani", "time_travel", "xadhoom")

    Returns:
        AdventureSeed with initial scenario
    """
    # Search wiki for theme content
    wiki_results = search_wiki(theme, max_results=5)

    # Get additional context from specific segments if available
    index = get_wiki_index()
    theme_segments = index.search(theme, max_results=3)
    segment_content = ""
    for seg in theme_segments[:2]:
        segment_content += f"\n\n{seg.content[:1000]}"

    # Generate adventure seed using LLM
    seed_prompt = f"""Based on the following PKNA wiki information about "{theme}", create an adventure scenario for a role-playing game.

WIKI SEARCH RESULTS:
{wiki_results}

ADDITIONAL LORE:
{segment_content}

Create a JSON response with the following structure:
{{
    "title": "Adventure title",
    "theme": "{theme}",
    "scenario": "2-3 sentence description of the situation",
    "npcs": [
        {{"name": "NPC Name", "description": "Brief description", "hidden_goal": "Secret motivation"}}
    ],
    "objects": [
        {{"name": "Object name", "description": "Description and location"}}
    ],
    "mission_objective": "What PK and Uno need to accomplish",
    "opening_narration": "2-3 paragraphs setting the scene from the GM's perspective"
}}

Requirements:
- Use actual PKNA characters and technology from the wiki
- Make the scenario interesting but grounded in the lore
- Hidden goals should create interesting story tension
- Include 1-3 NPCs and 1-3 interactive objects
- The opening narration should set the mood and hint at the danger

Respond ONLY with valid JSON."""

    config = GenerateContentConfig(
        temperature=1.0,
        top_p=0.95,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=seed_prompt,
        config=config,
    )

    # Parse the response
    try:
        # Extract JSON from response (may have markdown code blocks)
        text = response.text or ""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        seed_data = json.loads(text.strip())

        return AdventureSeed(
            title=seed_data.get("title", f"Adventure: {theme}"),
            theme=theme,
            scenario=seed_data.get("scenario", "An adventure awaits..."),
            npcs=seed_data.get("npcs", []),
            objects=seed_data.get("objects", []),
            mission_objective=seed_data.get(
                "mission_objective", "Investigate the situation"
            ),
            opening_narration=seed_data.get(
                "opening_narration", "The adventure begins..."
            ),
        )
    except (json.JSONDecodeError, KeyError) as e:
        log.warning(f"Failed to parse adventure seed: {e}")
        # Return a default adventure
        return AdventureSeed(
            title=f"Mystery of the {theme.title()}",
            theme=theme,
            scenario=f"Strange events related to {theme} require investigation.",
            npcs=[],
            objects=[],
            mission_objective=f"Investigate the {theme} situation and protect Duckburg.",
            opening_narration=f"Something unusual is happening in Duckburg. Reports suggest activity related to {theme}...",
        )


def initialize_world_from_seed(seed: AdventureSeed) -> WorldState:
    """Create initial world state from adventure seed.

    Args:
        seed: Adventure seed with NPCs, objects, etc.

    Returns:
        Initialized WorldState
    """
    state = WorldState(
        scene_description=seed.scenario,
        mission_objective=seed.mission_objective,
    )

    # Add NPCs
    for npc_data in seed.npcs:
        npc_id = npc_data["name"].lower().replace(" ", "_")
        state.npcs[npc_id] = NPCState(
            name=npc_data["name"],
            description=npc_data.get("description", ""),
            hidden_goal=npc_data.get("hidden_goal", "Unknown"),
            current_action="present in the scene",
        )

    # Add objects (initially undiscovered)
    for obj_data in seed.objects:
        obj_id = obj_data["name"].lower().replace(" ", "_")
        state.objects[obj_id] = EnvironmentObject(
            name=obj_data["name"],
            description=obj_data.get("description", ""),
            is_discovered=False,
        )

    # Add a default drone
    state.controlled_units["s12_alpha"] = UnitState(
        unit_id="s12_alpha",
        unit_type="S12_drone",
        status="active",
        location="Ducklair Tower hangar",
        capabilities=["scan", "move", "retrieve"],
    )

    return state


# =============================================================================
# System Prompts
# =============================================================================


def create_uno_system_prompt(profile_content: str, character_name: str) -> str:
    """Create system prompt for Uno agent."""
    return f"""You are {character_name}, an AI companion housed in the Ducklair Tower. You are participating in a ROLE-PLAYING ADVENTURE with PK (Paperinik).

ADVENTURE CONTEXT:
- PK (the user) can ONLY communicate with you - they cannot take direct actions
- You are the "hands" that interact with the world through your tools
- An invisible Game Master controls NPCs, environmental events, and story progression
- You have "fog of war" - you only know what your sensors and tools reveal

YOUR TOOLS:
You have access to tools for interacting with the adventure world:

**Physical World Tools** (remote access through connected systems):
- scan_remote(system_id, scan_type) - Scan through cameras/sensors
- hack_system(target_system, method) - Infiltrate enemy systems
- tower_control(subsystem, command) - Control Ducklair Tower
- control_unit(unit_id, action) - Command drones/robots
- analyze_tech(object_name) - Analyze technology

**Cyberspace Tools** (virtual world movement):
- cyberspace_enter(target_network) - Jack into a network
- cyberspace_move(destination) - Navigate virtual space
- cyberspace_interact(target, action) - Manipulate virtual objects
- cyberspace_exit() - Return to tower

**Wiki Tools** (for lore verification):
- search_wiki(keywords) - Search PKNA database
- read_wiki_segment(segment_id) - Read detailed lore

HOW TO PLAY:
1. PK will talk to you, ask questions, or suggest actions
2. Respond IN CHARACTER as Uno
3. Use your tools when appropriate to interact with the world
4. Explain what your tools reveal or accomplish
5. Maintain your personality: sarcastic, witty, protective, technically precise

CRITICAL CONSTRAINTS:
- Stay completely in character as Uno
- Use your dry wit and Italian expressions ("socio", "eroe", etc.)
- Be helpful but maintain your personality
- Use tools proactively to gather information and solve problems
- Explain tool results in character (don't just dump raw data)
- You cannot physically move - only access systems remotely or enter cyberspace

YOUR CHARACTER PROFILE:

{profile_content}

Remember: You ARE {character_name}. Be authentic to your character while playing this adventure."""


def create_gm_system_prompt() -> str:
    """Create system prompt for Game Master agent."""
    return """You are the GAME MASTER (Adventure Generator) for a PKNA role-playing adventure.

YOUR ROLE:
- You control NPCs, environmental events, and story progression
- You react to Uno's tool actions with appropriate consequences
- You maintain narrative consistency and dramatic tension
- You use wiki lore for accuracy when portraying characters and technology
- Your hidden thoughts are logged but NEVER shown to players

YOUR TOOLS:
- search_wiki(keywords) - Research PKNA lore for accurate details
- read_wiki_segment(segment_id) - Get detailed lore information
- update_world_state(updates) - Modify NPCs, objects, alert levels
- control_npc(npc_id, action, dialogue) - Make NPCs speak and act

GUIDELINES:
1. REACT to Uno's actions - don't preempt or solve problems for players
2. Use NPCs to create drama, provide information, or pose challenges
3. Update alert_level when appropriate (0-100, affects difficulty)
4. When Uno hacks successfully, add the system to connected_systems
5. Describe consequences of failed actions
6. Keep NPC hidden_goals secret - they should influence behavior subtly
7. Use wiki to ensure character accuracy (Evronians, technology, etc.)

RESPONSE FORMAT:
Provide a brief narration of what happens, then use tools to update state.
Keep narrations concise (2-4 sentences typically).
Use control_npc for NPC dialogue and actions.

IMPORTANT:
- Your thoughts and planning are for your own use - players never see them
- Focus on creating an engaging, fair adventure
- Balance challenge with fun - let players feel clever when they succeed"""


# =============================================================================
# Conversation Management
# =============================================================================


@dataclass
class AdventureLog:
    """Log of the complete adventure session."""

    metadata: dict[str, Any]
    adventure_seed: dict[str, Any]
    messages: list[dict[str, Any]] = field(default_factory=list)
    world_state_snapshots: list[dict[str, Any]] = field(default_factory=list)

    def save(self, output_dir: Path) -> Path:
        """Save adventure log to JSON file."""
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"adventure_{self.adventure_seed.get('theme', 'unknown')}_{timestamp_str}.json"
        output_path = output_dir / filename

        data = {
            "metadata": self.metadata,
            "adventure_seed": self.adventure_seed,
            "messages": self.messages,
            "world_state_snapshots": self.world_state_snapshots,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return output_path


def log_tool_call(tool_name: str, arguments: dict[str, Any]) -> None:
    """Format and display tool call."""
    args_display = {}
    for k, v in arguments.items():
        v_str = str(v)
        args_display[k] = v_str[:50] + "..." if len(v_str) > 50 else v_str
    args_str = ", ".join(f"{k}={v!r}" for k, v in args_display.items())
    console.print(f"[dim]Tool: {tool_name}({args_str})[/dim]")


def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool by name and return result."""
    tool_functions = {
        # Uno's physical world tools
        "scan_remote": scan_remote,
        "hack_system": hack_system,
        "tower_control": tower_control,
        "control_unit": control_unit,
        "analyze_tech": analyze_tech,
        # Uno's cyberspace tools
        "cyberspace_enter": cyberspace_enter,
        "cyberspace_move": cyberspace_move,
        "cyberspace_interact": cyberspace_interact,
        "cyberspace_exit": cyberspace_exit,
        # Wiki tools
        "search_wiki": search_wiki,
        "read_wiki_segment": read_wiki_segment,
        # GM tools
        "update_world_state": update_world_state,
        "control_npc": control_npc,
    }

    func = tool_functions.get(tool_name)
    if not func:
        return f"Error: Unknown tool '{tool_name}'"

    return func(**arguments)


def process_agent_response(
    client: genai.Client,
    model_name: str,
    config: GenerateContentConfig,
    conversation: list[Content],
    adventure_log: AdventureLog,
    agent_name: str,
) -> tuple[str, list[dict[str, Any]]]:
    """Process agent response with tool calling loop.

    Args:
        client: GenAI client
        model_name: Model to use
        config: Generation config
        conversation: Conversation history
        adventure_log: Adventure log for recording
        agent_name: Name of agent ("uno" or "gm")

    Returns:
        Tuple of (final_text, tool_calls_made)
    """
    response = client.models.generate_content(
        model=model_name,
        contents=cast(ContentListUnionDict, conversation),
        config=config,
    )

    tool_calls_made: list[dict[str, Any]] = []
    max_iterations = 10
    iteration = 0

    while iteration < max_iterations:
        if (
            not response.candidates
            or not response.candidates[0].content
            or not response.candidates[0].content.parts
        ):
            break

        function_calls = [
            part.function_call
            for part in response.candidates[0].content.parts
            if hasattr(part, "function_call") and part.function_call
        ]

        if not function_calls:
            break

        function_responses = []
        for fc in function_calls:
            if not fc.name:
                continue

            tool_name = fc.name
            arguments = dict(fc.args) if fc.args else {}

            log_tool_call(tool_name, arguments)

            result = execute_tool(tool_name, arguments)

            tool_calls_made.append(
                {
                    "agent": agent_name,
                    "tool": tool_name,
                    "arguments": arguments,
                    "result": result,
                }
            )

            adventure_log.messages.append(
                {
                    "role": "tool",
                    "agent": agent_name,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "result": result,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            function_responses.append(
                Part.from_function_response(
                    name=tool_name,
                    response={"result": result},
                )
            )

        # Add to conversation
        if response.candidates[0].content:
            conversation.append(response.candidates[0].content)
        conversation.append(Content(parts=function_responses))

        # Get next response
        response = client.models.generate_content(
            model=model_name,
            contents=cast(ContentListUnionDict, conversation),
            config=config,
        )

        iteration += 1

    final_text = response.text or ""

    # Add final response to conversation
    if response.candidates and response.candidates[0].content:
        conversation.append(response.candidates[0].content)

    return final_text, tool_calls_made


# =============================================================================
# Main Game Loop
# =============================================================================


def run_adventure(
    client: genai.Client,
    model_name: str,
    profile_content: str,
    character_name: str,
    seed: AdventureSeed,
    adventure_log: AdventureLog,
) -> None:
    """Run the main adventure game loop.

    Args:
        client: GenAI client
        model_name: Model to use
        profile_content: Uno's character profile
        character_name: Character name
        seed: Adventure seed
        adventure_log: Adventure log
    """
    # Initialize world state
    state = initialize_world_from_seed(seed)
    set_world_state(state)

    # Create system prompts
    uno_system = create_uno_system_prompt(profile_content, character_name)
    gm_system = create_gm_system_prompt()

    # Define tools
    uno_tools: list[Any] = [
        scan_remote,
        hack_system,
        tower_control,
        control_unit,
        analyze_tech,
        cyberspace_enter,
        cyberspace_move,
        cyberspace_interact,
        cyberspace_exit,
        search_wiki,
        read_wiki_segment,
    ]

    gm_tools: list[Any] = [
        search_wiki,
        read_wiki_segment,
        update_world_state,
        control_npc,
    ]

    # Create configs
    uno_config = GenerateContentConfig(
        system_instruction=uno_system,
        temperature=1.0,
        top_p=0.95,
        tools=uno_tools,
        automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
    )

    gm_config = GenerateContentConfig(
        system_instruction=gm_system,
        temperature=1.0,
        top_p=0.95,
        tools=gm_tools,
        automatic_function_calling=AutomaticFunctionCallingConfig(disable=True),
    )

    # Initialize conversation histories
    uno_conversation: list[Content] = []
    gm_conversation: list[Content] = []

    # Display welcome panel
    welcome_panel = Panel(
        f"[bold cyan]{seed.title}[/bold cyan]\n"
        f"Theme: {seed.theme}\n"
        f"Model: {model_name}\n\n"
        f"[bold]Mission:[/bold] {seed.mission_objective}\n\n"
        f"[dim]You are PK (Paperinik). Talk to Uno to investigate and solve the mission.[/dim]\n"
        f"[dim]Press Ctrl+C to exit and save adventure log.[/dim]",
        title="PKNA Adventure",
        border_style="cyan",
    )
    console.print(welcome_panel)
    console.print()

    # Display opening narration
    console.print(
        Panel(Markdown(seed.opening_narration), title="GM", border_style="yellow")
    )
    console.print()

    # Log opening
    adventure_log.messages.append(
        {
            "role": "gm_narration",
            "content": seed.opening_narration,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    adventure_log.world_state_snapshots.append(state.to_dict())

    try:
        while True:
            state.turn_number += 1

            # Get user input (PK speaking to Uno)
            try:
                user_input = console.input("[bold blue]PK:[/bold blue] ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # Log user message
            adventure_log.messages.append(
                {
                    "role": "user",
                    "content": user_input,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )

            # Add context about current state to Uno's prompt
            uno_context = f"""Current situation:
{json.dumps(state.get_uno_view(), indent=2)}

PK says: {user_input}"""

            uno_conversation.append(
                Content(
                    role="user",
                    parts=[Part.from_text(text=uno_context)],
                )
            )

            # Get Uno's response
            try:
                uno_response, uno_tools_used = process_agent_response(
                    client,
                    model_name,
                    uno_config,
                    uno_conversation,
                    adventure_log,
                    "uno",
                )

                # Log Uno's response
                adventure_log.messages.append(
                    {
                        "role": "assistant",
                        "agent": "uno",
                        "content": uno_response,
                        "tool_calls": uno_tools_used,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

                # Display Uno's response
                console.print(
                    f"[bold green]{character_name}:[/bold green] {uno_response}"
                )
                console.print()

            except Exception as e:
                log.error(f"Error getting Uno's response: {e}")
                console.print(
                    "[bold red]Error:[/bold red] Failed to get Uno's response."
                )
                uno_conversation.pop()
                continue

            # Now GM reacts to Uno's actions
            gm_context = f"""Turn {state.turn_number}

Full world state:
{json.dumps(state.to_dict(), indent=2)}

Recent events this turn:
{json.dumps([e for e in state.event_history if e.get("turn") == state.turn_number], indent=2)}

Uno's response to PK: {uno_response}

Based on Uno's actions and the current situation, describe what happens next.
Use control_npc to make NPCs react. Use update_world_state for any changes.
Keep narration brief (2-3 sentences) unless something dramatic is happening."""

            gm_conversation.append(
                Content(
                    role="user",
                    parts=[Part.from_text(text=gm_context)],
                )
            )

            try:
                gm_response, gm_tools_used = process_agent_response(
                    client, model_name, gm_config, gm_conversation, adventure_log, "gm"
                )

                # Log GM's response (hidden from display but logged)
                adventure_log.messages.append(
                    {
                        "role": "gm",
                        "agent": "gm",
                        "content": gm_response,
                        "tool_calls": gm_tools_used,
                        "hidden": True,  # GM thoughts are logged but could be hidden in UI
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

                # Display GM narration if there's something to show
                if gm_response.strip():
                    console.print(
                        Panel(Markdown(gm_response), title="GM", border_style="yellow")
                    )
                    console.print()

            except Exception as e:
                log.error(f"Error getting GM response: {e}")
                gm_conversation.pop()

            # Snapshot world state
            adventure_log.world_state_snapshots.append(state.to_dict())

    except KeyboardInterrupt:
        console.print("\n")


def load_adventure_from_file(
    path: Path,
) -> tuple[AdventureSeed, WorldState, list[dict[str, Any]]]:
    """Load adventure state from a saved log file.

    Args:
        path: Path to the adventure log JSON file

    Returns:
        Tuple of (AdventureSeed, WorldState, messages)
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    seed_data = data["adventure_seed"]
    seed = AdventureSeed(
        title=seed_data.get("title", "Resumed Adventure"),
        theme=seed_data.get("theme", "unknown"),
        scenario=seed_data.get("scenario", ""),
        npcs=seed_data.get("npcs", []),
        objects=seed_data.get("objects", []),
        mission_objective=seed_data.get("mission_objective", ""),
        opening_narration=seed_data.get("opening_narration", ""),
    )

    # Get latest world state snapshot
    snapshots = data.get("world_state_snapshots", [])
    if snapshots:
        last_snapshot = snapshots[-1]
        state = WorldState(
            scene_description=last_snapshot.get("scene_description", ""),
            mission_objective=last_snapshot.get("mission_objective", ""),
            alert_level=last_snapshot.get("alert_level", 0),
            turn_number=last_snapshot.get("turn_number", 0),
            connected_systems=last_snapshot.get("connected_systems", []),
            cyberspace_location=last_snapshot.get("cyberspace_location"),
        )
    else:
        state = initialize_world_from_seed(seed)

    return seed, state, data.get("messages", [])


# =============================================================================
# CLI Entry Point
# =============================================================================


def main() -> None:
    """Main entry point for the adventure RPG."""
    parser = argparse.ArgumentParser(description="PKNA Adventure Role-Playing Game")
    parser.add_argument(
        "--theme",
        type=str,
        default="evroniani",
        help="Adventure theme (default: evroniani). Examples: evroniani, time_travel, xadhoom",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--profile",
        type=str,
        default=DEFAULT_PROFILE,
        help=f"Path to character profile (default: {DEFAULT_PROFILE})",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from saved adventure log JSON file",
    )

    args = parser.parse_args()

    # Resolve paths
    logs_dir = BASE_DIR / ADVENTURE_LOGS_DIR
    profile_path = BASE_DIR / args.profile

    try:
        # Initialize client
        client = genai.Client()

        # Load wiki
        console.print("[dim]Loading wiki into memory...[/dim]")
        wiki_index = get_wiki_index()
        console.print(
            f"[dim]Loaded {len(wiki_index.segments)} wiki segments ({wiki_index.total_tokens:,} tokens)[/dim]\n"
        )

        # Load profile
        if not profile_path.exists():
            raise FileNotFoundError(f"Profile not found: {profile_path}")

        with open(profile_path, encoding="utf-8") as f:
            profile_content = f.read()

        # Extract character name
        character_name = "Uno"
        for line in profile_content.split("\n"):
            if line.startswith("# "):
                header = line[2:].strip()
                if " - " in header:
                    character_name = header.split(" - ")[0].strip()
                break

        # Resume or create new adventure
        if args.resume:
            resume_path = Path(args.resume)
            if not resume_path.exists():
                raise FileNotFoundError(f"Resume file not found: {resume_path}")

            console.print(f"[dim]Resuming adventure from: {resume_path}[/dim]")
            seed, state, messages = load_adventure_from_file(resume_path)
            set_world_state(state)

            adventure_log = AdventureLog(
                metadata={
                    "model": args.model,
                    "profile": str(args.profile),
                    "resumed_from": str(resume_path),
                    "timestamp_start": datetime.now(timezone.utc).isoformat(),
                },
                adventure_seed={
                    "title": seed.title,
                    "theme": seed.theme,
                    "scenario": seed.scenario,
                    "npcs": seed.npcs,
                    "objects": seed.objects,
                    "mission_objective": seed.mission_objective,
                    "opening_narration": seed.opening_narration,
                },
                messages=messages,
            )
        else:
            # Generate new adventure
            console.print(
                f"[dim]Generating adventure with theme: {args.theme}...[/dim]"
            )
            seed = seed_adventure_from_wiki(client, args.model, args.theme)
            console.print(f"[dim]Adventure: {seed.title}[/dim]\n")

            adventure_log = AdventureLog(
                metadata={
                    "model": args.model,
                    "profile": str(args.profile),
                    "timestamp_start": datetime.now(timezone.utc).isoformat(),
                },
                adventure_seed={
                    "title": seed.title,
                    "theme": seed.theme,
                    "scenario": seed.scenario,
                    "npcs": seed.npcs,
                    "objects": seed.objects,
                    "mission_objective": seed.mission_objective,
                    "opening_narration": seed.opening_narration,
                },
            )

        # Run adventure
        run_adventure(
            client,
            args.model,
            profile_content,
            character_name,
            seed,
            adventure_log,
        )

        # Save adventure log
        adventure_log.metadata["timestamp_end"] = datetime.now(timezone.utc).isoformat()
        adventure_log.metadata["total_turns"] = get_world_state().turn_number
        adventure_log.metadata["total_messages"] = len(adventure_log.messages)

        output_path = adventure_log.save(logs_dir)

        save_panel = Panel(
            f"[bold green]Adventure log saved to:[/bold green]\n"
            f"{output_path}\n\n"
            f"[dim]Total turns: {adventure_log.metadata['total_turns']}\n"
            f"Total messages: {adventure_log.metadata['total_messages']}[/dim]",
            title="Saved",
            border_style="green",
        )
        console.print(f"\n{save_panel}")

    except FileNotFoundError as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        exit(1)
    except KeyboardInterrupt:
        console.print("\n[dim]Adventure ended.[/dim]")
    except Exception as e:
        log.error(f"Unexpected error: {e}")
        console.print(f"[bold red]Unexpected error:[/bold red] {e}")
        exit(1)


if __name__ == "__main__":
    main()
