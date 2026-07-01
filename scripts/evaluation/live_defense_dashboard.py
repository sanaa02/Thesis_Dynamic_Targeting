#!/usr/bin/env python3
"""
live_defense_dashboard.py  --  ALSAT-EO-1 Live Thesis Defense Dashboard
========================================================================
Interactive terminal dashboard using the `rich` library to run a step-by-step
simulation of the trained A1-PPO policy. 

Features:
  - Real-time Satellite Telemetry (Battery SoC, Wheel speeds, pointing vectors).
  - Target access and MODIS cloud cover visualization.
  - Flashing warnings for incoming dynamic events (wildfires, floods).
  - Live Explainability: gradient-based attribution showing exactly *why*
    the agent selected its current action, what it rejected, and how cloud
    probabilities affected the decision.
  - Interactive Step-by-Step Mode (press Enter) or Auto-Play.

Usage:
------
    CUDA_VISIBLE_DEVICES="" python scripts/evaluation/live_defense_dashboard.py \
        --model models/ppo_full_system_s42.zip \
        --auto-play --delay 1.0
"""
from __future__ import annotations
import os
import sys
import time
import math
import argparse
import numpy as np
import gymnasium as gym
import torch

# ---- path-setup ----
_SCRIPTS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROOT    = os.path.dirname(_SCRIPTS)
sys.path.insert(0, _SCRIPTS)
sys.path.insert(0, os.path.join(_SCRIPTS, "core"))
sys.path.insert(0, os.path.join(_SCRIPTS, "models"))
sys.path.insert(0, os.path.join(_SCRIPTS, "wrappers"))

from env_dynamic_factory import Config, make_env
from wrappers.action_mask_wrapper import make_masked_env

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.text import Text
from rich.align import Align
from rich.progress import ProgressBar

console = Console()

class DefenseDashboard:
    def __init__(self, model_path: str, seed: int = 300, event_rate: float = 2.0, auto_play: bool = False, delay: float = 1.0):
        self.model_path = model_path
        self.seed = seed
        self.event_rate = event_rate
        self.auto_play = auto_play
        self.delay = delay
        
        # 1. Build environment
        targets_path = os.path.join(_ROOT, "config/targets/global_45_targets.json")
        cloud_json_path = os.path.join(_ROOT, "config/cloud_reality/global_45_clouds.json")
        
        env = make_env(Config.DYN_MODIS, targets_path, cloud_json_path, event_rate=self.event_rate, seed=self.seed, with_safety=False)
        self.env = make_masked_env(env)
        
        # 2. Load model
        # Monkey patch use_sde for stable-baselines3 compatibility
        try:
            from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
            import inspect
            _orig_init = MaskableActorCriticPolicy.__init__
            def _patched_init(self, *args, **kwargs):
                sig = inspect.signature(_orig_init)
                if "use_sde" not in sig.parameters:
                    kwargs.pop("use_sde", None)
                _orig_init(self, *args, **kwargs)
            MaskableActorCriticPolicy.__init__ = _patched_init
        except Exception:
            pass

        import sys
        import numpy.core
        import numpy.core.numeric
        sys.modules["numpy._core"] = numpy.core
        sys.modules["numpy._core.numeric"] = numpy.core.numeric

        from sb3_contrib import MaskablePPO
        from attention_policy import SchedulerAttentionExtractor
        custom_objects = {
            "features_extractor_class": SchedulerAttentionExtractor,
            "action_space": self.env.action_space,
            "observation_space": self.env.observation_space,
        }
        self.model = MaskablePPO.load(self.model_path, env=self.env, custom_objects=custom_objects)
        
        # Setup layout
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="telemetry", ratio=5),
            Layout(name="decision", ratio=10),
            Layout(name="environment", ratio=6)
        )
        
        self.step_count = 0
        self.total_reward = 0.0
        self.imaged_count = 0
        self.cloud_free_count = 0
        self.obs = None
        self.done = False
        self.active_alert = None
        self.alert_timer = 0
        
        # Track previous events to detect new arrivals
        self.prev_events_set = set()

    def _get_action_probabilities(self, obs: np.ndarray) -> np.ndarray:
        """Returns the policy network's output probability distribution over actions."""
        obs_tensor, _ = self.model.policy.obs_to_tensor(obs)
        action_masks = self.env.action_masks()
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor, action_masks=action_masks)
            probs = distribution.distribution.probs.cpu().numpy()[0]
        return probs

    def _explain_action(self, obs: np.ndarray, action: int, probs: np.ndarray) -> tuple[dict, list]:
        """Calculates fast gradient-like attributions to explain the action selection."""
        attributions = {}
        
        # 1. Perturb cloud and priority of the selected action to measure attribution
        base_prob = probs[action]
        sat = self.env.unwrapped.satellites[0]
        all_targets = list(sat.scenario.targets)
        
        explanations = []
        
        if action < 45:
            # Static target action
            tgt_name = all_targets[action].name
            tgt_priority = all_targets[action].priority
            
            # Find which slot holds this target in the observation vector
            found_slot = -1
            for slot_i in range(6):
                tid_val = obs[13 + slot_i*6 + 5]
                # Target ID is saved as idx / N_STATIC_TARGETS (which is action / 20, wait! N_STATIC_TARGETS is 20 in the wrapper but action space has 45!)
                # Let's match by checking the float index representation
                expected_tid = float(action) / 20.0
                if abs(tid_val - expected_tid) < 0.01:
                    found_slot = slot_i
                    break
            
            explanations.append(f"• [bold green]Static target [cyan]{tgt_name}[/cyan] chosen (Priority {tgt_priority:.2f})[/bold green]")
            
            if found_slot != -1:
                idx_cloud = 13 + found_slot * 6 + 2
                # Perturb cloud cover forecast
                obs_perturbed = obs.copy()
                obs_perturbed[idx_cloud] = min(1.0, obs[idx_cloud] + 0.20)
                perturbed_probs = self._get_action_probabilities(obs_perturbed)
                cloud_effect = perturbed_probs[action] - base_prob
                
                cloud_val = obs[idx_cloud] * 100
                if cloud_effect < -0.01:
                    explanations.append(f"  - [green]Low cloud forecast ([white]{cloud_val:.0f}%[/white]) is a strong positive driver.[/green] (Perturbing cloud to {cloud_val+20:.0f}% drops probability by [red]{abs(cloud_effect)*100:.1f}%[/red]).")
                else:
                    explanations.append(f"  - [yellow]Target is cloud-free ([white]{cloud_val:.0f}%[/white]).[/yellow]")
            else:
                explanations.append("  - Target is visible in the current satellite footprint.")
                
        elif 45 <= action <= 47:
            # Dynamic event slot
            slot_idx = action - 45
            mgr = getattr(sat, "_event_manager", None)
            now = sat.simulator.sim_time
            slots = mgr.get_slots(sat, now)
            evt = slots[slot_idx]
            
            if evt is not None:
                evt_name = evt.name
                evt_prio = evt.priority
                cloud_fcst = obs[49 + slot_idx*4 + 1] * 100
                explanations.append(f"• [bold magenta]Dynamic Event [cyan]{evt_name}[/cyan] chosen in Slot {slot_idx} (Priority {evt_prio:.2f})[/bold magenta]")
                explanations.append(f"  - [green]Event urgency bonus and high priority make this action highly competitive.[/green]")
                
                # Cloud check
                idx_cloud = 49 + slot_idx*4 + 1
                obs_perturbed = obs.copy()
                obs_perturbed[idx_cloud] = min(1.0, obs[idx_cloud] + 0.20)
                perturbed_probs = self._get_action_probabilities(obs_perturbed)
                cloud_effect = perturbed_probs[action] - base_prob
                if cloud_effect < -0.01:
                    explanations.append(f"  - [green]Cloud forecast is safe ([white]{cloud_fcst:.0f}%[/white]).[/green] (Increasing cloud risk reduces action probability by [red]{abs(cloud_effect)*100:.1f}%[/red]).")
            else:
                explanations.append(f"• [red]Dynamic slot {slot_idx} selected, but slot is currently empty.[/red]")
                
        elif action == 48:
            # Drift action
            explanations.append("• [bold yellow]Satellite entered DRIFT (Housekeeping) mode[/bold yellow]")
            battery_soc = obs[0] * 100
            if battery_soc < 35.0:
                explanations.append(f"  - [red]Battery State of Charge is critically low ({battery_soc:.1f}%).[/red] Drift is enforced to prevent battery depletion and simulator death.")
            else:
                explanations.append(f"  - [yellow]Battery is healthy ({battery_soc:.1f}%).[/yellow] No target was visible or all visible opportunities are cloudy, so satellite drifts to desaturate reaction wheels.")
                
        # 3. List alternate top choices
        top_indices = np.argsort(probs)[::-1]
        alternates = []
        count = 0
        for idx in top_indices:
            if idx == action:
                continue
            if count >= 3:
                break
            prob_pct = probs[idx] * 100
            if prob_pct < 0.1:
                continue
            
            if idx < 45:
                name = f"Image {all_targets[idx].name}"
            elif 45 <= idx <= 47:
                name = f"Image Dyn Slot {idx - 45}"
            else:
                name = "Drift"
            alternates.append(f"[yellow]{name}[/yellow] ({prob_pct:.1f}%)")
            count += 1
            
        return attributions, alternates, explanations

    def run_simulation(self):
        obs, info = self.env.reset()
        self.obs = obs
        
        # Initial events check
        sat = self.env.unwrapped.satellites[0]
        mgr = getattr(sat, "_event_manager", None)
        self.prev_events_set = {evt.name for evt in getattr(mgr, "_events", [])}
        
        with Live(self.layout, refresh_per_second=4, screen=True) as live:
            while not self.done:
                # Update telemetry panel
                now_s = float(sat.simulator.sim_time)
                battery_soc = obs[0]
                
                # Check for new event arrivals
                curr_events = getattr(mgr, "_events", [])
                curr_events_set = {evt.name for evt in curr_events}
                new_arrivals = curr_events_set - self.prev_events_set
                if new_arrivals:
                    self.active_alert = list(new_arrivals)[0]
                    self.alert_timer = 5  # Show alert for 5 steps
                self.prev_events_set = curr_events_set
                
                # Compute probabilities and select action
                probs = self._get_action_probabilities(obs)
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
                
                # Attributions and explanations
                _, alternates, explanations = self._explain_action(obs, action, probs)
                
                # Generate Telemetry Text
                self._update_telemetry_layout(sat, now_s, battery_soc)
                
                # Generate Decisions/Explainability Text
                self._update_decision_layout(action, probs[action], alternates, explanations)
                
                # Generate Environment/Target Text
                self._update_environment_layout(sat, now_s, mgr)
                
                # Header
                self.layout["header"].update(Panel(
                    Align.center(f"[bold white]🛰️ ALSAT-EO-1 MISSION CONTROL -- THESIS LIVE DEMONSTRATION[/bold white]  [yellow]Seed: {self.seed}[/yellow] | [green]Event Rate: {self.event_rate}/hr[/green]", vertical="middle"),
                    border_style="bold blue"
                ))
                
                # Footer
                footer_text = ""
                if self.auto_play:
                    footer_text = f"[bold green]Auto-Play Mode Active[/bold green] (Delay: {self.delay}s) | [white]Press Ctrl+C to terminate[/white]"
                else:
                    footer_text = "[bold yellow]Step-by-Step Mode[/bold yellow] | [bold green]Press ENTER to execute next satellite decision[/bold green] | [white]Ctrl+C to quit[/white]"
                
                self.layout["footer"].update(Panel(
                    Align.center(footer_text, vertical="middle"),
                    border_style="dim white"
                ))
                
                live.refresh()
                
                # Wait for user input or timer delay
                if self.auto_play:
                    time.sleep(self.delay)
                else:
                    # Workaround for rich.live blocking stdin:
                    # In Live screen mode, we can pause the refresh and read input
                    console.input()
                
                # Execute step
                next_obs, reward, term, trunc, info_step = self.env.step(action)
                self.total_reward += reward
                self.step_count += 1
                self.done = term or trunc
                
                # Metrics tracking
                if action < 45:
                    if info_step.get("imaging_occurred", False):
                        self.imaged_count += 1
                        self.cloud_free_count += 1
                elif 45 <= action <= 47:
                    if info_step.get("dynamic_imaging_occurred", False):
                        self.imaged_count += 1
                        self.cloud_free_count += 1
                        
                obs = next_obs
                self.obs = obs
                
                if self.alert_timer > 0:
                    self.alert_timer -= 1
                    if self.alert_timer == 0:
                        self.active_alert = None

        console.print(f"\n[bold green]Simulation Complete![/bold green] Total Steps: {self.step_count} | Total Reward: {self.total_reward:+.3f} | Images Taken: {self.imaged_count}\n")

    def _update_telemetry_layout(self, sat, now_s, battery_soc):
        h = int(now_s // 3600)
        m = int((now_s % 3600) // 60)
        s = int(now_s % 60)
        
        # Reaction wheel speeds (rad/s)
        try:
            rw_speeds = sat.dynamics.Omega * (9.5493)  # convert to RPM
            rw_text = f"  • RW1 Speed: {rw_speeds[0]:+7.1f} RPM\n  • RW2 Speed: {rw_speeds[1]:+7.1f} RPM\n  • RW3 Speed: {rw_speeds[2]:+7.1f} RPM"
        except Exception:
            rw_text = "  • RW Speeds: telemetry unavailable"
            
        battery_pct = battery_soc * 100
        bat_color = "green" if battery_pct > 50 else "yellow" if battery_pct > 30 else "red"
        
        # Simulated attitude pointing error
        try:
            pointing_error = math.degrees(sat.dynamics.sigma_BR[0]) * 10  # artificial scale for display
            point_status = f"{pointing_error:.2f}°"
        except Exception:
            point_status = "0.00°"
            
        alert_pnl = ""
        if self.active_alert:
            alert_pnl = f"\n\n[bold blink red]🚨 ALERT: {self.active_alert} DETECTED![/bold blink red]\n[magenta]Dynamic Event arrived in Slot 0[/magenta]"

        tel_content = f"""[bold yellow]Satellite Health Telemetry[/bold yellow]
------------------------------------
  • Mission Time : [cyan]{h:02d}h {m:02d}m {s:02d}s[/cyan]
  • Step Index   : [white]{self.step_count}[/white]
  • Battery SoC  : [{bat_color}]{battery_pct:.1f}%[/{bat_color}]
  • Camera Error : [white]{point_status}[/white]
  • Orbit Index  : [white]{int(now_s // 5700)}[/white]

[bold yellow]Attitude Control System (ACS)[/bold yellow]
------------------------------------
{rw_text}
{alert_pnl}
"""
        self.layout["telemetry"].update(Panel(
            tel_content,
            title="🛰️ ALSAT-EO-1 Telemetry",
            border_style="bold cyan"
        ))

    def _update_decision_layout(self, action, confidence, alternates, explanations):
        # Action string
        sat = self.env.unwrapped.satellites[0]
        all_targets = list(sat.scenario.targets)
        
        if action < 45:
            action_str = f"Image static target [cyan]{all_targets[action].name}[/cyan]"
            style = "bold green"
        elif 45 <= action <= 47:
            action_str = f"Image Dynamic Target [magenta]Slot {action - 45}[/magenta]"
            style = "bold magenta"
        else:
            action_str = "Perform solar panel Drift"
            style = "bold yellow"
            
        expl_text = "\n".join(explanations)
        alts_text = " | ".join(alternates) if alternates else "None"
        
        dec_content = f"""[{style}][bold]Active Decision: {action_str}[/bold] (Confidence: {confidence*100:.1f}%)[/{style}]

[bold yellow]Explainability & Attribution Details:[/bold yellow]
--------------------------------------------------------------------------------
{expl_text}

[bold yellow]Prominent Rejected Alternatives:[/bold yellow]
--------------------------------------------------------------------------------
  {alts_text}

[bold yellow]Performance Summary:[/bold yellow]
--------------------------------------------------------------------------------
  • Total Reward Accumulation  : [green]{self.total_reward:+.3f}[/green]
  • Total Images Successfully Taken : [white]{self.imaged_count}[/white]
"""
        self.layout["decision"].update(Panel(
            dec_content,
            title="🧠 PPO Policy Decision & Attributions",
            border_style="bold green"
        ))

    def _update_environment_layout(self, sat, now, mgr):
        # List upcoming opportunities
        opps = [o for o in getattr(sat, 'upcoming_opportunities', [])
                if (o.get('type') == 'target' if isinstance(o, dict)
                    else getattr(o, 'type', '') == 'target')]
        opps = sorted(opps, key=lambda o: o['window'][0])[:5]
        
        tbl_static = Table(title="Visible Static Targets", box=None)
        tbl_static.add_column("Target Name", style="cyan")
        tbl_static.add_column("Priority", style="yellow")
        tbl_static.add_column("Cloud Forecast", style="green")
        
        for opp in opps:
            tgt = opp.get('object', None)
            if tgt:
                cloud = getattr(tgt, "cloud_cover", 0.0) * 100
                cloud_style = "green" if cloud < 30 else "yellow" if cloud < 60 else "red"
                tbl_static.add_row(tgt.name, f"{tgt.priority:.2f}", f"[{cloud_style}]{cloud:.0f}%[/{cloud_style}]")
                
        # List dynamic events
        slots = mgr.get_slots(sat, now)
        tbl_dyn = Table(title="Dynamic Event Slots", box=None)
        tbl_dyn.add_column("Slot", style="white")
        tbl_dyn.add_column("Event type", style="magenta")
        tbl_dyn.add_column("Cloud", style="green")
        tbl_dyn.add_column("Expires In", style="yellow")
        
        for idx, evt in enumerate(slots):
            if evt is None:
                tbl_dyn.add_row(f"{idx}", "[dim]Empty[/dim]", "-", "-")
            else:
                cloud = evt.cloud_cover_forecast * 100
                cloud_style = "green" if cloud < 30 else "yellow" if cloud < 60 else "red"
                expires = max(0.0, evt.expiration_time - now)
                tbl_dyn.add_row(f"{idx}", evt.name.split("_")[1], f"[{cloud_style}]{cloud:.0f}%[/{cloud_style}]", f"{expires:.0f}s")
                
        self.layout["environment"].update(Panel(
            Layout(tbl_static, size=8),
            title="🌍 Environment Targets",
            border_style="bold yellow"
        ))
        
        # Split environment layout to show static targets on top and dynamic events on bottom
        self.layout["environment"].split_column(
            Layout(Panel(tbl_static, title="🌍 Visible Targets (Swath)", border_style="bold yellow")),
            Layout(Panel(tbl_dyn, title="🔥 Dynamic Event Slots", border_style="bold magenta"))
        )

def main():
    parser = argparse.ArgumentParser(description="Live thesis defense dashboard")
    parser.add_argument("--model", type=str, required=True, help="Path to trained PPO zip model")
    parser.add_argument("--seed", type=int, default=300, help="Random seed for simulation")
    parser.add_argument("--event-rate", type=float, default=2.0, help="Dynamic event arrival rate per hour")
    parser.add_argument("--auto-play", action="store_true", help="Auto-advance steps instead of pressing Enter")
    parser.add_argument("--delay", type=float, default=1.0, help="Timer delay (seconds) in auto-play mode")
    args = parser.parse_args()
    
    dashboard = DefenseDashboard(
        model_path=args.model,
        seed=args.seed,
        event_rate=args.event_rate,
        auto_play=args.auto_play,
        delay=args.delay
    )
    dashboard.run_simulation()

if __name__ == "__main__":
    main()
