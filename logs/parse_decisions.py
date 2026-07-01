import re

def parse_logs():
    input_path = "decisions.log"
    stages = []
    current_stage = None
    
    # Patterns
    run_start_pattern = re.compile(r"RUN START.*?stage=(\S+).*?total_steps=([\d,]+)")
    step_pattern = re.compile(r"\[ep=\s*(\d+)\s+step=\s*(\d+).*?\]\s+(\w+)\s+(.*?)\s+r=([+-]?\d+(?:\.\d+)?)\s+batt=([\d.]+)%")
    ep_end_pattern = re.compile(r"↳ EPISODE END.*?reward=([+-]?\d+(?:\.\d+)?)")

    stage_data = []
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            # Stage transition
            m_start = run_start_pattern.search(line)
            if m_start:
                stage_name = m_start.group(1)
                total_steps = int(m_start.group(2).replace(",", ""))
                current_stage = {
                    "name": stage_name,
                    "config_steps": total_steps,
                    "episodes": [],
                    "total_steps_count": 0,
                    "static_success": 0,
                    "static_cloudy": 0, # failed static due to clouds
                    "dynamic_success": 0,
                    "dynamic_failed": 0,
                    "drift": 0,
                }
                stage_data.append(current_stage)
                continue
                
            if current_stage is None:
                continue
                
            # Step actions
            m_step = step_pattern.search(line)
            if m_step:
                current_stage["total_steps_count"] += 1
                action_type = m_step.group(3) # static, dynamic, drift
                action_details = m_step.group(4)
                reward = float(m_step.group(5))
                batt = float(m_step.group(6))
                
                if action_type == "static":
                    # static   target=Boumerdes (idx=1)   or   static   target=Paris (idx=30)
                    # Let's check reward or text to classify cloudy vs successful
                    if "cloudy" in action_details or reward < 0:
                        current_stage["static_cloudy"] += 1
                    else:
                        current_stage["static_success"] += 1
                elif action_type == "dynamic":
                    if "event=" in action_details:
                        current_stage["dynamic_success"] += 1
                    else:
                        current_stage["dynamic_failed"] += 1
                elif action_type == "drift":
                    current_stage["drift"] += 1
                continue
                
            # Episode summaries
            m_ep = ep_end_pattern.search(line)
            if m_ep:
                reward = float(m_ep.group(1))
                current_stage["episodes"].append(reward)
                
    # Print report
    print("=========================================================================")
    print(" ALSAT-EO-1 Training Run Convergence Analysis (500k Curriculum Run)")
    print("=========================================================================")
    for idx, stage in enumerate(stage_data):
        name = stage["name"]
        steps = stage["total_steps_count"]
        n_eps = len(stage["episodes"])
        avg_reward = sum(stage["episodes"]) / max(1, n_eps)
        
        static_tot = stage["static_success"] + stage["static_cloudy"]
        dyn_tot = stage["dynamic_success"] + stage["dynamic_failed"]
        
        print(f"\nStage {idx}: {name}")
        print(f"  Config Steps       : {stage['config_steps']:,}")
        print(f"  Actual Steps Logged: {steps:,}")
        print(f"  Total Episodes     : {n_eps}")
        print(f"  Average Reward     : {avg_reward:+.3f}")
        print(f"  Actions Breakdown:")
        print(f"    - Static Target Observations:")
        print(f"       * Succeeded (Clear): {stage['static_success']} ({(stage['static_success']/max(1, static_tot))*100:.1f}%)")
        print(f"       * Failed (Cloudy)  : {stage['static_cloudy']} ({(stage['static_cloudy']/max(1, static_tot))*100:.1f}%)")
        print(f"       * Total Attempts   : {static_tot}")
        if name.startswith("dynamic"):
            print(f"    - Dynamic Target Observations:")
            print(f"       * Succeeded (Imaged): {stage['dynamic_success']} ({(stage['dynamic_success']/max(1, dyn_tot))*100:.1f}%)")
            print(f"       * Failed (No Image) : {stage['dynamic_failed']} ({(stage['dynamic_failed']/max(1, dyn_tot))*100:.1f}%)")
            print(f"       * Total Attempts   : {dyn_tot}")
        print(f"    - Drift (Idle/Recharge): {stage['drift']} ({stage['drift']/max(1, steps)*100:.1f}% of steps)")
    print("=========================================================================")

if __name__ == "__main__":
    parse_logs()
