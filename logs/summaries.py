import re

input_file = "decisions.log" 
output_file = "episode_summaries.log"

# Match:
# [ep=2540 ...]
ep_pattern = re.compile(r"\[ep=\s*(\d+)")

current_episode = None

with open(input_file, "r", encoding="utf-8") as fin, \
     open(output_file, "w", encoding="utf-8") as fout:

    for line in fin:
        # Keep track of the most recent episode number
        m = ep_pattern.search(line)
        if m:
            current_episode = m.group(1)

        # Extract episode-end lines
        if "EPISODE END" in line:
            cleaned = line.strip()
            cleaned = cleaned.replace("EPISODE END", f"EPISODE {current_episode}")
            fout.write(cleaned + "\n")

print(f"Saved results to {output_file}")