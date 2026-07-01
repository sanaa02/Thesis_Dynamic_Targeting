import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Create a figure with two subplots to handle both conceptual diagrams
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1.5, 1]})

# ==========================================
# SUBPLOT 1: Figure 1.3 - Agility Concept
# ==========================================
time = np.linspace(0, 100, 500)
# Synthetic ground track (sine wave mapped to 2D)
ground_track_y = np.sin(time / 10) * 10

ax1.plot(time, ground_track_y, color='black', linestyle='--', label='Satellite Ground Track')

# Mock target locations (off-nadir)
targets = [
    {'time': 20, 'y': 15, 'name': 'Target A (Algiers)'},
    {'time': 50, 'y': -12, 'name': 'Target B (Blida)'},
    {'time': 80, 'y': 18, 'name': 'Target C (Event)'}
]

for t in targets:
    # Plot target
    ax1.plot(t['time'], t['y'], 'ro', markersize=8)
    ax1.text(t['time'], t['y'] + 2, t['name'], ha='center')
    
    # Draw slew line from track to target
    track_y_at_t = np.sin(t['time'] / 10) * 10
    ax1.annotate('', xy=(t['time'], t['y']), xytext=(t['time'], track_y_at_t),
                 arrowprops=dict(arrowstyle='->', color='blue', lw=1.5))
    
    # Draw visibility window swath (grey circle)
    circle = patches.Circle((t['time'], t['y']), radius=8, color='grey', alpha=0.2)
    ax1.add_patch(circle)

ax1.set_title('Figure 1.3: Satellite Agility Concept and Off-Nadir Slewing', fontsize=12)
ax1.set_xlabel('Time along orbit')
ax1.set_ylabel('Cross-track distance (km)')
ax1.legend(loc='lower right')
ax1.grid(True, linestyle=':', alpha=0.6)

# ==========================================
# SUBPLOT 2: Figure 3.1 - MDP vs SMDP
# ==========================================
# Main timeline axis
ax2.axhline(0, color='black', lw=2)

# Define visibility windows [start, end]
windows = [[15, 25], [45, 60], [75, 80]]

# Plot the windows as colored blocks
for i, w in enumerate(windows):
    rect = patches.Rectangle((w[0], -0.2), w[1]-w[0], 0.4, color='orange', alpha=0.6)
    ax2.add_patch(rect)
    ax2.text((w[0]+w[1])/2, 0.3, f'Window {i+1}', ha='center', color='darkorange', fontweight='bold')

# MDP (Uniform steps) - Red markers above line
mdp_steps = np.arange(5, 95, 10)
ax2.plot(mdp_steps, np.ones_like(mdp_steps) * 0.6, 'rv', markersize=8, label='MDP Steps (Uniform)')

# SMDP (Variable Sojourn) - Green markers exactly on window boundaries below line
smdp_steps = [w[0] for w in windows]
ax2.plot(smdp_steps, np.ones_like(smdp_steps) * -0.6, 'g^', markersize=8, label='SMDP Steps (Aligned)')

# Annotate Tau (Sojourn Time)
ax2.annotate('', xy=(smdp_steps[0], -0.8), xytext=(smdp_steps[1], -0.8),
             arrowprops=dict(arrowstyle='<->', color='green'))
ax2.text((smdp_steps[0]+smdp_steps[1])/2, -0.95, r'Sojourn Time $\tau_1$', ha='center', color='green')

ax2.set_xlim(0, 100)
ax2.set_ylim(-1.2, 1.2)
ax2.set_yticks([]) 
ax2.set_xlabel('Elapsed Time (seconds)')
ax2.set_title('Figure 3.1: MDP vs. SMDP Decision Timing', fontsize=12)
ax2.legend(loc='upper right')

plt.tight_layout(pad=3.0)

# Save for manuscript
# plt.savefig('figures_conceptual.pdf', dpi=300, bbox_inches='tight')
plt.show()