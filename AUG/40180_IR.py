# =============================================================================
# ASDEX Upgrade - IR Heat Flux Profile Analysis
# Shot #40180
# =============================================================================

# --- Standard Library ---
import sys
from copy import deepcopy as copy

# --- Custom IR Tools ---
sys.path.append('/shares/departments/AUG/users/ircd/Software/py3irtools')
import Sektor7Unten
import ir

# --- Scientific Computing ---
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences
from scipy.integrate import trapezoid as trapz

# --- Plotting ---
import matplotlib.pyplot as plt
from matplotlib import gridspec

# =============================================================================
# CONFIGURATION
# =============================================================================

SHOT_NUMBER  = 40180
EDITION      = 4
TIME_START   = 4.34
TIME_END     = TIME_START + 1.33

# Spatial index range for region of interest
IND_START    = 800    # Start index along target (pixels)
IND_END      = 1400   # End index along target (pixels)

# Strike line detection: column range to search within
SL_COL_START = 1200
SL_COL_END   = 1350
SL_THRESHOLD = 0.6    # Fraction of local max to detect strike line edge

# Plotting
S_OFFSET_EXP = 99     # Spatial offset for target location (mm)
VMIN         = 0      # Colorbar min (MW/m²)
VMAX         = 32.0   # Colorbar max (MW/m²)

# =============================================================================
# SECTION 1: LOAD & PREPROCESS IR DATA
# =============================================================================

# --- Load heat flux profiles and apply Wiener filter ---
a = Sektor7Unten.heatFluxProfiles(SHOT_NUMBER, EDITION)
a.wiener()

# --- Reverse spatial axis ---
a.data = a.data[:, ::-1]

# =============================================================================
# SECTION 2: STRIKE LINE DETECTION & CORRECTION
# =============================================================================

def get_strikeline(i):
    """
    Detect the strike line position for time step i.

    Within the column window [SL_COL_START:SL_COL_END]:
      1. Normalize the signal to [0, 1]
      2. Reverse the array (search from the right)
      3. Find the first index exceeding SL_THRESHOLD
    
    Returns
    -------
    int : Pixel index of the strike line (within the search window)
    """
    signal        = a.data[i, SL_COL_START:SL_COL_END]
    signal_norm   = signal / np.max(signal)
    signal_rev    = signal_norm[::-1]
    return np.argmax(signal_rev > SL_THRESHOLD)


# --- Detect strike line pixel index for every time step ---
strikeline = np.uint32(
    list(map(get_strikeline, range(a.time.size)))
)

# --- Convert pixel indices to physical locations ---
strikeline_position = a.location[strikeline]

# --- Build continuous interpolator: f(t) → strike line position ---
f = interp1d(a.time, strikeline_position)

# --- Correct dataset relative to moving strike line ---
a = a.get_corrected_strikeline(f)

# =============================================================================
# SECTION 3: CROP & PREPARE DATA FOR ANALYSIS
# =============================================================================

IR = copy(a)

# --- Crop spatial axis to region of interest ---
IR.location = IR.location[IND_START:IND_END]
IR.data     = IR.data[:, IND_START:IND_END]

# --- Convert heat flux from W/m² to MW/m² ---
IR.data /= 1e6

# --- Crop time axis to analysis window ---
time_mask = (a.time > TIME_START) & (a.time < TIME_END)
IR.time   = IR.time[time_mask]
IR.data   = IR.data[time_mask]

# =============================================================================
# SECTION 4: PLOT HEAT FLUX MAP
# =============================================================================

# --- Derived plot quantities ---
target_loc_mm = IR.location * 1e3 - S_OFFSET_EXP   # Physical location (mm)
x_ticks       = np.linspace(0, np.pi, 3)            # Toroidal angle ticks

# --- Figure setup ---
fig, ax = plt.subplots(figsize=(8, 6))
fig.suptitle('ASDEX Upgrade', fontsize=22, family='serif')

# --- Heat flux image ---
img = ax.imshow(
    IR.data[::1].T,
    interpolation = 'bilinear',
    aspect        = 'auto',
    vmin          = VMIN,
    vmax          = VMAX,
    extent        = [
        0, np.pi,
        target_loc_mm[0],
        target_loc_mm[-1]
    ]
)

# --- Colorbar ---
cbar = fig.colorbar(img, ax=ax)
cbar.set_label(
    r'Heat flux ($\frac{\mathrm{MW}}{\,\mathrm{m}^2}$)',
    fontsize=18, family='serif'
)
cbar.set_ticks([0, 10, 20, 30])
cbar.set_ticks([0, 10, 20, 30], minor=True)

# --- Axes formatting ---
ax.set_xlim(0, np.pi)
ax.set_xticks(x_ticks[:-1])
ax.set_xticklabels(('$0$', r'$\frac{1}{2}\pi$'), fontsize=18, family='serif')

ax.set_ylim(-2, 60)
ax.set_yticks([0, 30, 60])
ax.set_yticks([0, 10, 20, 30, 40, 50, 60], minor=True)
ax.set_ylabel('Target location (mm)', family='serif', fontsize=20)

for tick in ax.get_yticklabels():
    tick.set_fontsize(18)
    tick.set_family('serif')

# --- Axis label (toroidal angle as figure text) ---
fig.text(
    0.5, 0.05,
    'Toroidal angle',
    family='serif', fontsize=20,
    va='center', ha='center'
)

# --- Annotations ---
ax.text(np.pi / 2, 55, 'Measurement',
        ha='center', family='serif', fontsize=20, color='w')
ax.text(0.1, 47, f'# {SHOT_NUMBER}',
        family='serif', fontsize=16, color='w')

# --- Final layout ---
plt.tight_layout()
plt.subplots_adjust(left=0.16, bottom=0.18, right=0.85, top=0.9)

plt.show()