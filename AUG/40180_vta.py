import numpy as np
import matplotlib.pyplot as plt
from vtaData import vtaData

vta = vtaData(shot=40180, edition=0)
print(vta.available_objects())

# Classic errorbars (default)
fig, ax = vta.plot_profiles(t=3.0, clean=True)
plt.show(block=False)

# # Shaded band with defaults
# fig, ax = vta.plot_profiles(t=4.0, error_style='fill', clean=True)
# plt.show()

# Shaded band, custom styling
fig, ax = vta.plot_profiles(t=3.0, error_style='fill',
                             color='steelblue', alpha=0.2, fmt='o-', clean=True)
plt.show()

# # Clean + shaded
# fig, ax = vta.plot_profiles(t=4.0, clean=True, error_style='fill', alpha=0.15)
# plt.show()
