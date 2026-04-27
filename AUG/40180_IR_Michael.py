import sys
#################################
sys.path.append('/shares/departments/AUG/users/ircd/Software/py3irtools')
import Sektor7Unten
import ir
################################

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import gridspec

import scipy
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt, find_peaks, peak_prominences
from scipy.integrate import trapezoid as trapz

from copy import deepcopy as copy


######################################IR#######################################################
shotNumber = 40180
edition=4
timeStart = 4.34
timeEnd = timeStart+ 1.33

a = Sektor7Unten.heatFluxProfiles(shotNumber, edition)
a.wiener()

indexAll = np.arange(a.data.shape[1])
indexExclude = []
index = list(set(indexAll) - set(indexExclude))

a.data = a.data[:,::-1]

get_strikeline = lambda i: np.argmax(((a.data[i,1200:1350]/np.max(a.data[i,1200:1350]))[::-1]>0.6))

#strikeline = np.uint32(map(get_strikeline, range(a.time.size)))
strikeline = np.uint32(list(map(get_strikeline, range(a.time.size))))
strikeline_position = a.location[strikeline]
f = interp1d(a.time, strikeline_position)
a = a.get_corrected_strikeline(f)

IR = copy(a)

indStart = 800#500 #numpy.argmin(numpy.abs(a.location*1.0e3-50))
indEnd = 1400 #numpy.argmin(numpy.abs(a.location*1.0e3-110))

IR.location = IR.location[indStart:indEnd]

IR.data = IR.data[:,indStart:indEnd]

IR.data /=1e6

idx = (a.time>timeStart) & (a.time<timeEnd)
IR.time = IR.time[idx]
IR.data = IR.data[idx]

sOffsetExp   = 99


##########################################
x = np.linspace(0,np.pi,3)

figure = plt.figure()
figure.suptitle('ASDEX Upgrade', fontsize=22, family='serif')
plot1 = figure.add_subplot(111)
ax1 = plot1.imshow(IR.data[::1].T, interpolation='bilinear', aspect='auto', vmin=0, vmax=32.0, extent=[0, np.pi,IR.location[0]*1.0e3-sOffsetExp,IR.location[-1]*1.0e3-sOffsetExp])
cb = figure.colorbar(ax1)
cb.set_label(r' Heat flux ($\frac{\mathrm{MW}}{\,\mathrm{m}^2}$)', fontsize = 18, family='serif')
plot1.set_xticks(x[:-1])
plot1.set_xticklabels(('$0$',r'$\frac{1}{2} \pi$'), fontsize=18, family='serif')
plot1.set_ylim(-2,60)
plot1.set_yticks([0,10,20,30,40,50,60], minor=True)
plot1.set_yticks([0,30,60])

cb.set_ticks([0,10, 20, 30], minor=True)
cb.set_ticks([0,10,20,30])

plot1.set_ylabel('Target location (mm)', family='serif', fontsize=20)
figure.text(0.5, 0.05, 'Toroidal angle', family='serif', fontsize=20, va='center', ha = 'center')
plot1.text(np.pi/2, 55, 'Measurement', ha='center', family='serif', fontsize=20, color='w')
plot1.text(0.1, 47, '# 40180', family='serif', fontsize=16, color='w')
plt.tight_layout()
plt.subplots_adjust(0.16,0.18,0.85,0.9)
for tick in plot1.get_yticklabels():
    tick.set_fontsize(18)
    tick.set_family('serif')

plt.show()
