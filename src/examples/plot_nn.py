import glob
from pylab import *
import brewer2mpl

 # brewer2mpl.get_map args: set name  set type  number of colors
bmap = brewer2mpl.get_map('Set2', 'qualitative', 7)
colors = bmap.mpl_colors

params = {
    'axes.labelsize': 8,
    'font.size': 8,
    'legend.fontsize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'text.usetex': False,
    'figure.figsize': [9, 4.5]
}
rcParams.update(params)


nn = np.loadtxt('nn.dat')
data = np.loadtxt('data.dat')

actual = []

for i in nn:
    actual.append(math.cos(i[0]))

fig = figure() # no frame
ax = fig.add_subplot(111)

ax.fill_between(nn[:,0], nn[:,1] - nn[:,2],  nn[:,1] + nn[:,2], alpha=0.25, linewidth=0, color=colors[0])
ax.plot(nn[:,0], nn[:,1], linewidth=2, color=colors[0])
ax.plot(nn[:,0], actual, linewidth=2, linestyle='--', color=colors[3])
ax.plot(data[:,0], data[:, 1], 'o', color=colors[2])

legend = ax.legend(["NN", 'cos(x)', 'Data'], loc=8)
frame = legend.get_frame()
frame.set_facecolor('1.0')
frame.set_edgecolor('1.0')

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()
ax.tick_params(axis='x', direction='out')
ax.tick_params(axis='y', length=0)
for spine in ax.spines.values():
  spine.set_position(('outward', 5))
ax.set_axisbelow(True)
ax.grid(axis='y', color="0.9", linestyle='-', linewidth=1)

fig.savefig('nn.png')