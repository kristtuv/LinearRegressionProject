import matplotlib.pyplot as plt

plt.style.use('ggplot')
params = {'legend.fontsize': 10,
         'axes.labelsize': 15,
         'axes.labelpad': 4.0,
         'axes.titlesize': 24,
         'axes.labelcolor':'black',
         'lines.linewidth': 3,
         'lines.markersize':8,
         'xtick.labelsize': 13,
         'ytick.labelsize':13,
         'xtick.major.width': 6,
         'ytick.major.width': 6}

plt.rcParams.update(params)
