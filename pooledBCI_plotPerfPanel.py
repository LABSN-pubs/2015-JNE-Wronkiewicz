'''
pooledBCI_plotPerfPanel.py
Generate and Plot BCI relevant labels
@Author: wronk
'''

#dpylint: disable-msg=C0103

import os
from surfer import Brain
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.ticker import MultipleLocator
plt.close()
plt.ion()
import warnings

#Only show warnings once
warnings.simplefilter('once')

subjectDir = os.environ['SUBJECTS_DIR']
modelSubj = 'fsaverage'
hemi = 'lh'


def pooledBCI_plotPerfPanel(labelList, poolSizes, labelPerfs, stdPerfs_y,
                            stddev, colors, legend=[],
                            axRange=np.arange(65., 90., 5.),
                            views=['l', 'm'], surface='white'):
    """ Plot performance as a function of subject pool size for a given set of
    labels along with their visualization on the brain's surface
    """

    ftSize_title = 28
    ftSize_axesLabel = ftSize_title - 2
    ftSize_axesTicks = ftSize_title - 8
    figSize = (15, 9)

    fig = plt.figure(figsize=figSize)
    fig.patch.set_fc('white')

    # Make two subplots, first is 1/3 of area, second is 2/3
    ax1 = plt.subplot2grid((1, 3), (0, 0))
    ax2 = plt.subplot2grid((1, 3), (0, 1), colspan=2)

###########################################################################
### Pooled performance gain axis

    # Add all the performance curves and stddev lines to the plot
    for i, label in enumerate(labelList):
        ax2.fill_between(poolSizes, labelPerfs[i, :] + stddev[i],
                         labelPerfs[i, :] - stddev[i], facecolor=colors[i],
                         linewidth=0, alpha=0.15)
        ax2.plot(poolSizes, labelPerfs[i, :], color=colors[i],
                 linewidth=2.5, zorder=2)

    # Add text box indicating that chance is 50 %
    # Light gray bounding bbox
    bbox = dict(boxstyle='round,pad=0.35, rounding_size=.25', ec='none',
                fc='.9', alpha=.9)
    ax2.text((poolSizes[-2] + poolSizes[-3]) / 2, min(axRange) + 1.25,
             'Chance = 50%', color='Black', bbox=bbox, transform=ax2.transData,
             ha='left', va='center', fontsize=ftSize_title - 6)

    # Turn on the grid
    ax2.grid(which='both', zorder=0, linewidth=0.5, linestyle=':',
             color='grey')

    #Titles/labels for second axis
    ax2.set_title('Effect of Pool Size on Accuracy',
                  fontsize=ftSize_title)
    ax2.set_xlabel('Training Pool Size (# Subjects)',
                   fontsize=ftSize_axesLabel)
    ax2.set_ylabel('Accuracy (%)', fontsize=ftSize_axesLabel)

    # Add dashed lines from subject specific (n=1) to pooled training (n=2)
    interp = [[stdPerfs_y[i], labelPerfs[i, 0]]
              for i in np.arange(len(labelPerfs))]
    for r in range(len(stdPerfs_y)):
        ax2.plot([1, poolSizes[0]], interp[r], c=colors[r], zorder=3,
                 linewidth=1.5, linestyle='--', clip_on=False)

    # Add markers for subject specific training (n=1)
    for t in range(len(stdPerfs_y)):
        ax2.plot(1, stdPerfs_y[t], marker='o', c=colors[t],
                 markersize=12, fillstyle='full', zorder=3, clip_on=False)

    # Give y-axis tick labels more padding (avoid overlapping with markers)
    for tick in ax2.get_yaxis().get_major_ticks():
        tick.set_pad(8.)
        tick.label1 = tick._get_text1()

    ax2.set_xticks(np.insert(poolSizes, 0, 1))
    ax2.set_yticks(axRange)
    ax2.tick_params(axis='both', labelsize=ftSize_axesTicks)

    ax2.yaxis.set_minor_locator(MultipleLocator(2.5))

    plt.xlim(1, poolSizes[-1])
    plt.ylim(axRange[0], axRange[-1])
###########################################################################
### Brain visualization axis

    #Turn off ticks/splines
    ax1.set_axis_off()

    brain = Brain(modelSubj, hemi=hemi, surf=surface,
                  config_opts=dict(background='white'))
    brain.set_data_smoothing_steps(10)

    # Add labels
    for i, label in enumerate(labelList):
        brain.add_label(label, color=colors[i], alpha=0.825, hemi=label.hemi)
        #brain.add_label(label, color='black', alpha=0.9, borders=1, hemi=label.hemi)

    # Save montage as an image
    montage = brain.save_montage(None, order=views, orientation='v',
                                 border_size=15, colorbar=None)

    # Add the montage to the first axis
    ax1.imshow(montage, interpolation='nearest', origin='upper')
    brain.close()

    # Titles/labels for first axis
    # Make custom transform so that the first axis title aligns to the first
    trans = transforms.blended_transform_factory(ax1.transAxes, ax2.transAxes)
    title1 = ax1.set_title('Regions of Interest', fontsize=ftSize_title,
                           transform=trans)
    title1.set_y(1.015)

    # Add legend
    legends = []
    for s in np.arange(len(legend)):
        legends.append(ax1.annotate(legend[s][0], xy=(0, - 0.01 - .065 * len(legends)),
                                    xycoords='axes fraction', ha='left',
                                    va='top', fontsize=ftSize_axesLabel,
                                    color=legend[s][1], zorder=0))

    # Adjust layout
    fig.subplots_adjust(left=0.025, right=.975, bottom=0.1, top=0.9, wspace=.3)

    return fig
