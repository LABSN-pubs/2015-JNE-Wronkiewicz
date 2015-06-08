#dpylint: disable-msg=C0103
"""
Plot Figure: Absolute performance of pooled vs. traditional
@author wronk
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
#from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


def pooledBCI_plotAbsPerformance(meanDataDict, snrs, trial_counts,
                                 class_methods, titles, nSubj, nLabels):

    mpl.rcParams['pdf.fonttype'] = 42

    figSize = (13., 9)
    #figWidth = 1. / nPlots
    #figSpace = .2
    cmap = mpl.cm.hot
    boxIndicatorColor = 'Lime'
    ftsize = 16
    vmin, vmax = 50., 100.
    figLabels = ['(a)', '(b)', '(c)']

    # offsets for small boxes containing performance percentage
    offsetX = np.linspace(1. / len(snrs) - .005, 1., len(snrs)) - .015
    offsetY = np.linspace(1 - 1. / len(trial_counts), 0,
                          len(trial_counts)) + .025
    bbox = dict(boxstyle='round,pad=0.15, rounding_size=.25', ec='none',
                fc='1.', alpha=.9)

    # offset for comparison plot and colorbar
    lowerAxOffset = .125

    fig, (stdAx, pooledAx) = plt.subplots(ncols=2, figsize=figSize)
    fig.subplots_adjust(left=.075, right=.925, bottom=.20, wspace=0.65)

    pooledResults = np.array([meanDataDict['accuracy'][key][-1, :]
                             for key in ['unweighted', 'centroid', 'kde']])
    #######################################
    # Generate the standard classifier plot (on left side of figure)
    stdAx.imshow(meanDataDict['accuracy'][class_methods[-1]],
                 interpolation='nearest', cmap=cmap,
                 vmin=vmin, vmax=vmax)

    for x in range(len(snrs)):
        for y in range(len(trial_counts)):
            stdAx.text(offsetX[x], offsetY[y],
                       '%.1f%%' % meanDataDict['accuracy']['std'][y][x],
                       bbox=bbox, transform=stdAx.transAxes, ha='right',
                       va='bottom', fontsize=ftsize - 2)

    stdAx.set_title('Subject-Specific Classifier', fontsize=ftsize + 4)
    [sp.set_color('none') for sp in stdAx.spines.itervalues()]
    stdAx.xaxis.set_ticks_position('bottom')
    stdAx.set_xticks(range(len(snrs)))
    stdAx.set_xticklabels([''])

    stdAx.xaxis.set_tick_params(tick1On=False, tick2On=False)
    stdAx.yaxis.set_tick_params(tick1On=False, tick2On=False)

    stdAx.set_ylabel('# Training Trials\nfrom Subject N', fontsize=ftsize + 2,
                     va='bottom')
    stdAx.set_yticklabels(trial_counts, fontsize=ftsize + 2)
    stdAx.set_yticks(range(len(trial_counts)))

    stdAx.annotate(figLabels[0], xy=(1., 1.), xycoords='axes fraction',
                   xytext=(15, 0), textcoords='offset points',
                   size=ftsize + 6, va='top')
    #######################################
    # Generate the copied gaussian mixture (lower left)
    position = [stdAx.get_position().x0, stdAx.get_position().y0 - lowerAxOffset,
                stdAx.get_position().width,
                stdAx.get_position().height / float(len(trial_counts))]
    comparisonAx = fig.add_axes(position)#, transform=stdAx.get_transform)
    comparisonAx.imshow(pooledResults[-1, :].reshape(1, -1),
                        interpolation='nearest', cmap=cmap, vmin=vmin,
                        vmax=vmax, aspect=1.)

    [sp.set_color('none') for sp in comparisonAx.spines.itervalues()]
    comparisonAx.xaxis.set_tick_params(tick1On=False, tick2On=False)
    comparisonAx.yaxis.set_tick_params(tick1On=False, tick2On=False)

    comparisonAx.set_xlabel('SNR (dB)', fontsize=ftsize + 2)
    comparisonAx.set_xticklabels(snrs, fontsize=ftsize)
    comparisonAx.set_xticks(range(len(snrs)))

    comparisonAx.set_yticks([0])
    comparisonAx.set_yticklabels([' 0'], fontsize=ftsize + 4)
    comparisonAx.set_ylabel('# Training Trials\nfrom Subject N',
                            fontsize=ftsize + 2, va='bottom')

    for x in range(len(snrs)):
        comparisonAx.text(offsetX[x], offsetY[y] * len(trial_counts),
                          '%.1f%%' % pooledResults[-1][x], bbox=bbox,
                          transform=comparisonAx.transAxes, ha='right',
                          va='bottom', fontsize=ftsize - 2)

    greenRectComp = mpl.patches.Rectangle((0.005, 0.005), .99, .99,
                                          ec=boxIndicatorColor, fc='none',
                                          transform=comparisonAx.transAxes,
                                          zorder=1, linewidth=5)
    comparisonAx.add_patch(greenRectComp)

    comparisonAx.annotate(figLabels[2], xy=(1., 1.), xycoords='axes fraction',
                          xytext=(15, 0), textcoords='offset points',
                          size=ftsize + 6, va='top')
    #######################################
    # Generate the pooled classifier plot (on right side of figure)

    imgPool = pooledAx.imshow(pooledResults, interpolation='nearest', cmap=cmap,
                              vmin=vmin, vmax=vmax)
    pooledAx.set_title('Transfer Learning Classifier', fontsize=ftsize + 4)
    pooledAx.set_xlabel('SNR (dB)', fontsize=ftsize + 2)
    pooledAx.set_xticklabels(snrs, fontsize=ftsize)
    pooledAx.set_xticks(range(len(snrs)))

    [sp.set_color('none') for sp in pooledAx.spines.itervalues()]
    pooledAx.xaxis.set_tick_params(tick1On=False, tick2On=False)
    pooledAx.yaxis.set_tick_params(tick1On=False, tick2On=False)

    pooledAx.set_ylabel('Classifier Weighting Scheme\n40*(N-1) Training Trials',
                        fontsize=ftsize + 2)
    pooledAx.set_yticks(range(len(pooledResults)))
    pooledAx.set_yticklabels(['Unweighted', 'Centroid', 'Gaussian\nMixture'],
                             rotation=45, fontsize=ftsize)

    offsetYPooled = np.linspace(1 - 1. / len(pooledResults), 0,
                                len(pooledResults)) + .025
    for x in range(len(snrs)):
        for y in range(len(pooledResults)):
            pooledAx.text(offsetX[x], offsetYPooled[y],
                          '%.1f%%' % pooledResults[y][x], bbox=bbox,
                          transform=pooledAx.transAxes, ha='right',
                          va='bottom', fontsize=ftsize - 2)

    greenRectPool = mpl.patches.Rectangle((0.005, 0.005), .99, 1 / 3. - .005,
                                          ec=boxIndicatorColor, fc='none',
                                          transform=pooledAx.transAxes,
                                          zorder=1, linewidth=5)
    pooledAx.add_patch(greenRectPool)

    pooledAx.annotate(figLabels[1], xy=(1., 1.), xycoords='axes fraction',
                      xytext=(15, 0), textcoords='offset points',
                      size=ftsize + 6, va='top')
    #######################################
    # Arrow and Colorbar

    xyStart = (pooledAx.get_position().x0, pooledAx.get_position().y0 + .115)
    xyEnd = (1.01, .5)
    comparisonAx.annotate('', xy=xyEnd, xycoords='axes fraction',
                          xytext=xyStart, textcoords='figure fraction',
                          size=35, arrowprops=dict(arrowstyle='fancy',
                                                   connectionstyle='arc3,rad=-.25',
                                                   ec='k', fc=boxIndicatorColor,
                                                   alpha=.9))

    cax = fig.add_axes([pooledAx.get_position().x0,
                        pooledAx.get_position().y0 - lowerAxOffset / 1.5,
                        pooledAx.get_position().width, .05])
    cbar = fig.colorbar(imgPool, cax, orientation='horizontal')
    cbar.set_label('% Correct', va='top', fontsize=ftsize + 2)
    cbar.set_ticks([50, 60, 70, 80, 90, 100])

    return fig
