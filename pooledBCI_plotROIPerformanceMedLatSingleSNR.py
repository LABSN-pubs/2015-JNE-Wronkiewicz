#dpylint: disable-msg=C0103
from surfer import Brain
import numpy as np
import matplotlib.pyplot as plt
from mne.viz import mne_analyze_colormap
from mpl_toolkits.axes_grid1 import ImageGrid


def pooledBCI_plotROIPerformanceMedLatSingleSNR(deltaAccuracy,
                                                brainPlotTrialInds, snrs,
                                                snrIndToUse, trial_count,
                                                labelList, fs_srcs,
                                                brainVizSize, flim, cm):
    '''
    Plot Figure: ROI performance
    '''

    #surface = 'smoothwm'
    surface = 'inflated_pre'
    views = ['l', 'm']

    brainPlots = []

    cm = mne_analyze_colormap(flim)
    cm_mpl = mne_analyze_colormap(flim, format='matplotlib')

    for trialRow in brainPlotTrialInds:
        brainViz = Brain('fsaverage', 'lh', surface,
                         config_opts=dict(background='white',
                                          width=brainVizSize[0],
                                          height=brainVizSize[1]))
        vtx_data = np.zeros((fs_srcs[0]['np']))

        for labelInd in range(len(labelList)):
            vtx_data[labelList[labelInd].vertices] = \
                deltaAccuracy[trialRow, labelInd, snrIndToUse]

        brainViz.add_data(vtx_data, -1 * flim[-1], flim[-1], colormap=cm)
        brainPlots.append(brainViz.save_montage(None, order=views,
                                                orientation='h',
                                                border_size=10, colorbar=None))
        brainViz.close()
    #######################################
    # plot all individual brains in a single grid figure
    figSpace = .4
    ftsize = 16
    figSize = [12, 5]

    fig = plt.figure(figsize=figSize)
    fig.patch.set_fc('white')

    gridBrains = ImageGrid(fig, 111, nrows_ncols=(1, 1), axes_pad=figSpace,
                           cbar_mode='single', cbar_location='right',
                           cbar_size='7%', cbar_pad=figSpace, share_all=True,
                           label_mode="L", direction='row', add_all=True)

    fig.subplots_adjust(left=0.05, right=0.90, bottom=0.05, top=0.70)

    for i in range(len(brainPlots)):
        img = gridBrains[i].imshow(brainPlots[i], interpolation='nearest',
                                   cmap=cm_mpl, vmin=-1 * flim[-1],
                                   vmax=flim[-1])

        gridBrains[i].spines['left'].set_color('none')
        gridBrains[i].spines['top'].set_color('none')
        gridBrains[i].spines['right'].set_color('none')
        gridBrains[i].spines['bottom'].set_color('none')

    # Get rid of any tics, unneccessary
    gridBrains.axes_llc.set_xticks([])
    gridBrains.axes_llc.set_xticklabels('')
    gridBrains.axes_llc.set_yticks([])
    gridBrains.axes_llc.set_yticklabels('')

    # Format colorbar
    cbar = gridBrains.cbar_axes[0].colorbar(img)
    cbar.ax.tick_params(labelsize=ftsize - 2)
    cbar.set_clim((-1 * flim[-1], flim[-1]))
    cbar.set_label_text('% Classification Change', rotation=270, va='bottom',
                        fontsize=ftsize + 2)

    plt.suptitle('Classification Differences for Individual ROIs\n' +
                 '[Gaussian Mixture] - [Traditional Classifier]' +
                 '\n' + str(trial_count) + ' Trials, SNR: ' +
                 str(snrs[snrIndToUse]),
                 fontsize=ftsize + 12)
    fig.set_size_inches(figSize)

    return fig
