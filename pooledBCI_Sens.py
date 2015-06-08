#dpylint: disable-msg=C0103
'''
Generate simulated EEG data using real subject BEM and brain models. Conduct
transfer learning using simulated data.

@Author: wronk
'''

import sys
import mne

from mne.simulation.source import generate_stc
from copy import deepcopy
from time import time, strftime
from os import environ, path as op
import numpy as np
from sklearn.lda import LDA
from sklearn.svm import SVC
from scipy.spatial import distance_matrix
from scipy.io import loadmat, savemat
from sklearn import preprocessing
from sklearn.grid_search import GridSearchCV
from ldaReg import ldaRegWeights as ldaReg
import surfer
import cPickle
import warnings

#Only show warnings once
warnings.simplefilter('once')
mne.set_log_level(False)


class FakeEvoked():
    """Make evoked-like class"""
    def __init__(self, data, info, tmin=0.0, sfreq=1000.0):
        self._data = data
        self.info = deepcopy(info)
        self.info['sfreq'] = sfreq
        self.times = np.arange(data.shape[-1]) / sfreq + tmin
        self._current = 0
        self.ch_names = info['ch_names']


class FakeCov(dict):
    def __init__(self, data, info, diag=False):
        self.data = data
        self['data'] = data
        self['bads'] = info['bads']
        self['names'] = info['ch_names']
        self.ch_names = info['ch_names']
        self['eig'] = None
        self['eig_vec'] = None
        self['diag'] = diag


subjectDir = op.join(environ['CODE_ROOT'], 'AnatomBCI_Mark')
structDir = op.join(environ['SUBJECTS_DIR'])
saveDir = op.join(environ['CODE_ROOT'], 'AnatomBCI_Mark',
                  'AnatomBCI_Figures_Python')

#######################################
subjectSet = []
subjectSet.append(['RON006_AKCLEE', 'RON007_AKCLEE', 'RON008_AKCLEE',
                   'RON010_AKCLEE', 'RON011_AKCLEE', 'RON014_AKCLEE',
                   'RON016_AKCLEE', 'RON021_AKCLEE'])

subjectSet.append(['AKCLEE_101', 'AKCLEE_103', 'AKCLEE_104', 'AKCLEE_105',
                   'AKCLEE_106', 'AKCLEE_107', 'AKCLEE_109', 'AKCLEE_110',
                   'AKCLEE_113', 'AKCLEE_114', 'AKCLEE_115', 'AKCLEE_118',
                   'AKCLEE_119', 'AKCLEE_120', 'AKCLEE_121', 'AKCLEE_122',
                   'AKCLEE_123', 'AKCLEE_124', 'AKCLEE_125', 'AKCLEE_126',
                   'AKCLEE_127'])


subjSetNum = 1
subjects = subjectSet[subjSetNum]
subjects = subjects[0:22]
n_max_ch = 74
#######################################
# Load a few custom labels that aren't in the parcellation or aren't G/S
labels_all = []
nonParcLabels = ['G_precentral_handMotor_radius_15mm-lh',
                 'G_precentral_handMotor_radius_10mm-lh',
                 'G_precentral_handMotor_radius_5mm-lh',
                 'Pole_occipital-lh']
#######################################
#reinitialize forwards, inverse, covariance
doAnalysis = True
saveAnalysis = False
savePlots = True
re_init = True
saveIndividualInfo = True
saveConvBank = False
loadConvBank = True
clfSchemes = ['LDA', 'LDA_Reg', 'SVM']
clfScheme = clfSchemes[1]  # classifier schemes
plotsToGen = [0, 1, 2]  # indicates which figures to show
brainPlotTrialInds = [2]  # for brain plots, which trial inds to print
plotFormat = ['png', 'pdf']

class_methods = ['kde', 'centroid', 'unweighted', 'std']  # Weighting schemes

#get label names. The '[0]' index keeps only the label and not the label color
labelList = mne.read_labels_from_annot(subject='fsaverage',
                                       parc='aparc.a2009s')

labelList = [elem for elem in labelList if ((elem.name[0] == 'G' or elem.name[0] == 'S') or
             elem.name == 'Pole_occipital-lh') and elem.hemi == 'lh' and
             'Jensen' not in elem.name]

# Load customized labels into label list
for label_name in nonParcLabels:
    if 'handMotor_radius' in label_name:
        label_fname = op.join(structDir, 'fsaverage', 'label', label_name + '.label')
        labelList.append(mne.read_label(label_fname, subject='fsaverage'))
    if 'UDStd' in label_name or 'LRStd' in label_name:
        label_fname = op.join(structDir, 'fsaverage', 'label', label_name + '.label')
        labelList.insert(0, (mne.read_label(label_fname, subject='fsaverage')))

n_smooth = 5
lambda2 = 1. / 9.  # MNE regularization param
regFactors = [0.05]  # LDA reglarization to optimize benchmark
n_jobs = 6  # Processors
#######################################
# higher magnitude = faster rolloff with increasing distance
expFactor_centroid = -30
expFactor_KDE = -30
#######################################
# Activity simulation params
tstep = 1e-3
snrs = range(-15, 0, 5)
trial_counts = [10, 20, 40]
max_trials = max(trial_counts)
current_mag = 1.
repeats = 25  # Number of times to repeat analysis
C_range = 10.0 ** np.arange(-6, -4)  # SVM params if using SVM
gamma_range = 10.0 ** np.arange(-2, 3)
levelRatio = np.zeros((len(subjects), len(subjects)))
#######################################

# For debugging conditions, so make simulation fast
if len(subjects) < 10:
    repeats = 1
    snrs = range(-15, 0, 5)
    trial_counts = [10, 20, 40]
    brainPlotTrialInds = [2]
    #lenLabelSetToRun = 1
    #label_inds = np.random.randint(0, len(labelList), (lenLabelSetToRun))
    #labelList = [labelList[i] for i in label_inds]

    # for frontal_middle label
    labelList = [l for l in labelList if 'G_precentral-lh' in l.name]

labelNames = [label.name for label in labelList]
nLabels = len(labelList)
print labelList

# Load fsaverage information
fs_vertices = [np.arange(10242), np.arange(10242)]
n_src_fs = sum([len(i) for i in fs_vertices])
fs_srcs = mne.read_source_spaces(op.join(structDir, 'fsaverage', 'bem',
                                         'fsaverage-5-src.fif'))

# File names to save for faster processing in future
fileBanks = ['fwd_bank', 'fwdmat_bank', 'invMat_bank', 'noiseCov_bank'
             'conv_bank', 'vdist_bank']
if(subjSetNum == 0):
    cache_fname = op.join(subjectDir, 'RON__cache')
else:
    cache_fname = op.join(subjectDir, 'AKCLEE__cache')

subjTxt_fname = op.join(cache_fname, 'included_subjects.txt')
subjBank_fname = op.join(cache_fname, 'banks.pkl')

# Initialize lists
vertNum_bank = []
vertPos_bank = []
fwd_bank = []
fwdMat_bank = []
invMat_bank = []
conv_bank = []
convSrc_bank = []
#conv_bank = np.empty((len(subjects), len(subjects)), dtype=object)
label_bank = []
vdist_bank = []
noiseCov_bank = []
sphereSurf_bank = []
fakeEvoked_bank = []
fwdColorers = []
sensorNoise = []

start_time = time()
###############################################################################
### Initializing subject data ###
if doAnalysis:
    # Initialize score matrix
    scoreDict = {'logRatio': {}, 'accuracy': {}}
    meanDataDict = deepcopy(scoreDict)
    labelAccuracy = {'accuracy': {}}

    for method in class_methods:
        scoreDict['logRatio'] = {method: [] for method in class_methods}
        scoreDict['accuracy'][method] = []
        labelAccuracy['accuracy'][method] = {method: [] for method in
                                             class_methods}

    if re_init:
        print '!!! COMMENCE SIMULATION !!! (@ ' + strftime('%H:%M:%S') + ')'
        print 'nSubjs:\t\t' + str(len(subjects))
        print 'Trials:\t\t' + str(trial_counts)
        print 'SNRs:\t\t' + str(snrs)
        print 'nLabels:\t' + str(nLabels)
        print 'nRepeats:\t' + str(repeats)
        print 'exp_cent: \t' + str(expFactor_centroid)
        print 'exp_gaus: \t' + str(expFactor_KDE) + '\n'

        print 'Processing fwd, inv, noise cov, etc:'
        for si, subj in enumerate(subjects):

            print '  ' + subj,
            sys.stdout.flush()
            # Load/generate forwards
            if(subjSetNum == 0):
                fwd_fname = op.join(subjectDir, subj, subj + '-2-fwd.fif')
                cov_fname = op.join(subjectDir, subj, subj + '-noise-cov.fif')
                inv_fname = op.join(subjectDir, subj, subj + '_eeg-1-inv.fif')
            else:
                fwd_fname = op.join(subjectDir, subj, subj + '-7-fwd-eeg.fif')
                cov_fname = op.join(subjectDir, subj, subj + '-noise-cov-eeg.fif')
                inv_fname = op.join(subjectDir, subj, subj + '-inv-eeg-python.fif')

            src_fname = op.join(structDir, subj, 'bem', subj + '-7-src.fif')

            # Load forward solution
            fwd = mne.read_forward_solution(fwd_fname, force_fixed=True,
                                            surf_ori=False)
            fwd = mne.pick_types_forward(fwd, meg=False, eeg=True,
                                         ref_meg=False, exclude='bads')

            fwd_bank.append(fwd)
            info = deepcopy(fwd['info'])
            info['projs'] = []
            vertices = [s['vertno'] for s in fwd['src']]
            n_src = sum([len(v) for v in vertices])

            # Load and store spherical surface coordinate points
            surf_fname = op.join(structDir, subj, 'surf/')
            sphereSurf_bank.append([mne.read_surface(surf_fname + 'lh.sphere')[0],
                                    mne.read_surface(surf_fname + 'rh.sphere')[0]])
            # Load covariance and inverse
            cov = mne.read_cov(cov_fname)
            inv = mne.minimum_norm.read_inverse_operator(inv_fname)

            # Generate forward matrix
            d = np.zeros((len(info['ch_names']), 1))
            fake_evoked = FakeEvoked(d, info)
            d = np.eye(n_src)
            stc = mne.SourceEstimate(data=d, vertices=vertices, tmin=0, tstep=1)
            evoked = mne.simulation.generate_evoked(fwd, stc, fake_evoked, cov,
                                                    snr=np.inf)
            fwdMat_bank.append(evoked.data)

            # Generate inverse matrix
            evoked.data = np.eye(len(info['ch_names']))
            invApplied = mne.minimum_norm.apply_inverse(evoked=evoked,
                                                        inverse_operator=inv,
                                                        lambda2=lambda2,
                                                        method='MNE').data
            invMat_bank.append(invApplied)
            fakeEvoked_bank.append(evoked)

            # Generate noise covariances
            noiseCov_bank.append(FakeCov(np.cov(fwd['sol']['data']),
                                         deepcopy(fwd['info'])))

            # Load labels from parcellation
            label_bank.append(mne.labels_from_parc(subj, parc='aparc.a2009s'))
            label_bank[si][1][:] = []  # Clear out ROI color info

            #Load custom labels too and add them onto the end
            for label_name in nonParcLabels:
                if 'handMotor_radius' in label_name:
                    label_fname = op.join(structDir, subj, 'label', label_name + '.label')
                    label_bank[si][0].append(mne.read_label(label_fname, subject=subj))
                # Check if using LRStd or UDStd
                if 'UDStd' in label_name or 'LRStd' in label_name:
                    label_fname = op.join(structDir, subj, 'label', label_name + '.label')
                    tempLabel = mne.read_label(label_fname, subject=subj)
                    tempHemi = ([0, 1], [1, 0])[tempLabel.hemi == 'rh']
                    label_bank[si][tempHemi[0]].append(tempLabel)

            # Generate distances between vertices
            vert_coord = [fwd['src'][0]['rr'][vertices[0]],
                          fwd['src'][1]['rr'][vertices[1]]]
            vertPos_bank.append(vert_coord)

            # Compute distance between every source point and normalize
            # Euclidean method
            euclidean = False  # Toggle for method
            if euclidean:
                temp_dists = [distance_matrix(hemiVerts, hemiVerts)
                              for hemiVerts in vert_coord]
                vdist_bank.append([hemi / hemi.max() for hemi in temp_dists])
            # Cortical distance method
            else:
                src = mne.read_source_spaces(fname=src_fname)
                temp_dists = [src[hemi]['dist'][vertices[hemi]][:, vertices[hemi]].A
                              for hemi in range(len(src))]
                vdist_bank.append([hemi / hemi.max() for hemi in temp_dists])

            # Save vertices, channel names
            vertNum_bank.append(vertices)

            print '... ' + 'Done (' + str(si + 1) + '/' + str(len(subjects)) + ')'

        if(saveIndividualInfo):
            '''
            savemat(op.join(cache_fname, 'bankInfo.mat'),
                    {'fwd_bank': fwd_bank, 'fwdMat_bank': fwdMat_bank,
                    'invMat_bank': invMat_bank, 'noiseCov_bank': noiseCov_bank,
                    'vdist_bank': vdist_bank, 'fakeEvoked_bank': fakeEvoked_bank,
                    'label_bank': label_bank})
                    '''
    else:
        # Load all the subject info banks (instead of calculating them)
        print 'Loading fwd, inv, noise cov, etc:'

    '''
        if not op.exists(subjTxt_fname) or not op.exists(subjBank_fname):
            raise Exception('Missing bank file(s).')

        with open(subjTxt_fname, 'r') as f_txt:
            expected_subjects = [line.rstrip('\n') for line in f_txt]
        f_txt.close()
        if expected_subjects != subjects:
            raise Exception('Cached subject bank does not match subjects being analyzed.')
        with open(subjBank_fname, 'r') as f:
            [fwd_bank, fwdMat_bank, invMat_bank, noiseCov_bank,
            conv_bank, vdist_bank, fakeEvoked_bank, label_bank] = cPickle.load(f)
        f.close()
        print' Banks Loaded'
    '''

    # Compute or load conversion matrices
    if(loadConvBank):
        print 'Loading Conversion Matrices.'
        conv_dict = loadmat(op.join(cache_fname, 'convBank_python.mat'))
        conv_bank = conv_dict['convBank_python']
        conv_bank = conv_bank[:len(subjects), :len(subjects)]

        convSrc_bank = conv_dict['convBankSrc_python']
        convSrc_bank = convSrc_bank[:len(subjects), :len(subjects)]
    else:
        print 'Computing Conversion Matrices:'
        for sFrom, subjFrom in enumerate(subjects):
            tempConv = []
            tempConvSrc = []
            #noiseIn = np.reshape(sensorNoise[sTo], (fwdMat_bank[sTo].shape[0], -1))
            #covIn = np.cov(noiseIn)
            covIn = noiseCov_bank[sFrom]['data']
            levelIn = np.mean(np.sqrt(np.diag(covIn)))
            for sTo, subjTo in enumerate(subjects):
                if subjFrom != subjTo:
                    convMat = mne.compute_morph_matrix(subjFrom, subjTo,
                                                       vertices_from=vertNum_bank[sFrom],
                                                       vertices_to=vertNum_bank[sTo],
                                                       smooth=n_smooth)

                    # Forward * Src Conversion * Inverse
                    fullConvMat = np.dot(fwdMat_bank[sTo], convMat.A).dot(
                        invMat_bank[sFrom])

                    # Normalize cortical amplitude ratios
                    transCov = fullConvMat.dot(covIn.dot(fullConvMat.T))
                    levelTrans = np.mean(np.sqrt(np.diag(transCov)))
                    levelRatio[sFrom, sTo] = levelIn / levelTrans

                    tempConv.append(levelRatio[sFrom, sTo] * fullConvMat)
                    tempConvSrc.append(convMat)

                    print '  ' + subjFrom + ' to ' + subjTo + ' ... Done',
                    print '  [' + str(sFrom) + ']' + '[' + str(sTo) + '] ' + str(fullConvMat.shape)
                else:
                    tempConv.append([])
                    tempConvSrc.append([])

            conv_bank.append(tempConv)
            convSrc_bank.append(tempConvSrc)
        #####################################
        if(saveConvBank):
            print 'Saving Conversion Matrices'
            savemat(op.join(cache_fname, 'convBank_python.mat'),
                    {'convBank_python': conv_bank,
                     'convBankSrc_python': convSrc_bank})
            '''
            # Save all the subject info banks
            if not op.exists(cache_fname):
                makedirs(cache_fname)

            with open(subjTxt_fname, 'w') as f_txt:
                f_txt.writelines([subj + '\n' for subj in subjects])
            f_txt.close()

            #Open file and dump data
            #for bank_fname in file_banks:
            with open(subjBank_fname, 'w') as f:
                cPickle.dump([fwd_bank, fwdMat_bank, invMat_bank, noiseCov_bank,
                            conv_bank, vdist_bank, fakeEvoked_bank, label_bank], f)
            f.close()
            print 'Banks Saved\n'
            '''

###############################################################################
# Activity Simulation
    rng = np.random.RandomState()
    databank = np.zeros((len(labelList), len(snrs), len(trial_counts),
                        len(subjects), 2 * max(trial_counts),
                        fwd_bank[0]['nchan']))

    print 'Simulating/Classifying Data'

### Iteration guide
# trials - number of training trials for the classifier
        # Labels - each label in the parcellation
            #SNRs - several signal to noise ratios
                #repeats- number of times to repeat the classification task
                    #subj - make each subj the subject of interest one time

    for ti, n_trials in enumerate(trial_counts):
        print '  ' + str(n_trials) + ' Trial Group [',
        sys.stdout.flush()

        current = np.ones((1, n_trials)) * current_mag

        #  nLabels x nSNRs x nRepeats x nSubjs x off and on classification
        std_logRatioBlock = np.zeros((len(labelList), len(snrs),
                                      repeats, len(subjects), 2 * n_trials))
        std_accuracyBlock = np.zeros((len(labelList), len(snrs),
                                      repeats, len(subjects)))

        unweighted_logRatioBlock = np.zeros((len(labelList), len(snrs),
                                            repeats, len(subjects),
                                             2 * n_trials))
        unweighted_accuracyBlock = np.zeros((len(labelList), len(snrs),
                                            repeats, len(subjects)))
        C_optimum = np.zeros((len(labelList), len(snrs), repeats,
                              len(subjects)))
        g_optimum = np.zeros((len(labelList), len(snrs), repeats,
                              len(subjects)))

        centroid_logRatioBlock = np.zeros((len(labelList), len(snrs), repeats,
                                           len(subjects), 2 * n_trials))
        centroid_accuracyBlock = np.zeros((len(labelList), len(snrs),
                                           repeats, len(subjects)))
        kde_logRatioBlock = np.zeros((len(labelList), len(snrs),
                                      repeats, len(subjects), 2 * n_trials))
        kde_accuracyBlock = np.zeros((len(labelList), len(snrs),
                                      repeats, len(subjects)))
        for li, label in enumerate(labelList):
            for snri, snr in enumerate(snrs):
                for ri in range(repeats):
                    trialBlock = []
                    powerMeas = []
                    for si, subj in enumerate(subjects):
                        # Generate evoked data (sensor space)
                        evoked_template = fakeEvoked_bank[si]
                        #######################################################
                        # Generate and store evoked data for one subject
                        h = ([0, 1], [1, 0])[label.hemi == 'rh']
                        labelInd = [l.name for l in
                                    label_bank[si][h[0]]].index(label.name)
                        stc = generate_stc(src=fwd_bank[si]['src'],
                                           labels=[label_bank[si][0][labelInd]],
                                           stc_data=current, tmin=0, tstep=tstep)
                        evoked = mne.simulation.generate_evoked(fwd_bank[si], stc, evoked_template,
                                                                noiseCov_bank[si], snr=snr, random_state=rng)
                        # generate evoked data noise by subtracting pure signal
                        # from evoked data
                        evoked_sig = mne.simulation.generate_evoked(fwd_bank[si], stc, evoked_template,
                                                                    noiseCov_bank[si], snr=np.inf, random_state=rng)
                        trialBlock.append(np.array([(evoked.data - evoked_sig.data).T,
                                                    evoked.data.T]))

                        #######################################################
                        ### Standard Leave-one-out classifier
                        # Loop through all trials and do std leave-one-trial-out
                        # training for one subject

                        trialBlockAccuracy = []
                        trialBlockC = []
                        trialBlockLogRatio = np.zeros(n_trials * 2)
                        for testInd in np.arange(n_trials):
                            trainInds = np.delete(np.arange(n_trials), testInd)
                            train_std = (np.r_[trialBlock[-1][0][trainInds, :],
                                               trialBlock[-1][1][trainInds, :]])
                            test_std = (np.r_[trialBlock[-1][0][testInd, :].reshape(1, -1),
                                              trialBlock[-1][1][testInd, :].reshape(1, -1)])

                            y_train_std = np.r_[np.zeros(len(trainInds), dtype=np.int8),
                                                np.ones(len(trainInds), dtype=np.int8)]
                            y_test_std = np.array([0, 1])

                            if(clfScheme == clfSchemes[0]):
                                # Train and test LDA algorithm
                                clf_std = LDA()
                                clf_std.fit(train_std, y_train_std,
                                            store_covariance=False)
                                trialBlockAccuracy.append(clf_std.score(test_std, y_test_std))
                            elif(clfScheme == clfSchemes[1]):
                                # Train and test Regularized LDA algorithm
                                weights = ldaReg(train_std, y_train_std, regFactors)[:, :, 0]
                                test_std_aug = np.c_[np.ones((len(y_test_std), 1)), test_std]
                                LDAOutput = test_std_aug.dot(weights)
                                pred_std = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                                trialBlockAccuracy.append(np.mean(pred_std == y_test_std))
                                '''
                                # Train and test Regularized LDA algorithm
                                weights_all = ldaReg(train_std, y_train_std, regFactors)
                                test_std_aug = np.c_[np.ones((len(y_test_std), 1)), test_std]
                                lambAccuracy = []
                                for lamb, i in enumerate(regFactors):
                                    LDAOutput = test_std_aug.dot(weights_all[:, :, i])
                                    pred_std = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                                    lambAccuracy.append(np.mean(pred_std == y_test_std))
                                trialBlockAccuracy.append(lambAccuracy)
                                '''

                            elif(clfScheme == clfSchemes[2]):
                                # Train and test Support Vector Machine algorithm
                                clf_std = SVC(cache_size=2048)
                                '''
                                ###############################################
                                # Grid Search
                                param_grid = [{'kernel': ['linear'], 'C': C_range}]
                                #cv = StratifiedKFold(y=y_train_std, n_folds=3)
                                gridSearch = GridSearchCV(clf_std, param_grid=param_grid,
                                                    pre_dispatch=n_jobs)
                                gridSearch.fit(train_std, y_train_std)
                                #gridSearch.fit(train_std, y_train_std)
                                trialBlockAccuracy.append(gridSearch.score(test_std, y_test_std))

                                #print('Best Classifier is: ', gridSearch.best_estimator_)
                                trialBlockC.append(gridSearch.best_estimator_.C)
                                #g_optimum[li, snri, ri, soi] = gridSearch.best_estimator_.gamma
                                '''
                                ###############################################
                                # Set parameter estimation
                                #clf_std.set_params(kernel='rbf', C=1000, gamma=5e-5
                                clf_std.set_params(kernel='linear', C=1e-4)
                                clf_std.fit(train_std, y_train_std)
                                trialBlockAccuracy.append(clf_std.score(test_std, y_test_std))
                            '''
                            y_pred_log_probs = lda.predict_log_proba(test_std)
                            trialBlockLogRatio[(testInd * 2):(testInd * 2 + 2)] =\
                                np.reshape(y_pred_log_probs[:, 0] - y_pred_log_probs[:, 1],
                                        (-1,))
                            '''

                        '''
                        C_optimum[li, snri, ri, si] = \
                            np.mean(trialBlockC)
                        '''
                        std_accuracyBlock[li, snri, ri, si] = \
                            np.mean(trialBlockAccuracy)
                        std_logRatioBlock[li, snri, ri, si] = \
                            trialBlockLogRatio = np.array(trialBlockLogRatio)
                    ###########################################################
                    # Morph all data between all subjects
                    morphedData = []
                    for soi in range(len(subjects)):
                        # Get index subjects whose data will be morphed
                        otherSubjs = np.delete(range(len(subjects)), soi)
                        morphedData1Subj = np.empty((len(otherSubjs), 2,
                                                     n_trials,
                                                     len(fwdMat_bank[soi])))

                        # Morph data from all subjects to subj of interest
                        # (fwd * (conv * (inv * data)))
                        for ind, sj in enumerate(otherSubjs):
                            #make matrix to convert sensor data
                            #tempConverter = conv_bank[sj][soi]

                            morphedData1Subj[ind, 0, :, :] = \
                                conv_bank[sj][soi].dot(trialBlock[sj][0, :, :].T).T
                                #np.dot(tempConverter, trialBlock[sj][1, :, :].T).T

                            morphedData1Subj[ind, 1, :, :] = \
                                conv_bank[sj][soi].dot(trialBlock[sj][1, :, :].T).T
                                #np.dot(tempConverter, trialBlock[sj][1, :, :].T).T

                        morphedData.append(morphedData1Subj)

                    ###########################################################
                    # BEGIN POOLED TRAINING
                    for soi in range(len(subjects)):

                        # Training data comes solely from other subjects
                        train_0 = np.reshape(morphedData[soi][:, 0, :, :],
                                             (-1, fwdMat_bank[soi].shape[0]))
                        train_1 = np.reshape(morphedData[soi][:, 1, :, :],
                                             (-1, fwdMat_bank[soi].shape[0]))
                        train_pool = np.r_[train_0, train_1]
                        y_train_pool = np.r_[np.zeros(len(train_0), dtype=np.int8),
                                             np.ones(len(train_1), dtype=np.int8)]

                        # Test data comes from subject of interest
                        test_0 = trialBlock[soi][0, :, :]
                        test_1 = trialBlock[soi][1, :, :]
                        test_pool = np.r_[test_0, test_1]
                        y_test_pool = np.r_[np.zeros(len(test_0), dtype=np.int8),
                                            np.ones(len(test_1), dtype=np.int8)]
                        #######################################################
                        ### Unweighted Classifier
                        # For each subject, train on all other subjects and then test
                        # on subject of interest
                        if(clfScheme == clfSchemes[0]):
                            # Train and test LDA algorithm
                            clf_unwt = LDA()
                            clf_unwt.fit(train_pool, y_train_pool,
                                         store_covariance=False)
                            unweighted_accuracyBlock[li, snri, ri, soi] = \
                                clf_unwt.score(test_pool, y_test_pool)
                        elif(clfScheme == clfSchemes[1]):
                            # Train and test Regularized LDA algorithm
                            weights = ldaReg(train_pool, y_train_pool, regFactors)[:, :, 0]
                            test_pool_unwt = np.c_[np.ones((len(y_test_pool), 1)), test_pool]
                            LDAOutput = test_pool_unwt.dot(weights)
                            pred_unwt = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                            unweighted_accuracyBlock[li, snri, ri, soi] = \
                                np.mean(pred_unwt == y_test_pool)
                            '''

                            # Train and test Regularized LDA algorithm
                            weights_all = ldaReg(train_pool, y_train_pool, regFactors)
                            test_pool_unwt = np.c_[np.ones((len(y_test_pool), 1)), test_pool]
                            for i in range(len(regFactors)):
                                LDAOutput = test_pool_unwt.dot(weights_all[:, :, i])
                                pred_unwt = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                                unweighted_accuracyBlock[li, snri, ri, soi, i] = \
                                        np.mean(pred_unwt == y_test_pool)
                            '''
                        elif(clfScheme == clfSchemes[2]):
                            # Train and test Support Vector Machine algorithm
                            # Scale inputs to [-1 1] as SVM is scale sensitive
                            scaler_unwt = preprocessing.data.StandardScaler().fit(train_pool)
                            train_pool_unwt = scaler_unwt.transform(train_pool)
                            test_pool_unwt = scaler_unwt.transform(test_pool)

                            clf_unwt = SVC(cache_size=2048)
                            '''
                            ###################################################
                            # Grid Search
                            param_grid = [{'kernel': ['linear'], 'C': C_range}]
                            #cv = StratifiedKFold(y=y_train_pool, n_folds=3)
                            gridSearch = GridSearchCV(clf_unwt, param_grid=param_grid,
                                                pre_dispatch=n_jobs)
                            gridSearch.fit(train_pool_unwt, y_train_pool)
                            #gridSearch.fit(train_pool, y_train_pool)
                            unweighted_accuracyBlock[li, snri, ri, soi] = \
                                gridSearch.score(test_pool_unwt, y_test_pool)

                            #print('Best Classifier is: ', gridSearch.best_estimator_)
                            C_optimum[li, snri, ri, soi] = gridSearch.best_estimator_.C
                            #g_optimum[li, snri, ri, soi] = gridSearch.best_estimator_.gamma
                            '''
                            ###################################################
                            # Set parameter estimation
                            #clf_unwt.set_params(kernel='rbf', C=1000, gamma=5e-5
                            clf_unwt.set_params(kernel='linear', C=1e-1)
                            clf_unwt.fit(train_pool_unwt, y_train_pool)
                            unweighted_accuracyBlock[li, snri, ri, soi] = \
                                clf_unwt.score(test_pool_unwt, y_test_pool)

                        '''
                        y_pred_log_probs = lda.predict_log_proba(test_pool_scaled)
                        unweighted_logRatioBlock[li, snri, ri, soi, :] = \
                            (y_pred_log_probs[:, 0] - y_pred_log_probs[:, 1])
                        '''

                        #######################################################
                        ### Centroid weighting classifier

                        # Find label center for weighting
                        h = ([0, 1], [1, 0])[label.hemi == 'rh']
                        labelInd = [l.name for l in label_bank[soi][h[0]]].index(label.name)

                        centerSurf = surfer.Surface(subjects[soi], ['lh', 'rh'][h[0]], 'sphere')
                        centerSurf.load_geometry()

                        # Get label mean position for soi and find vertex closest to center
                        labelVerts = label_bank[soi][h[0]][labelInd].vertices
                        labelAvgPos = np.mean(sphereSurf_bank[soi][h[0]][labelVerts], axis=0)

                        #labelAvgPos = np.mean(label_bank[soi][h[0]][labelInd].pos, axis=0)

                        centerVtx = surfer.utils.find_closest_vertices(
                            centerSurf.coords[vertNum_bank[soi][h[0]]], labelAvgPos)

                        # Pull pre-computed distances from the center vertex
                        dists = vdist_bank[soi][h[0]][centerVtx[0]]

                        # Make sure we calculate for both hemispheres
                        if(h[0] == 0):
                            dists = np.r_[dists, np.ones(len(vdist_bank[soi][h[1]])) * np.max(dists)]
                        else:
                            dists = np.r_[np.ones(len(vdist_bank[soi][h[1]])) * np.max(dists), dists]
                        # Compute centroid weighting based on exp(dist)
                        centroidSrc_weights = np.exp(expFactor_centroid * dists ** 2)
                        centroid_weights = fwdMat_bank[soi].dot(centroidSrc_weights)
                        centroid_weights = np.abs(centroid_weights) / np.max(np.abs(centroid_weights))

                        #######################################################
                        '''
                        convertedLabelInds = np.zeros((len(invMat_bank[soi]), len(otherSubjs)))
                        otherSubjs = np.delete(range(len(subjects)), soi)

                        labelCenterVerts = np.zeros(len(otherSubjs))
                        for ind, otherSubj in enumerate(otherSubjs):
                            h = [[0, 1], [1, 0]][label.hemi == 'rh']
                            hemi = ['lh', 'rh'][h[0]]

                            #tempLabel = label_bank[otherSubj][hemi][labelInd]
                            #tempLabel.values.fill(1.0)
                            #morphedLabel = tempLabel.morph(subject_from=subjects[otherSubj],
                            #    subject_to=subjects[soi], n_jobs=n_jobs, copy=True, smooth=1)

                            unmorphedCenterVert = np.mean(label_bank[otherSubj][h[0]][labelInd].pos, axis=0)
                            tempCenterSurf = surfer.Surface(subjects[otherSubj], hemi, 'sphere')
                            tempCenterSurf.load_geometry()
                            labelCenterVerts[ind] = surfer.utils.find_closest_vertices(centerSurf.coords,
                                    unmorphedCenterVert.reshape(-1, 3))

                            tempCoordLabel = surfer.utils.coord_to_label(subject_id=subjects[otherSubj],
                                                                         coord=labelCenterVerts[ind], label='center_point',
                                                                         hemi=hemi, n_steps=0,
                                                                         coord_as_vert=True)
                            morphedCenter = tempCoordLabel.morph(subject_from=subjects[otherSubj], subject_to=subjects[soi],
                                                                 n_jobs=n_jobs, copy=True, smooth=0)



                            labelInds = label_bank[otherSubj][0][labelInd].vertices
                            # generate binary index list that has 1 at inds where the label is
                            existingVerts = np.r_[np.in1d(vertNum_bank[otherSubj][0], labelInds) * 1,
                                                np.zeros(len(vertNum_bank[otherSubj][1]))]
                            # Convert vertices from all other subjects to soi
                            #convertedLabelInds[:, ind] = convSrc_bank[otherSubj][soi].dot(existingVerts)

                            convertedLabelInds[:, ind] = np.sum((convSrc_bank[otherSubj][soi].A)[:, existingVerts>0], axis=1)
                            '''
                        #######################################################

                        if(clfScheme == clfSchemes[0]):
                            # Weight training and testing matrices
                            train_pool_cent = train_pool * centroid_weights
                            test_pool_cent = test_pool * centroid_weights
                            # Train and test LDA algorithm
                            clf_cent = LDA()
                            clf_cent.fit(train_pool_cent, y_train_pool,
                                         store_covariance=False)
                            centroid_accuracyBlock[li, snri, ri, soi] = \
                                clf_cent.score(test_pool_cent, y_test_pool)
                        elif(clfScheme == clfSchemes[1]):
                            # Weight training and testing matrices
                            train_pool_cent = train_pool * centroid_weights
                            test_pool_cent = test_pool * centroid_weights

                            # Train and test Regularized LDA algorithm
                            weights = ldaReg(train_pool_cent, y_train_pool, regFactors)[:, :, 0]
                            test_pool_cent = np.c_[np.ones((len(y_test_pool), 1)), test_pool_cent]
                            LDAOutput = test_pool_cent.dot(weights)
                            pred_cent = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                            centroid_accuracyBlock[li, snri, ri, soi] = \
                                np.mean(pred_cent == y_test_pool)
                        elif(clfScheme == clfSchemes[2]):
                            ### Train and test SVM
                            # Scale inputs to [-1 1] as SVM is scale sensitive
                            scaler_cent = preprocessing.data.StandardScaler().fit(train_pool)
                            train_pool_cent = scaler_cent.transform(train_pool) * centroid_weights
                            test_pool_cent = scaler_cent.transform(test_pool) * centroid_weights

                            clf_cent = SVC(cache_size=2048)
                            ###################################################
                            # SVM Grid Search
                            param_grid = [{'kernel': ['linear'], 'C': C_range}]
                            #cv = StratifiedKFold(y=y_train_pool, n_folds=3)
                            gridSearch = GridSearchCV(clf_cent,
                                                      param_grid=param_grid,
                                                      pre_dispatch=n_jobs)
                            gridSearch.fit(train_pool_cent, y_train_pool)
                            #gridSearch.fit(train_pool, y_train_pool)
                            centroid_accuracyBlock[li, snri, ri, soi] = \
                                gridSearch.score(test_pool_cent, y_test_pool)
                            '''
                            ###################################################
                            # Set parameter estimation
                            #clf_cent.set_params(kernel='rbf', C=1000, gamma=5e-5
                            clf_cent.set_params(kernel='linear', C=1000)
                            clf_cent.fit(train_pool_cent, y_train_pool)
                            centroid_accuracyBlock[li, snri, ri, soi] = \
                                clf_cent.score(test_pool_cent, y_test_pool)
                            '''
                            '''
                            #print('Best Classifier is: ', gridSearch.best_estimator_)
                            C_optimum[li, snri, ri, soi] = gridSearch.best_estimator_.C
                            #g_optimum[li, snri, ri, soi] = gridSearch.best_estimator_.gamma
                            '''

                        #######################################################
                        ### KDE weighting classifier
                        convertedLabelInds = np.zeros((len(invMat_bank[soi]),
                                                       len(otherSubjs)))
                        otherSubjs = np.delete(range(len(subjects)), soi)
                        # Get vertices for all subjects for the given label
                        for ind, otherSubj in enumerate(otherSubjs):
                            hemi = (0, 1)[label.hemi == 'rh']

                            labelInds = label_bank[otherSubj][0][labelInd].vertices
                            # generate binary index list that has 1 at inds where the label is
                            existingVerts = np.r_[np.in1d(vertNum_bank[otherSubj][0], labelInds) * 1,
                                                  np.zeros(len(vertNum_bank[otherSubj][1]))]
                            # Convert vertices from all other subjects to soi
                            #convertedLabelInds[:, ind] = convSrc_bank[otherSubj][soi].dot(existingVerts)

                            convertedLabelInds[:, ind] = np.sum((convSrc_bank[otherSubj][soi].A)[:, existingVerts > 0], axis=1)

                        dipoleMags = np.mean(convertedLabelInds, axis=1)
                        hemiMags = [dipoleMags[:len(vdist_bank[soi][0])],
                                    dipoleMags[len(vdist_bank[soi][0]):]]

                        # Compute KDE exponential weight
                        hemiWeights = []
                        for hemi in [0, 1]:
                            hemiWeights.append(np.sum((np.exp(expFactor_KDE *
                                                              vdist_bank[soi][hemi]) *
                                                       hemiMags[hemi]), axis=1))
                        kdeSrc_weights = np.r_[hemiWeights[0], hemiWeights[1]]
                        #kdeSrc_weights /= max(np.abs(kdeSrc_weights))
                        kde_weights = fwdMat_bank[soi].dot(kdeSrc_weights)
                        kde_weights = np.abs(kde_weights) / np.max(np.abs(kde_weights))
                        #import pdb; pdb.set_trace()

                        # Apply weights
                        if(clfScheme == clfSchemes[0]):
                            # Weight training and testing matrices
                            train_pool_kde = train_pool * kde_weights
                            test_pool_kde = test_pool * kde_weights
                            # Train and test LDA algorithm
                            clf_kde = LDA()
                            clf_kde.fit(train_pool_kde, y_train_pool, store_covariance=False)
                            kde_accuracyBlock[li, snri, ri, soi] = \
                                clf_kde.score(test_pool_kde, y_test_pool)
                        elif(clfScheme == clfSchemes[1]):
                            # Weight training and testing matrices
                            train_pool_kde = train_pool * kde_weights
                            test_pool_kde = test_pool * kde_weights

                            # Train and test Regularized LDA algorithm
                            weights = ldaReg(train_pool_kde, y_train_pool, regFactors)[:, :, 0]
                            test_pool_kde = np.c_[np.ones((len(y_test_pool), 1)), test_pool_kde]
                            LDAOutput = test_pool_kde.dot(weights)
                            pred_kde = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                            kde_accuracyBlock[li, snri, ri, soi] = \
                                np.mean(pred_kde == y_test_pool)
                        elif(clfScheme == clfSchemes[2]):
                            ### Train and test SVM
                            # Scale inputs to [-1 1] as SVM is scale sensitive
                            scaler_kde = preprocessing.data.StandardScaler().fit(train_pool)
                            train_pool_kde = scaler_kde.transform(train_pool) * kde_weights
                            test_pool_kde = scaler_kde.transform(test_pool) * kde_weights

                            clf_kde = SVC(cache_size=2048)
                            ###################################################
                            # Grid Search
                            param_grid = [{'kernel': ['linear'], 'C': C_range}]
                            #cv = StratifiedKFold(y=y_train_pool, n_folds=3)
                            gridSearch = GridSearchCV(clf_kde, param_grid=param_grid,
                                                      pre_dispatch=n_jobs)
                            gridSearch.fit(train_pool_kde, y_train_pool)
                            #gridSearch.fit(train_pool, y_train_pool)
                            kde_accuracyBlock[li, snri, ri, soi] = \
                                gridSearch.score(test_pool_kde, y_test_pool)

                            C_optimum[li, snri, ri, soi] = gridSearch.best_estimator_.C
                            #print('Best Classifier is: ', gridSearch.best_estimator_)
                            #g_optimum[li, snri, ri, soi] = gridSearch.best_estimator_.gamma
                            '''
                            ###################################################
                            # Set parameter estimation
                            #clf_kde.set_params(kernel='rbf', C=1000, gamma=5e-5
                            clf_kde.set_params(kernel='linear', C=1000)
                            clf_kde.fit(train_pool_kde, y_train_pool)
                            kde_accuracyBlock[li, snri, ri, soi] = \
                                clf_kde.score(test_pool_kde, y_test_pool)
                            '''
            print '=',
            sys.stdout.flush()
        print '] Done (' + str(ti + 1) + '/' + str(len(trial_counts)) + ')'

        scoreDict['logRatio']['kde'].append(kde_logRatioBlock)
        scoreDict['logRatio']['centroid'].append(centroid_logRatioBlock)
        scoreDict['logRatio']['unweighted'].append(unweighted_logRatioBlock)
        scoreDict['logRatio']['std'].append(std_logRatioBlock)

        scoreDict['accuracy']['kde'].append(kde_accuracyBlock)
        scoreDict['accuracy']['centroid'].append(centroid_accuracyBlock)
        scoreDict['accuracy']['unweighted'].append(unweighted_accuracyBlock)
        scoreDict['accuracy']['std'].append(std_accuracyBlock)
    ###############################################################################
    # Plot Prep

    # Get mean accuracies for each classification method
    for keyInd in range(len(scoreDict['accuracy'])):
        tempScores = [np.mean(scoreDict['accuracy'][class_methods[keyInd]][i],
                              axis=(0, 2, 3)) for i in
                      range(len(scoreDict['accuracy'][class_methods[keyInd]]))]
        # each classification method is trials x SNR percentage
        meanDataDict['accuracy'][class_methods[keyInd]] = 100 * np.array(tempScores)

    # Compute differences and convert to an np array (reshaped)
    ls_accuracyDif = []
    for keyInd1 in range(len(scoreDict['accuracy'])):
        diffSet = []
        for keyInd2 in range(len(scoreDict['accuracy'])):
            diffSet.append(meanDataDict['accuracy'][class_methods[keyInd1]] -
                           meanDataDict['accuracy'][class_methods[keyInd2]])
        ls_accuracyDif.append(diffSet)
    accuracyDifTemp = np.array(ls_accuracyDif)

    #reshape for easier plotting
    accuracyDif = accuracyDifTemp.reshape(-1, accuracyDifTemp.shape[2],
                                          accuracyDifTemp.shape[3])

    # Compute means across labels for brain plots
    deltaAccuracy = np.zeros((len(trial_counts), len(labelList), len(snrs)))
    for trialCountInd in range(len(trial_counts)):
        deltaAccuracy[trialCountInd, :, :] = (np.mean(scoreDict['accuracy']['kde'][trialCountInd], axis=(2, 3)) -
                                              np.mean(scoreDict['accuracy']['std'][trialCountInd], axis=(2, 3))) * 100

    # Save performance results
    if saveAnalysis:
        # Pickle non-matrix objects
        with open(op.join(cache_fname, 'meanDataDict.pkl'), 'wb') as outfile:
            cPickle.dump([meanDataDict, class_methods, labelList], outfile)
            #json.dump(meanDataDict, outfile)
        # Pickle and save standard score for use in nSubjectsGain plot to
        #     compare pooled learning vs std learning
        with open(op.join(cache_fname, 'stdScoreDict.pkl'), 'wb') as outfile:
            cPickle.dump([scoreDict['accuracy']['std'], labelList], outfile)

        savemat(op.join(cache_fname, 'simResults.mat'), {'accuracyDif': accuracyDif, 'deltaAccuracy': deltaAccuracy,
                                                         'snrs': np.array(snrs), 'trial_counts': np.array(trial_counts),
                                                         'kdeSrc_weights': kdeSrc_weights,
                                                         'centroidSrc_weights': centroidSrc_weights})

    elapsed_time = time() - start_time
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    print 'Time Elasped: ' + '%d:%02d:%02d' % (h, m, s)

else:
    # If we're not redoing the analysis, load performance results
    with open(op.join(cache_fname, 'meanDataDict.pkl'), 'rb') as infile:
        [meanDataDict, class_methods, labelList] = cPickle.load(infile)

    storedMat = loadmat(op.join(cache_fname, 'simResults.mat'))
    accuracyDif = storedMat['accuracyDif']
    deltaAccuracy = storedMat['deltaAccuracy']
    snrs = list(np.squeeze(storedMat['snrs']))
    trial_counts = list(np.squeeze(storedMat['trial_counts']))

###############################################################################
# Plots
import matplotlib as mpl
# Use Agg backend so allow pdfs to generate properly
mpl.use('Agg')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
import os

from pooledBCI_plotAbsPerformance import pooledBCI_plotAbsPerformance
from pooledBCI_plotRelPerformance import pooledBCI_plotRelPerformance
from pooledBCI_plotRelPerformanceCompact import pooledBCI_plotRelPerformanceCompact
from pooledBCI_plotROIPerformance import pooledBCI_plotROIPerformance
from pooledBCI_plotROIPerformanceMedLat import pooledBCI_plotROIPerformanceMedLat
from pooledBCI_plotROIPerformanceMedLatSingleSNR import pooledBCI_plotROIPerformanceMedLatSingleSNR

plt.close('all')
plt.ion()
mpl.rcParams['pdf.fonttype'] = 42

titles = ['Pooled Anatom. Gauss. Mixture', 'Pooled Anatom. Centroid',
          'Pooled Unweighted', 'Traditional Leave-One-Out\nClassifier']
classificationDetails = 'Source ' + clfScheme + ': ' + str(regFactors[0]) + \
    ' nSubjects=' + str(len(subjects)) + ' nLabels= ' + str(len(labelList))
plotList = []

nPlots = len(meanDataDict['accuracy'])


###############################################################################
## Figure 0: Absolute performance
if 0 in plotsToGen:
    print 'Plotting Figure 0',
    # Call Figure 0 Function
    plotList.append(pooledBCI_plotAbsPerformance(meanDataDict, snrs,
                                                 trial_counts, class_methods,
                                                 titles, len(subjects),
                                                 len(labelList)))
    print ' ... Done'
###############################################################################
### Figure 1: plot difference in performance between classification methods
if 1 in plotsToGen:
    print 'Plotting Figure 1',
    plotList.append(pooledBCI_plotRelPerformanceCompact(accuracyDif, snrs,
                                                        trial_counts,
                                                        class_methods, titles))
    print ' ... Done'
# Plot difference in performance between all classification methods
if 2 in plotsToGen:
    print 'Plotting Figure 2',
    '''
        fwd_fname = op.join(subjectDir, subj, subj + '-7-fwd-eeg.fif')

    # Load forward solution
    fwd = mne.read_forward_solution(fwd_fname, force_fixed=True,
                                    surf_ori=False)
    fwd = mne.fiff.pick_types_forward(fwd, meg=False, eeg=True,
                                    ref_meg=False, exclude='bads')

    fwd_bank.append(fwd)
    info = deepcopy(fwd['info'])
    info['projs'] = []
    vertices = [s['vertno'] for s in fwd['src']]
    n_src = sum([len(v) for v in vertices])
    hemi = 0
    '''

    brainPlots = []
    brainVizSize = [1000, 700]
    flim = [0., 2.5, 10.]
    cm = mne.viz.mne_analyze_colormap(flim)
    cm_mpl = mne.viz.mne_analyze_colormap(flim, format='matplotlib')

    #######################################
    # generate individual brain image plot
    # Call Figure 2 Function

    # Plot single SNR Med/Lat row
    '''
    plotList.append(pooledBCI_plotROIPerformanceMedLat(deltaAccuracy, brainPlotTrialInds,
                    snrs, trial_counts[-1], labelList, fs_srcs, brainVizSize, flim, cm))
                    '''

    # Plot single SNR Med/Lat row
    snrIndToUse = 1
    plotList.append(pooledBCI_plotROIPerformanceMedLatSingleSNR(deltaAccuracy,
                    brainPlotTrialInds, snrs, snrIndToUse, trial_counts[-1],
                    labelList, fs_srcs, brainVizSize, flim, cm))

    print ' ... Done'
##############################################################################
## Figure 3: Brain Plots by Label
if 3 in plotsToGen:
    print 'Plotting Figure 3',
    subj = 'AKCLEE_101'
    if(subjSetNum == 0):
        fwd_fname = op.join(subjectDir, subj, subj + '-2-fwd.fif')
    else:
        fwd_fname = op.join(subjectDir, subj, subj + '-7-fwd-eeg.fif')

    # Load forward solution
    fwd = mne.read_forward_solution(fwd_fname, force_fixed=True,
                                    surf_ori=False)
    fwd = mne.fiff.pick_types_forward(fwd, meg=False, eeg=True,
                                      ref_meg=False, exclude='bads')

    fwd_bank.append(fwd)
    info = deepcopy(fwd['info'])
    info['projs'] = []
    vertices = [src['vertno'] for src in fwd['src']]
    n_src = sum([len(v) for v in vertices])
    hemi = 0

    brainPlots = []
    brainVizSize = [1000, 700]
    flim = [0., 5., 20.]
    cm = mne.viz.mne_analyze_colormap(flim)
    cm_mpl = mne.viz.mne_analyze_colormap(flim, format='matplotlib')

    #######################################
    # generate individual brain image plot
    # Call Figure 3 Function

    plotList.append(pooledBCI_plotROIPerformance(deltaAccuracy,
                                                 brainPlotTrialInds, snrs,
                                                 trial_counts, labelList,
                                                 fs_srcs, brainVizSize, flim,
                                                 cm))
    print ' ... Done'

###############################################################################
### Save plots
if savePlots:
    # Find last folder in series
    foldNames = [folder.split('/')[-1]
                 for folder in glob.glob(saveDir + '/Paper5_v*')]
    if len(foldNames) > 0:
        versionInd = [string.find('v') + 1 for string in foldNames]
        existVersions = [int(foldNames[i][versionInd[i]:versionInd[i] + 2])
                         for i in range(len(foldNames))]
        # Create version number
        version = '%02d' % (max(existVersions) + 1)
    else:
        version = '01'

    # Make Folder
    plotFolder_fname = op.join(saveDir, 'Paper5_v' + version + '_cent' +
                               str(int(expFactor_centroid)) + '_kde' +
                               str(int(expFactor_KDE)))
    if not op.exists(plotFolder_fname):
        os.makedirs(plotFolder_fname)
    for i, plotInd in enumerate(plotsToGen):
        for fType in plotFormat:
            plotList[i].savefig(plotFolder_fname + '/fig_' + str(plotsToGen[i]) +
                                '_cent' + str(int(expFactor_centroid)) + '_kde' +
                                str(int(expFactor_KDE)) + '.' + fType)

plt.show()
