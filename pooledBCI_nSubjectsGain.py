#dpylint: disable-msg=C0103
'''
pooledBCI_nSubjectsGain.py
Plots performance as a function of number of subjects in pool

Requires that all labels being used have had std analysis run
previously (pooledBCI_Sens.py)
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
import random
from scipy.spatial.distance import cdist
from scipy.io import loadmat, savemat
from scipy.stats import sem
from sklearn import preprocessing
from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from ldaReg import ldaRegWeights as ldaReg
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
                  'AnatomBCI_Figures_Python', 'PaperFig_poolSize')

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
poolSizes = [2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 20]  # Pool sizes for paper
#poolSizes = [2, 4, 5, 10, 15, 20]
#poolSizes = [2, 3, 4, 5]
assert max(poolSizes) < len(subjects), 'Max number of desired subjects in the \
    pool is smaller than total number of subjects.'

#reinitialize forwards, inverse, covariance
doAnalysis = False
saveAnalysis = False

savePlots = False
re_init = True
saveIndividualInfo = False
saveConvBank = False
loadConvBank = True
clfSchemes = ['LDA', 'LDA_Reg', 'SVM']
clfScheme = clfSchemes[1]
plotsToGen = [1]

weightMethods = ['kde', 'centroid', 'unweighted']

labels_all = []

labelNames_motor = []
labelNames_p300 = []
labelNames_ssvep_auditory = []
labelNames_space_pitch = []

labelNames_motor = ['S_precentral-sup-part-lh',
                    'G_precentral_handMotor_radius_15mm-lh',
                    'G_precentral_handMotor_radius_10mm-lh',
                    'G_precentral_handMotor_radius_5mm-lh']
labelNames_p300 = ['G_front_middle-lh',
                   'S_front_inf-lh',
                   'G_pariet_inf-Supramar-lh',
                   'G_parietal_sup-lh',
                   'G_pariet_inf-Angular-lh',
                   'S_intrapariet_and_P_trans-lh']

labelNames_ssvep_auditory = ['Pole_occipital-lh',
                             'S_calcarine-lh',
                             'G_cuneus-lh',
                             'G_temp_sup-G_T_transv-lh']
labelNames_space_pitch = ['UDRon-UDStd_01-lh',
                          'LRRon-LRStd_01-rh']

labels_all.extend(labelNames_motor)
labels_all.extend(labelNames_p300)
labels_all.extend(labelNames_ssvep_auditory)
#labels_all.extend(labelNames_space_pitch)

labelList, _ = mne.labels_from_parc(subject='fsaverage', parc='aparc.a2009s')


labelList = [elem for elem in labelList
             if elem.name in labels_all and elem.hemi == 'lh' and
             'Jensen' not in elem.name]
for label_name in labelNames_motor:
    if 'handMotor_radius' in label_name:
        label_fname = op.join(structDir, 'fsaverage', 'label', label_name + '.label')
        labelList.append(mne.read_label(label_fname, subject='fsaverage'))
for label_name in labelNames_space_pitch:
    label_fname = op.join(structDir, 'fsaverage', 'label', label_name + '.label')
    labelList.append(mne.read_label(label_fname, subject='fsaverage'))
#labelList = [elem for elem in labelList
#             if (elem.name[0] == 'G' or elem.name[0] == 'S') and elem.hemi == 'lh' and
#                'Jensen' not in elem.name]

n_smooth = 5
lambda2 = 1. / 10000.
regFactors = [0.05]
n_jobs = 6
#######################################
# higher magnitude = faster rolloff with increasing distance
expFactor_centroid = -30
expFactor_KDE = -30
#######################################

# Activity simulation params
tstep = 1e-3
snr = -10
trial_counts = [40]
max_trials = max(trial_counts)
current_mag = 1.
repeats = 25  # 25 repeats used for paper quality
C_range = 10.0 ** np.arange(-6, -4)
gamma_range = 10.0 ** np.arange(-2, 3)
levelRatio = np.zeros((len(subjects), len(subjects)))

# If in debugging mode, make simulation faster
if len(subjects) < 6:
    repeats = 1
    snr = -10
    trial_counts = [40]
    #lenLabelSetToRun = 3
    #label_inds = np.random.randint(0, len(labelList), (lenLabelSetToRun))
    #labelList = [labelList[i] for i in label_inds]
    #labelList = [label for label in labelList if 'UDRon' in label.name or
    #             'LRRon' in label.name]

nLabels = len(labelList)

#load fsaverage information
fs_vertices = [np.arange(10242), np.arange(10242)]
n_src_fs = sum([len(i) for i in fs_vertices])
fs_srcs = mne.read_source_spaces(op.join(structDir, 'fsaverage', 'bem',
                                         'fsaverage-5-src.fif'))

#File names
fileBanks = ['fwd_bank', 'fwdmat_bank', 'invMat_bank', 'noiseCov_bank'
             'conv_bank', 'vdist_bank']
if(subjSetNum == 0):
    cache_fname = op.join(subjectDir, 'RON__cache')
else:
    cache_fname = op.join(subjectDir, 'AKCLEE__cache')

subjTxt_fname = op.join(cache_fname, 'included_subjects.txt')
subjBank_fname = op.join(cache_fname, 'banks.pkl')

#initialize lists
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
fakeEvoked_bank = []
fwdColorers = []
sensorNoise = []

start_time = time()
###############################################################################
### Initializing subject data ###
if doAnalysis:
    # Score matrix
    scoreDict = {'logRatio': {}, 'accuracy': {}}
    meanDataDict = deepcopy(scoreDict)
    labelAccuracy = {'accuracy': {}}

    for method in weightMethods:
        scoreDict['logRatio'] = {method: [] for method in weightMethods}
        scoreDict['accuracy'][method] = []
        labelAccuracy['accuracy'][method] = {method: []
                                             for method in weightMethods}

    if re_init:
        print '!!! COMMENCE SIMULATION !!! (@ ' + strftime('%H:%M:%S') + ')'
        print 'nSubjs:\t\t' + str(len(subjects))
        print 'poolSizes:\t' + str(poolSizes)
        print 'Trials:\t\t' + str(trial_counts)
        print 'SNR:\t\t' + str(snr)
        print 'nLabels:\t' + str(nLabels)
        print 'nRepeats:\t' + str(repeats) + '\n'
        print 'Processing fwd, inv, noise cov, etc:'
        for si, subj in enumerate(subjects):

            print '  ' + subj,
            sys.stdout.flush()
            #load/generate forwards
            if(subjSetNum == 0):
                fwd_fname = op.join(subjectDir, subj, subj + '-2-fwd.fif')
                cov_fname = op.join(subjectDir, subj, subj + '-noise-cov.fif')
                inv_fname = op.join(subjectDir, subj, subj + '_eeg-1-inv.fif')
            else:
                fwd_fname = op.join(subjectDir, subj, subj + '-7-fwd-eeg.fif')
                cov_fname = op.join(subjectDir, subj, subj + '-noise-cov-eeg.fif')
                inv_fname = op.join(subjectDir, subj, subj + '-inv-eeg-python.fif')

            src_fname = op.join(structDir, subj, 'bem', subj + '-7-src.fif')

            # Load forward solution #
            fwd = mne.read_forward_solution(fwd_fname, force_fixed=True,
                                            surf_ori=False)
            fwd = mne.fiff.pick_types_forward(fwd, meg=False, eeg=True,
                                              ref_meg=False, exclude='bads')

            fwd_bank.append(fwd)
            info = deepcopy(fwd['info'])
            info['projs'] = []
            vertices = [s['vertno'] for s in fwd['src']]
            n_src = sum([len(v) for v in vertices])

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

            #Load labels from parcellation
            label_bank.append(mne.labels_from_parc(subj, parc='aparc.a2009s'))
            label_bank[si][1][:] = []  # Clear out ROI color info

            #Load custom labels too and add them onto the end
            for label_name in labelNames_motor:
                if 'handMotor_radius' in label_name:
                    label_fname = op.join(structDir, subj, 'label', label_name + '.label')
                    label_bank[si][0].append(mne.read_label(label_fname, subject=subj))
            for label_name in labelNames_space_pitch:
                #label_fname = op.join(structDir, subj, 'label', label_name + '.label')
                #label_bank[si][0].append(mne.read_label(label_fname, subject=subj))
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
            # Euclidean way
            euclidean = False
            if euclidean:
                temp_dists = [distance_matrix(hemiVerts, hemiVerts)
                              for hemiVerts in vert_coord]
                vdist_bank.append([hemi / hemi.max() for hemi in temp_dists])
            else:
                src = mne.read_source_spaces(fname=src_fname)
                temp_dists = [src[hemi]['dist'][vertices[hemi]][:, vertices[hemi]].A
                              for hemi in range(len(src))]
                vdist_bank.append([hemi / hemi.max() for hemi in temp_dists])

            # Save vertices, channel names
            vertNum_bank.append(vertices)

            if(saveIndividualInfo):
                '''
                savemat(op.join(subjectDir, subj, subj + '_cache.mat'),
                        {'fwd_bank': fwd_bank, 'fwdMat_bank': fwdMat_bank})
                'invMat_bank': invMat_bank, 'noiseCov_bank': noiseCov_bank,
                'vdist_bank': vdist_bank, 'fakeEvoked_bank': fakeEvoked_bank,
                'label_bank': label_bank})
                '''
            print '... ' + 'Done (' + str(si + 1) + '/' + str(len(subjects)) + ')'

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

                    # Continue magnitude ratio calculation
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

    ###########################################################################
    # Activity Simulation
    rng = np.random.RandomState()
    databank = np.zeros((len(labelList), len(trial_counts),
                        len(subjects), 2 * max(trial_counts),
                        fwd_bank[0]['nchan']))

    print 'Simulating/Classifying Data'

    ### Iteration guide
    # trials - number of training trials for the classifier
                # Labels - each label in the parcellation
                    #SNR - several signal to noise ratios
                        #repeats- number of times to repeat the classification task
                            #subj - make each subj the subject of interest one time

    for ti, n_trials in enumerate(trial_counts):
        print '  ' + str(n_trials) + ' Trial Group [',
        sys.stdout.flush()

        current = np.ones((1, n_trials)) * current_mag

        #  nLabels x nRepeats x nSubjs x off and on classification

        unweighted_logRatioBlock = np.zeros((len(labelList), len(poolSizes),
                                             repeats, len(subjects),
                                             2 * n_trials))
        unweighted_accuracyBlock = np.zeros((len(labelList), len(poolSizes),
                                             repeats, len(subjects)))
        C_optimum = np.zeros((len(labelList), len(poolSizes), repeats,
                              len(subjects)))
        g_optimum = np.zeros((len(labelList), len(poolSizes), repeats,
                              len(subjects)))

        centroid_logRatioBlock = np.zeros((len(labelList), len(poolSizes),
                                           repeats, len(subjects),
                                           2 * n_trials))
        centroid_accuracyBlock = np.zeros((len(labelList), len(poolSizes),
                                           repeats, len(subjects)))
        kde_logRatioBlock = np.zeros((len(labelList), len(poolSizes), repeats,
                                      len(subjects), 2 * n_trials))
        kde_accuracyBlock = np.zeros((len(labelList), len(poolSizes), repeats,
                                      len(subjects)))
        for li, label in enumerate(labelList):
            for pi, poolSize in enumerate(poolSizes):
                for ri in range(repeats):

                    trialBlock = []
                    powerMeas = []
                    for si, subj in enumerate(subjects):
                        # Generate evoked data (sensor space)
                        evoked_template = fakeEvoked_bank[si]
                        #######################################################
                        # Generate and store evoked data for one subject
                        tempHemi = ([0, 1], [1, 0])[label.hemi == 'rh']
                        labelInd = [l.name for l in label_bank[si][tempHemi[0]]].index(label.name)
                        stc = generate_stc(src=fwd_bank[si]['src'],
                                           labels=[label_bank[si][tempHemi[0]][labelInd]],
                                           stc_data=current, tmin=0, tstep=tstep)
                        evoked = mne.simulation.generate_evoked(fwd_bank[si], stc, evoked_template,
                                                                noiseCov_bank[si],
                                                                snr=snr, random_state=rng)
                        evoked_sig = mne.simulation.generate_evoked(fwd_bank[si], stc, evoked_template,
                                                                    noiseCov_bank[si],
                                                                    snr=np.inf, random_state=rng)
                        # generate evoked data by subtracting pure signal from
                        # evoked data
                        #evoked_0.data -= evoked_0_sig.data
                        trialBlock.append(np.array([(evoked.data - evoked_sig.data).T,
                                                    evoked.data.T]))

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

                        # Morphed Data is [soi]  x (nOtherSubj x on/off x trials x electrodes
                        morphedData.append(morphedData1Subj)

                    ###########################################################
                    # BEGIN POOLED TRAINING
                    for soi in range(len(subjects)):
                        # Subselect training data to modify pool size
                        poolInds = random.sample(range(len(morphedData[soi])),
                                                 poolSize)

                        # Training data comes from other subjects
                        train_0 = np.reshape(morphedData[soi][poolInds][:, 0, :, :], (-1, fwdMat_bank[soi].shape[0]))
                        train_1 = np.reshape(morphedData[soi][poolInds][:, 1, :, :], (-1, fwdMat_bank[soi].shape[0]))
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
                        # For each subject, train on all other subjects and
                        # then test on subject of interest
                        if(clfScheme == clfSchemes[0]):
                            # Train and test LDA algorithm
                            clf_unwt = LDA()
                            clf_unwt.fit(train_pool, y_train_pool, store_covariance=False)
                            unweighted_accuracyBlock[li, pi, ri, soi] = \
                                clf_unwt.score(test_pool, y_test_pool)
                        elif(clfScheme == clfSchemes[1]):
                            # Train and test Regularized LDA algorithm
                            ldaFactors = ldaReg(train_pool, y_train_pool, regFactors)[:, :, 0]
                            test_pool_unwt = np.c_[np.ones((len(y_test_pool), 1)), test_pool]
                            LDAOutput = test_pool_unwt.dot(ldaFactors)
                            pred_unwt = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                            unweighted_accuracyBlock[li, pi, ri, soi] = \
                                np.mean(pred_unwt == y_test_pool)
                            '''

                            # Train and test Regularized LDA algorithm
                            weights_all = ldaReg(train_pool, y_train_pool, regFactors)
                            test_pool_unwt = np.c_[np.ones((len(y_test_pool), 1)), test_pool]
                            for i in range(len(regFactors)):
                                LDAOutput = test_pool_unwt.dot(weights_all[:, :, i])
                                pred_unwt = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                                unweighted_accuracyBlock[li, pi, ri, soi, i] = \
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
                            unweighted_accuracyBlock[li, pi, ri, soi] = \
                                gridSearch.score(test_pool_unwt, y_test_pool)

                            #print('Best Classifier is: ', gridSearch.best_estimator_)
                            C_optimum[li, pi, ri, soi] = gridSearch.best_estimator_.C
                            #g_optimum[li, pi, ri, soi] = gridSearch.best_estimator_.gamma
                            '''
                            ###################################################
                            # Set parameter estimation
                            #clf_unwt.set_params(kernel='rbf', C=1000, gamma=5e-5
                            clf_unwt.set_params(kernel='linear', C=1e-1)
                            clf_unwt.fit(train_pool_unwt, y_train_pool)
                            unweighted_accuracyBlock[li, pi, ri, soi] = \
                                clf_unwt.score(test_pool_unwt, y_test_pool)

                        '''
                        y_pred_log_probs = lda.predict_log_proba(test_pool_scaled)
                        unweighted_logRatioBlock[li, pi, ri, soi, :] = \
                            (y_pred_log_probs[:, 0] - y_pred_log_probs[:, 1])
                        '''

                        #######################################################
                        ### Centroid classifier
                        # Find label center and calculate centroid distances
                        h = ([0, 1], [1, 0])[label.hemi == 'rh']
                        labelInd = [l.name for l in label_bank[soi][h[0]]].index(label.name)

                        # Get label mean position for soi and find vertex closest to center
                        labelAvgPos = np.reshape(np.mean(a=label_bank[soi][h[0]][labelInd].pos, axis=0),
                                                (-1, 3))
                        centerVertInd = np.argmin(cdist(labelAvgPos, vertPos_bank[soi][h[0]]))

                        # Pull distances from the most central point
                        dists = vdist_bank[soi][h[0]][centerVertInd]

                        # Make sure we calculate for both hemispheres
                        if(h[0] == 0):
                            dists = np.r_[dists, np.ones(len(vdist_bank[soi][h[1]])) * 10 * max(dists)]
                        else:
                            dists = np.r_[np.ones(len(vdist_bank[soi][h[1]])) * 10 * max(dists), dists]

                        centroidSrc_weights = np.exp(expFactor_centroid * dists ** 2)
                        centroid_weights = fwdMat_bank[soi].dot(centroidSrc_weights)
                        centroid_weights = np.abs(centroid_weights) / np.max(np.abs(centroid_weights))

                        if(clfScheme == clfSchemes[0]):
                            # Weight training and testing matrices
                            train_pool_cent = train_pool * centroid_weights
                            test_pool_cent = test_pool * centroid_weights
                            # Train and test LDA algorithm
                            clf_cent = LDA()
                            clf_cent.fit(train_pool_cent, y_train_pool,
                                         store_covariance=False)
                            centroid_accuracyBlock[li, pi, ri, soi] = \
                                clf_cent.score(test_pool_cent, y_test_pool)
                        elif(clfScheme == clfSchemes[1]):
                            # Weight training and testing matrices
                            train_pool_cent = train_pool * centroid_weights
                            test_pool_cent = test_pool * centroid_weights

                            # Train and test Regularized LDA algorithm
                            ldaFactors = ldaReg(train_pool_cent, y_train_pool,
                                                regFactors)[:, :, 0]
                            test_pool_cent = np.c_[np.ones((len(y_test_pool), 1)), test_pool_cent]
                            LDAOutput = test_pool_cent.dot(ldaFactors)
                            pred_cent = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                            centroid_accuracyBlock[li, pi, ri, soi] = \
                                np.mean(pred_cent == y_test_pool)
                        elif(clfScheme == clfSchemes[2]):
                            ### Train and test SVM
                            # Scale inputs to [-1 1] as SVM is scale sensitive
                            scaler_cent = preprocessing.data.StandardScaler().fit(train_pool)
                            train_pool_cent = scaler_cent.transform(train_pool) * centroid_weights
                            test_pool_cent = scaler_cent.transform(test_pool) * centroid_weights

                            clf_cent = SVC(cache_size=2048)
                            ###################################################
                            # Grid Search
                            param_grid = [{'kernel': ['linear'], 'C': C_range}]
                            #cv = StratifiedKFold(y=y_train_pool, n_folds=3)
                            gridSearch = GridSearchCV(clf_cent,
                                                      param_grid=param_grid,
                                                      pre_dispatch=n_jobs)
                            gridSearch.fit(train_pool_cent, y_train_pool)
                            #gridSearch.fit(train_pool, y_train_pool)
                            centroid_accuracyBlock[li, pi, ri, soi] = \
                                gridSearch.score(test_pool_cent, y_test_pool)
                            '''
                            ###################################################
                            # Set parameter estimation
                            #clf_cent.set_params(kernel='rbf', C=1000, gamma=5e-5
                            clf_cent.set_params(kernel='linear', C=1000)
                            clf_cent.fit(train_pool_cent, y_train_pool)
                            centroid_accuracyBlock[li, ri, soi] = \
                                clf_cent.score(test_pool_cent, y_test_pool)
                            '''
                            '''
                            #print('Best Classifier is: ', gridSearch.best_estimator_)
                            C_optimum[li, ri, soi] = gridSearch.best_estimator_.C
                            #g_optimum[li, ri, soi] = gridSearch.best_estimator_.gamma
                            '''

                        #######################################################
                        ### KDE classifier
                        convertedLabelInds = np.zeros((len(invMat_bank[soi]),
                                                       len(otherSubjs)))
                        otherSubjs = np.delete(range(len(subjects)), soi)
                        # Get vertices for all subjects for the given label
                        for ind, otherSubj in enumerate(otherSubjs):
                            #hemi = 0 if label_bank[otherSubj][0][labelInd].hemi == 'lh' else 1
                            tempHemi = ([0, 1], [1, 0])[label.hemi == 'rh']

                            labelInds = label_bank[otherSubj][tempHemi[0]][labelInd].vertices
                            # generate binary index list that has 1 at inds where the label is
                            existingVerts = np.r_[np.in1d(vertNum_bank[otherSubj][tempHemi[0]], labelInds) * 1,
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
                        kde_weights = fwdMat_bank[soi].dot(kdeSrc_weights)
                        kde_weights = np.abs(kde_weights) / np.max(np.abs(kde_weights))

                        if(clfScheme == clfSchemes[0]):
                            # Weight training and testing matrices
                            train_pool_kde = train_pool * kde_weights
                            test_pool_kde = test_pool * kde_weights
                            # Train and test LDA algorithm
                            clf_kde = LDA()
                            clf_kde.fit(train_pool_kde, y_train_pool, store_covariance=False)
                            kde_accuracyBlock[li, pi, ri, soi] = \
                                clf_kde.score(test_pool_kde, y_test_pool)
                        elif(clfScheme == clfSchemes[1]):
                            # Weight training and testing matrices
                            train_pool_kde = train_pool * kde_weights
                            test_pool_kde = test_pool * kde_weights

                            # Train and test Regularized LDA algorithm
                            ldaFactors = ldaReg(train_pool_kde, y_train_pool, regFactors)[:, :, 0]
                            test_pool_kde = np.c_[np.ones((len(y_test_pool), 1)), test_pool_kde]
                            LDAOutput = test_pool_kde.dot(ldaFactors)
                            pred_kde = ((LDAOutput[:, 1] - LDAOutput[:, 0]) > 0) * 1
                            kde_accuracyBlock[li, pi, ri, soi] = \
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
                            gridSearch = GridSearchCV(clf_kde,
                                                      param_grid=param_grid,
                                                      pre_dispatch=n_jobs)
                            gridSearch.fit(train_pool_kde, y_train_pool)
                            #gridSearch.fit(train_pool, y_train_pool)
                            kde_accuracyBlock[li, pi, ri, soi] = \
                                gridSearch.score(test_pool_kde, y_test_pool)

                            C_optimum[li, pi, ri, soi] = gridSearch.best_estimator_.C
                            #print('Best Classifier is: ', gridSearch.best_estimator_)
                            #g_optimum[li, ri, soi] = gridSearch.best_estimator_.gamma
                            '''
                            ###################################################
                            # Set parameter estimation
                            #clf_kde.set_params(kernel='rbf', C=1000, gamma=5e-5
                            clf_kde.set_params(kernel='linear', C=1000)
                            clf_kde.fit(train_pool_kde, y_train_pool)
                            kde_accuracyBlock[li, pi, ri, soi] = \
                                clf_kde.score(test_pool_kde, y_test_pool)
                            '''
            print '=',
            sys.stdout.flush()
        print '] Done (' + str(ti + 1) + '/' + str(len(trial_counts)) + ')'

        scoreDict['logRatio']['kde'].append(kde_logRatioBlock)
        scoreDict['logRatio']['centroid'].append(centroid_logRatioBlock)
        scoreDict['logRatio']['unweighted'].append(unweighted_logRatioBlock)

        scoreDict['accuracy']['kde'].append(kde_accuracyBlock)
        scoreDict['accuracy']['centroid'].append(centroid_accuracyBlock)
        scoreDict['accuracy']['unweighted'].append(unweighted_accuracyBlock)

    # Save performance results
    if saveAnalysis:
        # Pickle non-matrix objects
        with open(op.join(cache_fname, 'nSubjGainMeanDataDict.pkl'), 'wb') as outfile:
            cPickle.dump([scoreDict, weightMethods, labelList], outfile)

        savemat(op.join(cache_fname, 'nSubjGainSimResults.mat'), {'snr': np.array(snr),
                                                                  'trial_counts': np.array(trial_counts),
                                                                  'poolSizes': np.array(poolSizes)})

    elapsed_time = time() - start_time
    m, s = divmod(elapsed_time, 60)
    h, m = divmod(m, 60)
    print 'Finished @ \t' + strftime('%H:%M:%S')
    print 'Time Elasped: \t' + '%d:%02d:%02d' % (h, m, s)

else:
    print 'Loading Analysis ',
    # If we're not redoing the analysis, load performance results

    # nSubjGainMeanDataDict contains complete score info (scoreDict),
    # weightMethods, and labelList
    # By default, will use label list from parameter section at beginning of
    # script to define labels to plot

    # scoreDict['accuracy']['classification method'] is list (of arrays) of length trialCounts
    # Each array is  (Label) x (Pool Size) x (Repeats) x (Subjects)
    with open(op.join(cache_fname, 'nSubjGainMeanDataDict.pkl'), 'rb') as infile:
        [scoreDict, weightMethods, loadedLabelList] = cPickle.load(infile)

    # nSubjGainSimResults contains SNR, trial_counts, poolSizes
    storedMat = loadmat(op.join(cache_fname, 'nSubjGainSimResults.mat'))
    snr = np.atleast_1d(np.squeeze(storedMat['snr']))[0]
    trial_counts = list(np.atleast_1d(np.squeeze(storedMat['trial_counts'])))
    poolSizes = list(np.atleast_1d(np.squeeze(storedMat['poolSizes'])))
    print ' ... Done'

###############################################################################
# Plot Prep
meanDataDict = {'logRatio': {}, 'accuracy': {}, 'stddev': {}}
labelDataDict = {'logRatio': {}, 'accuracy': {}, 'stddev': {}}

# Get mean accuracys for each classification method
for keyInd in range(len(scoreDict['accuracy'])):
    tempScores = [np.mean(scoreDict['accuracy'][weightMethods[keyInd]][i], axis=(0, 2, 3))
                  for i in range(len(trial_counts))]
                  #for i in range(len(scoreDict['accuracy'][weightMethods[keyInd]]))]
    tempStddev = [np.std(scoreDict['accuracy'][weightMethods[keyInd]][i], axis=(0, 2, 3))
                  for i in range(len(trial_counts))]
    tempScores2 = [np.mean(scoreDict['accuracy'][weightMethods[keyInd]][i], axis=(2, 3))
                   for i in range(len(trial_counts))]
    tempSEM2 = [sem(scoreDict['accuracy'][weightMethods[keyInd]][i], axis=(2))
                for i in range(len(trial_counts))]

    tempSEM2 = [np.mean(tempSEM2[i], axis=(2)) for i in range(len(trial_counts))]

    # each classification method is trials
    meanDataDict['accuracy'][weightMethods[keyInd]] = 100 * np.array(tempScores)
    meanDataDict['stddev'][weightMethods[keyInd]] = 100 * np.array(tempStddev)
    labelDataDict['accuracy'][weightMethods[keyInd]] = 100 * np.array(tempScores2)
    labelDataDict['stddev'][weightMethods[keyInd]] = 100 * np.array(tempSEM2)

    ####################################
    # Load label performances that used standard training (subject-specific)
    # REQUIRES STD ANALYSIS IN pooledBCI_Sens.py HAS BEEN RUN

    # Load data not pickled
    storedMatStd = loadmat(op.join(cache_fname, 'simResults.mat'))
    trial_countsStd = (np.squeeze(storedMatStd['trial_counts'])).tolist()
    #accuracyDif = storedMat['accuracyDif']
    #deltaAccuracy = storedMat['deltaAccuracy']
    snrsStd = np.squeeze(storedMatStd['snrs']).tolist()

    # Load pickled data from pooledBCI_Sens.py results
    with open(op.join(cache_fname, 'stdScoreDict.pkl'), 'rb') as infile:
        [stdDataDict, stdLabelList] = cPickle.load(infile)
        stdLabelNames = [l.name for l in stdLabelList]

        # Find labels from standard evaluation matching the label names here
        # Account for names like G_precentral...-lh vs lh.G_precent...
        labelInds = []
        for label in labelList:
            try:
                labelInds.append(stdLabelNames.index(label.name))
            except:
                print 'Missing label ' + label.name
                try:
                    closeName = label.name[0:-3]
                    ind = [i for i in range(len(stdLabelNames))
                           if closeName in stdLabelNames[i]]

                    if len(ind) == 1:
                        labelInds.append(ind[0])
                        print 'Close label: \'' + stdLabelNames.index[labelInds[0]] + '\' being used'
                except:
                    'Missing label ' + label.name + ' and no close match found'

        assert labelInds > 0, 'No labels found in std performance list.'

        # Average label percentages across subjects/repeats and zip with label names
        # stdPerf is (nBCILabels, nRepeats, nSubjects)
        stdPerf = stdDataDict[trial_countsStd.index(40)][labelInds, snrsStd.index(-10), :, :]
        stdPerfLabels = zip(np.mean(stdPerf, axis=(1, 2)), [stdLabelNames[ind] for ind in labelInds])

###############################################################################
# Plots
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import glob
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

from pooledBCI_plotNSubjectGain import pooledBCI_plotNSubjectGain
from pooledBCI_plotPerfPanel import pooledBCI_plotPerfPanel

plt.close('all')
plt.ion()
plt.rcParams['pdf.fonttype'] = 42

classificationDetails = 'Source ' + clfScheme + ': ' + str(regFactors[0]) + \
    ' nSubjects=' + str(len(subjects)) + ' nLabels= ' + str(len(labelList))

nPlots = len(meanDataDict['accuracy'])

################################################################################
## Figure 0: Absolute performance
if 0 in plotsToGen:
    print 'Plotting Figure 0',
    # Call Figure 0 Function
    #fig0 = pooledBCI_plotNSubjectGain(meanDataDict, poolSizes, trial_counts, weightMethods)

    nPlots = len(trial_counts)
    difMin, difMax = -10., 10.
    ftsize = 14
    figSize = (7 * nPlots, 7)
    plotMarker = ['o', 'D', 's', '^']
    plotColors = ['DodgerBlue', 'ForestGreen', 'Maroon', 'DarkMagenta']

    fig0, axList = plt.subplots(ncols=nPlots, figsize=figSize)
    fig0.tight_layout(h_pad=.5)
    fig0.subplots_adjust(left=0.04, right=0.975, bottom=0.1, top=0.8)

    '''
    axesPad = .5
    gridAx = ImageGrid(fig0, 111, nrows_ncols=(1, nPlots), axes_pad=axesPad,
                       share_all=True, label_mode="L", direction='row', add_all=True)
    '''

    for i in range(len(trial_counts)):
        for j, method in enumerate(weightMethods):
            axList[i].plot(poolSizes, meanDataDict['accuracy'][method][i, :], c=plotColors[j], linewidth=3,
                           markersize=11, markeredgewidth=1, alpha=.8, label=weightMethods[j])

        if i == 0:
            axList[i].set_ylabel('Accuracy (%)', fontsize=ftsize + 3)
            #axList[i].text((poolSizes[-1] + poolSizes[-2])/2, 51, 'Chance', va='bottom', color='red', fontsize=ftsize + 4,
            #               ha='center')
            axList[i].legend(loc='upper left', fancybox=True, shadow=True,
                             borderaxespad=1)

        #axList[i].axhline(y=50, color='r', linewidth=4, ls='--', alpha=0.8)
        axList[i].set_title(str(trial_counts[i]) + ' Trials/Subject',
                            fontsize=ftsize + 5)
        axList[i].set_xlabel('Pool Size (subjects)', fontsize=ftsize + 3)
        axList[i].set_xticks(poolSizes)
        axList[i].set_yticks(range(65, 90, 5))
        axList[i].xaxis.set_ticks_position('bottom')
        axList[i].grid()

    #gridAx.axes_llc.set_xticks(poolSizes)
    #gridAx.axes_llc.set_xticklabels(poolSizes, fontsize=ftsize)

    plt.suptitle('Accuracy for Different Training Pool Sizes\nSNR: ' +
                 str(snr), fontsize=ftsize + 10)

    print ' ... Done'

###############################################################################
## Figure 1: BCI area performance as a function of changing pool size
if 1 in plotsToGen:
    print 'Plotting Figure Set 1',
    mpl.pyplot.autoscale(enable=False)
    fig1List = []
    surf = 'smoothwm'
    surf = 'inflated_pre'

    kdeData = labelDataDict['accuracy']['kde'][0]
    kdeStddev = labelDataDict['stddev']['kde'][0]

    ### P300 ROIs, Dodger Blue and Lime Green
    if len(labelNames_p300) > 0:
        colors = ['#84E184', '#196619', '#1565B2', '#4BA6FF', '#061D33',
                  '#BCDEFF']
        views = ['l', 'd', 'p']
        legend = [['P300a (2)', '#32CD32'], ['P300b (4)', '#1E90FF']]

        axRange = np.arange(75., 90., 5.)
        axRange = np.insert(axRange, 0, 72.5)
        axRange = np.append(axRange, 87.5)

        # Find standard (subject specific) test results for each label
        stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                   for l in labelNames_p300]
        stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]

        # Find weighted classification (pooled training) test results
        labelInds = [[label.name for label in labelList].index(l)
                     for l in labelNames_p300]
        poolPerfs = kdeData[labelInds]

        # Call plotting function with SEM
        fig1List.append(pooledBCI_plotPerfPanel(labelList=[labelList[i] for i in labelInds],
                                                poolSizes=np.array(poolSizes),
                                                labelPerfs=poolPerfs, stdPerfs_y=stdPerfs_y,
                                                stddev=kdeStddev, legend=legend, views=views,
                                                colors=colors, axRange=axRange, surface=surf))

    ### Motor ROIs
    if len(labelNames_motor) > 0:
        colors = ['#1B82E6', '#70DC80', '#1E7B1E', '#0A290A', '#C2F0C2']
        closeupView = dict(azimuth=-178, elevation=31, distance=100,
                           focalpoint=[-22, -23, 21])
        views = ['l', 'd', closeupView]
        axRange = np.arange(70, 90, 5)
        #axRange = np.insert(axRange, -1, 87.5)
        legend = [['Hand Motor (3)', '#32CD32'], ['Premotor (1)', '#1E90FF']]

        # Find standard (subject specific) test results for each label
        stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                   for l in labelNames_motor]
        stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]

        # Find labels in classification results
        labelInds = [[label.name for label in labelList].index(l)
                     for l in labelNames_motor]
        poolPerfs = kdeData[labelInds]

        # Call plotting function with SEM
        fig1List.append(pooledBCI_plotPerfPanel(labelList=[labelList[i] for i in labelInds],
                                                poolSizes=np.array(poolSizes),
                                                labelPerfs=poolPerfs, stdPerfs_y=stdPerfs_y,
                                                stddev=kdeStddev, legend=legend, views=views,
                                                colors=colors, axRange=axRange, surface=surf))
        # Create inset
        tempAx = fig1List[-1].axes[0]
        extent = tempAx.axis()
        # Create inset axes and set properties
        axins = zoomed_inset_axes(tempAx, zoom=2.75, loc=4)
        axins.imshow(tempAx.get_images()[0].get_array(), extent=extent,
                     interpolation='nearest', origin='upper')

        axins.set_xlim(270, 375)  # Set extent of inset
        axins.set_ylim(1015, 912)
        axCol = '#FFFF33'  # Set inset spine/line color
        axins.xaxis.set_tick_params(tick1On=False, tick2On=False,
                                    label1On=False)
        axins.yaxis.set_tick_params(tick1On=False, tick2On=False,
                                    label1On=False)
        # Set properties of spines on inset axes
        [sp.set_linewidth(2.0) for sp in axins.spines.itervalues()]
        [sp.set_color(axCol) for sp in axins.spines.itervalues()]

        # Create inset box/lines
        mark_inset(tempAx, axins, loc1=1, loc2=3, fc='none', ec=axCol, lw=2)

    ### SSVEP/Auditory ROIs
    if len(labelNames_ssvep_auditory) > 0:
        colors = ['#1E90FF', '#BCDEFF', '#0C3A66', '#84E184']
        #colors = ['#B2B2FF', '#7A0000', '#84E184', '#3333FF']
        legend = [['Auditory (1)', '#32CD32'], ['SSVEP (3)', '#1E90FF']]
        views = ['l', 'm', 'c']
        axRange = np.arange(60., 90., 5.)

        # Find standard (subject specific) test results for each label
        stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                   for l in labelNames_ssvep_auditory]
        stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]

        # Find labels in classification results
        labelInds = [[label.name for label in labelList].index(l)
                     for l in labelNames_ssvep_auditory]
        poolPerfs = kdeData[labelInds]

        # Call plotting function with SEM
        fig1List.append(pooledBCI_plotPerfPanel(labelList=[labelList[i] for i in labelInds],
                                                poolSizes=np.array(poolSizes),
                                                labelPerfs=poolPerfs, stdPerfs_y=stdPerfs_y,
                                                stddev=kdeStddev, legend=legend, views=views,
                                                colors=colors, axRange=axRange, surface=surf))

    '''
    ### Auditory Space/Pitch Attention Switch ROIs
    if len(labelNames_space_pitch) > 0:
        colors = ['#1565B2', '#84E184']
        legend = [['Pitch', '#1E90FF'], ['Space', '#84E184']]
        leftLat = dict(azimuth=0, elevation=-90, distance=100,
                        focalpoint=[0, -20, 0])
        views = [leftLat, 'l']
        axRange = np.arange(60., 90., 5.)

        # Find standard (subject specific) test results for each label
        stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                   for l in labelNames_space_pitch]
        stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]

        # Find labels in classification results
        labelInds = [[label.name for label in labelList].index(l)
                     for l in labelNames_space_pitch]
        poolPerfs = kdeData[labelInds]

        # Call plotting function
        fig1List.append(pooledBCI_plotPerfPanel(labelList=[labelList[i] for i in labelInds],
                                                poolSizes=np.array(poolSizes),
                                                labelPerfs=poolPerfs, stdPerfs_y=stdPerfs_y,
                                                legend=legend, views=views, colors=colors,
                                                axRange=axRange, surface=surf))
    '''
    '''
    ### Combined ROI classes
    colors = ['#1565B2', '#84E184']
    legend = [['Pitch', '#1E90FF'], ['Space', '#84E184']]
    views = ['l', 'm', 'd']
    axRange = np.arange(70., 90., 5.)

    # hemisphere tag differs between lh.... and ...-lh
    labelNames_space_pitch_format2 = ['UDRon-UDStd_01-lh', 'LRRon-LRStd_01-rh']
    # Find standard (subject specific) test results for each label
    stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                for l in labelNames_space_pitch_format2]
    stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]

    # Find labels in classification results
    labelInds = [[label.name for label in labelList].index(l) for l in labelNames_space_pitch_format2]
    poolPerfs = kdeData[labelInds]

    # Call plotting function
    fig1List.append(pooledBCI_plotPerfPanel(labelList=[labelList[i] for i in labelInds],
                                            poolSizes=np.array(poolSizes),
                                            labelPerfs=poolPerfs, stdPerfs_y=stdPerfs_y,
                                            legend=legend, views=views, colors=colors,
                                            axRange=axRange, surface=surf))
                                            '''
    print ' ... Done'

###############################################################################
## Figure 2: BCI area performance with changing pool size
## This section groups labels by area before plotting (i.e., all P300 plotted
    # together
# This plotting section experimental and not used in manuscript

if 2 in plotsToGen:
    print 'Plotting Figure Set 2',
    mpl.pyplot.autoscale(enable=False)
    surf = 'inflated_pre'

    colors = ['#1E90FF', '#84E184', '#1E7B1E']
    views = ['lateral', 'dorsal', 'medial']

    # Set y-axis range
    axRange = np.arange(65., 90., 5.)

    # Pull data from master dictionary
    kdeData = labelDataDict['accuracy']['kde'][0]

    # Initialize lists that will be passed to the plotting function
    labelConglomList = []
    stdAvgPerfs = []
    poolAvgPerfs = []
    legend = []

    # Colors: HTML Dodger Blue and Lime Green
    if len(labelNames_p300) > 0:
        # Create conglomerate label for P300
        p300_labels = [l for l in labelList if l.name in labelNames_p300]
        p300_conglom = p300_labels[0]
        for label in p300_labels[1:]:
            p300_conglom += label
        labelConglomList.append(p300_conglom)

        # Find standard (subject specific) test results for each label
        stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                   for l in labelNames_p300]
        stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]
        stdAvgPerfs.append(sum(stdPerfs_y) / len(stdPerfs_y))

        # Find average transfer learning test results for each label
        labelInds = [[label.name for label in labelList].index(l)
                     for l in labelNames_p300]
        poolPerfs = kdeData[labelInds]
        poolAvgPerfs.append(sum(poolPerfs) / len(poolPerfs))
        legend.append(['P300', '#1E90FF'])

    ### Motor ROIs
    if len(labelNames_motor) > 0:
        # Create conglomerate label for motor
        motor_labels = [l for l in labelList if l.name in labelNames_motor]
        motor_conglom = motor_labels[0]
        for label in motor_labels[1:]:
            motor_conglom += label
        labelConglomList.append(motor_conglom)

        # Find standard (subject specific) test results for each label
        stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                   for l in labelNames_motor]
        stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]
        stdAvgPerfs.append(sum(stdPerfs_y) / len(stdPerfs_y))

        # Find average transfer learning test results for each label
        labelInds = [[label.name for label in labelList].index(l)
                     for l in labelNames_motor]
        poolPerfs = kdeData[labelInds]
        poolAvgPerfs.append(sum(poolPerfs) / len(poolPerfs))
        legend.append(['Hand- and pre-motor', '#84E184'])

    ### SSVEP/Auditory ROIs
    if len(labelNames_ssvep_auditory) > 0:
        # Create conglomerate label for SSVEP and auditory BCIs
        ssvep_auditory_labels = [l for l in labelList
                                 if l.name in labelNames_ssvep_auditory]
        ssvep_auditory_conglom = ssvep_auditory_labels[0]
        for label in ssvep_auditory_labels[1:]:
            ssvep_auditory_conglom += label
        labelConglomList.append(ssvep_auditory_conglom)

        # Find standard (subject specific) test results for each label
        stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                   for l in labelNames_ssvep_auditory]
        stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]
        stdAvgPerfs.append(sum(stdPerfs_y) / len(stdPerfs_y))

        # Find average transfer learning test results for each label
        labelInds = [[label.name for label in labelList].index(l)
                     for l in labelNames_ssvep_auditory]
        poolPerfs = kdeData[labelInds]
        poolAvgPerfs.append(sum(poolPerfs) / len(poolPerfs))
        legend.append(['SSVEP, Auditory', '#1E7B1E'])

    ### Auditory Space/Pitch Attention Switch ROIs
    # (For post manuscript continuation)
    if len(labelNames_space_pitch) > 0:

        # (Hemisphere tag differs between lh.... and ...-lh)
        labelNames_space_pitch_format2 = ['UDRon-UDStd_01-lh', 'LRRon-LRStd_01-rh']

        # Create conglomerate label for space/pitch switch task
        space_pitch_labels = [l for l in labelList
                              if l.name in labelNames_space_pitch_format2]
        space_pitch_conglom = space_pitch_labels[0]
        for label in space_pitch_labels[1:]:
            space_pitch_conglom += label
        labelConglomList.append(space_pitch_conglom)

        # Find standard (subject specific) test results for each label
        stdInds = [[zipLabel for zipPerf, zipLabel in stdPerfLabels].index(l)
                   for l in labelNames_space_pitch_format2]
        stdPerfs_y = [stdPerfLabels[i][0] * 100 for i in stdInds]
        stdAvgPerfs.append(sum(stdPerfs_y) / len(stdPerfs_y))

        # Find average transfer learning test results for each label
        labelInds = [[label.name for label in labelList].index(l)
                     for l in labelNames_space_pitch]
        poolPerfs = kdeData[labelInds]
        poolAvgPerfs.append(sum(poolPerfs) / len(poolPerfs))
        legend.append(['Space/Pitch Switch', '#196619'])

    # Call plotting function
    fig2 = pooledBCI_plotPerfPanel(labelList=labelConglomList,
                                   poolSizes=np.array(poolSizes),
                                   labelPerfs=np.array(poolAvgPerfs),
                                   stdPerfs_y=stdAvgPerfs, legend=legend,
                                   views=views, colors=colors,
                                   axRange=axRange, surface=surf)

    print ' ... Done'

plt.draw()
plt.show()

###############################################################################
### Save plots
if savePlots:
    if 0 in plotsToGen:
        # Find file in series
        fileNames = [f.split('/')[-1] for f in glob.glob(saveDir + '/poolSizeEffects_10-40Trials*')]
        fileNames = [f.split('.')[0] for f in fileNames]
        fileVersions = [int(f.split('_')[-1][1:]) for f in fileNames]
        if len(fileVersions) > 0:
            version = '%02d' % (max(fileVersions) + 1)
        else:
            version = '01'

        # save Plot
        plot_fname = op.join(saveDir, 'poolSizeEffects_10-40Trials_v' + version + '.pdf')
        fig0.savefig(plot_fname)
    if 1 in plotsToGen:

        # save Plots
        plot_fname_p300 = op.join(saveDir, 'poolSizeEffects_p300' + '.pdf')
        plot_fname_motor = op.join(saveDir, 'poolSizeEffects_motor' + '.pdf')
        plot_fname_audVis = op.join(saveDir, 'poolSizeEffects_audVis' + '.pdf')
        #plot_fname_spacePitch = op.join(saveDir, 'poolSizeEffects_spacePitch' + '.pdf')

        fig1List[0].savefig(plot_fname_p300)
        fig1List[1].savefig(plot_fname_motor)
        fig1List[2].savefig(plot_fname_audVis)
        #fig1List[3].savefig(plot_fname_spacePitch)
