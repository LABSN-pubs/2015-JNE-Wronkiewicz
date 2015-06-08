'''
gen_BCILabels.py

Generate and plot BCI relevant labels (P300, SSVEP) to be used in displaying
how number of subjects in subject pool changes gains to be expected

@Author: wronk
'''

#dpylint: disable-msg=C0103

import os
import os.path as op
from surfer import Brain
import mne
import numpy as np
from scipy.spatial.distance import cdist

doPlot = True
# whether or not to morph hand motor labels to all other subjects
doMorph = True

subjectDir = os.environ['SUBJECTS_DIR']
modelSubj = 'fsaverage'
hemi = 'lh'
surface = 'inflated'
views = ['lat']

savePath = '/media/Toshiba/Code/AnatomBCI_Mark/custom_labels'

labelRadii = np.arange(0, 25, 5)[1:]

subjectSet = []
subjectSet.extend(['RON006_AKCLEE', 'RON007_AKCLEE', 'RON008_AKCLEE',
                   'RON010_AKCLEE', 'RON011_AKCLEE', 'RON014_AKCLEE',
                   'RON016_AKCLEE', 'RON021_AKCLEE'])

subjectSet.extend(['AKCLEE_101', 'AKCLEE_103', 'AKCLEE_104', 'AKCLEE_105',
                   'AKCLEE_106', 'AKCLEE_107', 'AKCLEE_109', 'AKCLEE_110',
                   'AKCLEE_113', 'AKCLEE_114', 'AKCLEE_115', 'AKCLEE_118',
                   'AKCLEE_119', 'AKCLEE_120', 'AKCLEE_121', 'AKCLEE_122',
                   'AKCLEE_123', 'AKCLEE_124', 'AKCLEE_125', 'AKCLEE_126',
                   'AKCLEE_127'])

# Load list of labels and pull out pre-central gyrus
labelList, _ = mne.labels_from_parc(subject=modelSubj, parc='aparc.a2009s')
primeMotor = [l for l in labelList if l.name == 'G_precentral-' + hemi][0]
primeMotorMNI = mne.vertex_to_mni(primeMotor.vertices, 0, modelSubj)

# Find closest point in fsaverage brain space according to
# Witt 2008, Functional neuroimaging correlates of finger-tapping task
# variations coords in Talairach space
leftPrecentral = np.atleast_2d(np.array([-38, -26, 50]))
rightPrecentral = np.atleast_2d(np.array([36, -22, 54]))
SMA = [-4, -8, 52]

dists = np.squeeze(cdist(leftPrecentral, primeMotorMNI, 'euclidean'))
min_dist, ind = min((min_dist, ind) for (ind, min_dist) in enumerate(dists))

#temp fix until we know seed vertex
#seed =
seed = [primeMotor.vertices[ind]]
seed = seed * len(labelRadii)

# Generate circular labels
handMotorLabels = mne.grow_labels(modelSubj, seed, labelRadii, 0, n_jobs=6)

# find intersection circular labels with pre-central sulcus label
overlapInds = [np.in1d(np.array(labels.vertices), np.array(primeMotor.vertices))
               for labels in handMotorLabels]

# List to hold all labels
labels_motor = []
labels_motor.extend(l for l in labelList if l.name == 'S_precentral-sup-part' +
                    '-' + hemi)
for i, label in enumerate(handMotorLabels):

    label.vertices = label.vertices[overlapInds[i]]
    label.pos = label.pos[overlapInds[i]]
    label.values = label.values[overlapInds[i]]

    label.subject = modelSubj
    label.name = 'G_precentral_handMotor_radius_' + str(labelRadii[i]) + 'mm'
    labels_motor.append(label)

    subjectFolder_fname = op.join(savePath, 'HandMotor', label.subject)
    if not op.exists(subjectFolder_fname):
        os.makedirs(subjectFolder_fname)

    labelSavePath = op.join(subjectFolder_fname, label.name + '-' + hemi +
                            '.label')
    mne.write_label(labelSavePath, label)

    if doMorph:
        for subject in subjectSet:
            morphedLabel = label.morph(subject_from=modelSubj,
                                       subject_to=subject,
                                       smooth=5, n_jobs=6)
            labelSavePath = op.join(subjectDir, subject, 'label',
                                    morphedLabel.name + '-' + hemi + '.label')
            morphedLabel.save(labelSavePath)

### Add other BCI labels
# P300 Frontal: ACC, middle frontal gyrus, inferior frontal sulcus
    # G_and_S_cingul-Ant
        # No exact match, try dACC, dACC_fsaverage, G_cingulate-Main_part,
        # G_cingulate-Isthmus, rostralanteriorcingulate,
    #15 G_frontal_middle
    #52 S_frontal_inferior
# P300 Parietal: TPJ, Superior parietal lobule, intraparietal sulcus, angular gyrus
    #26 G_parietal_inferior-Supramarginal_part
    #27 G_parietal_superior
    #25 G_parietal_inferior-Angular_part
    #56 S_intraparietal-and_Parietal_transverse
# V1: calcarine Sulcus area and extrastriate cortex
    #42 Pole_occipital
    #11 G_cuneus
    #44 S_calcarine
    #58 S_oc_sup_and_transversal
    #20 G_occipital_sup

labels_p300 = []
labels_ssvep = []
labels_auditory = []

p300 = ['G_front_middle',
        'S_front_inf',
        'G_pariet_inf-Supramar',
        'G_parietal_sup',
        'G_pariet_inf-Angular',
        'S_intrapariet_and_P_trans']
ssvep = ['Pole_occipital',
         'S_calcarine',
         'G_cuneus',
         'S_oc_sup_and_transversal',
         'G_occipital_sup']
auditory = ['G_temp_sup-G_T_transv']
#            'G_temp_sup-Lateral',
#            'G_temp_sup-Plan_tempo',
#            'S_temporal_transverse']

# construct filenames to each label
for name in p300:
    try:
        labels_p300.extend([l for l in labelList if l.name == name + '-' + hemi])
    except:
        label_fname = op.join(subjectDir, modelSubj, 'label', hemi + '.' + name + '.label')
        labels_p300.append(mne.read_label(label_fname, subject=modelSubj))
for name in ssvep:
    try:
        labels_ssvep.extend(l for l in labelList if l.name == name + '-' + hemi)
    except:
        label_fname = op.join(subjectDir, modelSubj, 'label', hemi + '.' + name + '.label')
        labels_ssvep.append(mne.read_label(label_fname, subject=modelSubj))
for name in auditory:
    try:
        labels_auditory.extend(l for l in labelList if l.name == name + '-' + hemi)
    except:
        label_fname = op.join(subjectDir, modelSubj, 'label', hemi + '.' + name + '.label')
        labels_auditory.append(mne.read_label(label_fname, subject=modelSubj))

###############################################################################
# Plot
shades_yellow = ['FFEB80', '806C00', 'E6C200', 'FFDB19', '906C00', 'FFEF99']
shades_red = ['4C0000', 'FF4D4D', 'FF0000', 'FF8080', 'FFCCCC']
shades_purple = ['260026', '5A005A', 'A64DA6', 'D9B2D9']
alpha_motor = [.7]
alpha_motor.extend([.2] * (len(labels_motor) - 1))

if(doPlot):
    brain = Brain(modelSubj, hemi=hemi, surf='white')
    brain.toggle_toolbars = True

    # Add hand motor area information
    brain.add_label('G_precentral', color='limegreen', alpha=0.7)
    brain.add_foci(seed[0], coords_as_verts=True, scale_factor=.3, color='red')

    for h, label in enumerate(labels_motor):
        brain.add_label(label=label, color='blue', alpha=alpha_motor[h])
    for i, label in enumerate(labels_p300):
        brain.add_label(label=label, color='#' + shades_yellow[i], alpha=0.7)
    for j, label in enumerate(labels_ssvep):
        brain.add_label(label=label, color='#' + shades_red[j], alpha=0.7)
    for k, label in enumerate(labels_auditory):
        brain.add_label(label=label, color='#' + shades_purple[k], alpha=0.7)
