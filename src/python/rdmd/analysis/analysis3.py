from datetime import datetime
from matplotlib.mlab import normpdf
import numpy as np
import pandas as pd
import statsmodels.api as sm
from psychopy.data import FitWeibull
from scipy.stats import wilcoxon, norm
from sklearn.linear_model import LinearRegression
from rdmd.analysis.subject_info import conditions, read_subjects, exclude_subjects
from rdmd.utils import FitSigmoid, FitRT, get_twod_confidence_interval
import matplotlib.pyplot as plt

colors={
    'ShamPreAnode': 'b',
    'Anode': 'r',
    'ShamPreCathode': 'b',
    'Cathode': 'g'
}

def analyze_subject_accuracy_rt(subject, plot=False):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    for condition,trial_data in subject.session_data.iteritems():
        condition_coherence_accuracy[condition]={}
        condition_coherence_rt[condition]={}
        # For each trial
        for trial_idx in range(trial_data.shape[0]):
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]
            correct=trial_data[trial_idx,3]
            rt=trial_data[trial_idx,6]

            if not coherence in condition_coherence_accuracy[condition]:
                condition_coherence_accuracy[condition][coherence]=[]
            condition_coherence_accuracy[condition][np.abs(coherence)].append(float(correct))

            if not coherence in condition_coherence_rt[condition]:
                condition_coherence_rt[condition][coherence]=[]
            condition_coherence_rt[condition][np.abs(coherence)].append(rt)

        coherences = sorted(condition_coherence_accuracy[condition].keys())
        accuracy=[]
        for coherence in coherences:
            accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
        acc_fit = FitSigmoid(coherences, accuracy, guess=[0.0, 0.2], display=0)
        condition_accuracy_thresh[condition]=acc_fit.inverse(0.8)

    for stim_condition in ['Anode', 'Cathode']:
        condition_coherence_rt_diff[stim_condition]={}
        coherences=sorted(condition_coherence_rt[stim_condition].keys())
        for coherence in coherences:
            condition_coherence_rt_diff[stim_condition][coherence]=np.mean(condition_coherence_rt[stim_condition][coherence])-np.mean(condition_coherence_rt['ShamPre%s' % stim_condition][coherence])
    condition_coherence_rt_diff['Sham']={}
    coherences=sorted(condition_coherence_rt['ShamPreAnode'].keys())
    for coherence in coherences:
        condition_coherence_rt_diff['Sham'][coherence]=np.mean(condition_coherence_rt['ShamPreAnode'][coherence])-np.mean(condition_coherence_rt['ShamPreCathode'][coherence])

    if plot:
        plot_choice_accuracy(colors, condition_coherence_accuracy)

        plot_choice_rt(colors, condition_coherence_rt)

        plot_choice_rt_diff(colors, condition_coherence_rt, plot_err=False)

    return condition_coherence_accuracy, condition_coherence_rt, condition_coherence_rt_diff, condition_accuracy_thresh


def analyze_subject_choice_hysteresis(subject, plot=False, itis='all'):
    condition_coherence_choices={
        'L*': {},
        'R*': {}
    }
    condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    condition_logistic_params={
        'a1': {},
        'a2': {}
    }
    for condition,trial_data in subject.session_data.iteritems():
        # Dict of coherence levels
        condition_coherence_choices['L*'][condition]={}
        condition_coherence_choices['R*'][condition]={}

        # Mean iti
        median_iti=np.median(trial_data[:,7])
        trials_to_use=range(trial_data.shape[0])
        if itis=='low':
            trials_to_use=np.where(trial_data[:,7]<=median_iti)[0]
        elif itis=='high':
            trials_to_use=np.where(trial_data[:,7]>median_iti)[0]

        # For each trial
        for trial_idx in trials_to_use:
            # Get coherence - negative coherences when direction is to the left
            coherence=trial_data[trial_idx,2]*trial_data[trial_idx,1]
            last_resp=trial_data[trial_idx,5]
            resp=trial_data[trial_idx,4]

            if last_resp<0:
                if not coherence in condition_coherence_choices['L*'][condition]:
                    condition_coherence_choices['L*'][condition][coherence]=[]
                    # Append 0 to list if left (-1) or 1 if right
                condition_coherence_choices['L*'][condition][coherence].append(np.max([0,resp]))
            elif last_resp>0:
                # List of rightward choices (0=left, 1=right)
                if not coherence in condition_coherence_choices['R*'][condition]:
                    condition_coherence_choices['R*'][condition][coherence]=[]
                    # Append 0 to list if left (-1) or 1 if right
                condition_coherence_choices['R*'][condition][coherence].append(np.max([0,resp]))

        choice_probs=[]
        full_coherences=[]
        for coherence in condition_coherence_choices['L*'][condition]:
            choice_probs.append(np.mean(condition_coherence_choices['L*'][condition][coherence]))
            full_coherences.append(coherence)
        acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
        condition_sigmoid_offsets['L*'][condition]=acc_fit.inverse(0.5)

        choice_probs=[]
        full_coherences=[]
        for coherence in condition_coherence_choices['R*'][condition]:
            choice_probs.append(np.mean(condition_coherence_choices['R*'][condition][coherence]))
            full_coherences.append(coherence)
        acc_fit=FitSigmoid(full_coherences, choice_probs, guess=[0.0, 0.2], display=0)
        condition_sigmoid_offsets['R*'][condition]=acc_fit.inverse(0.5)

        data=pd.DataFrame({
            'resp': np.clip(trial_data[1:,4],0,1),
            'coh': trial_data[1:,2]*trial_data[1:,1],
            'last_resp': trial_data[1:,5]
        })
        data['intercept']=1.0

        logit = sm.Logit(data['resp'], data[['coh','last_resp','intercept']])
        result = logit.fit(disp=False)
        condition_logistic_params['a1'][condition]=result.params['coh']
        condition_logistic_params['a2'][condition]=result.params['last_resp']

    if plot:
        plot_choice_probability(colors, condition_coherence_choices)

    return condition_coherence_choices, condition_sigmoid_offsets, condition_logistic_params



def analyze_accuracy_rt(subjects, plot=True, print_stats=True):
    condition_coherence_accuracy={}
    condition_coherence_rt={}
    condition_coherence_rt_diff={}
    condition_accuracy_thresh={}
    # For each subject
    for subj_id in subjects:
        subject=subjects[subj_id]

        subj_condition_coherence_accuracy, subj_condition_coherence_rt, subj_condition_coherence_rt_diff,subj_condition_accuracy_thresh=analyze_subject_accuracy_rt(subject, plot=False)

        for condition in conditions:
            if not condition in condition_coherence_accuracy:
                condition_coherence_accuracy[condition]={}
                condition_coherence_rt[condition]={}
                condition_accuracy_thresh[condition]=[]
            condition_accuracy_thresh[condition].append(subj_condition_accuracy_thresh[condition])

            for coherence in subj_condition_coherence_accuracy[condition]:
                if not coherence in condition_coherence_accuracy[condition]:
                    condition_coherence_accuracy[condition][coherence]=[]
                condition_coherence_accuracy[condition][coherence].append(np.mean(subj_condition_coherence_accuracy[condition][coherence]))

            for coherence in subj_condition_coherence_rt[condition]:
                if not coherence in condition_coherence_rt[condition]:
                    condition_coherence_rt[condition][coherence]=[]
                condition_coherence_rt[condition][coherence].append(np.mean(subj_condition_coherence_rt[condition][coherence]))

        for condition in subj_condition_coherence_rt_diff:
            if not condition in condition_coherence_rt_diff:
                condition_coherence_rt_diff[condition]={}
            for coherence in subj_condition_coherence_rt_diff[condition]:
                if not coherence in condition_coherence_rt_diff[condition]:
                    condition_coherence_rt_diff[condition][coherence]=[]
                condition_coherence_rt_diff[condition][coherence].append(subj_condition_coherence_rt_diff[condition][coherence])

    if plot:
        plot_choice_accuracy(colors, condition_coherence_accuracy, plot_err=True)

        plot_choice_rt(colors, condition_coherence_rt)

        plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=True)

        plot_accuracy_thresh(colors, condition_accuracy_thresh)

    thresh_results={
        'sham': {},
        'cathode': {},
        'anode': {},
    }
    (thresh_results['sham']['x'],thresh_results['sham']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreCathode'],condition_accuracy_thresh['ShamPreAnode'])
    (thresh_results['cathode']['x'],thresh_results['cathode']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreCathode'],condition_accuracy_thresh['Cathode'])
    (thresh_results['anode']['x'],thresh_results['anode']['p'])=wilcoxon(condition_accuracy_thresh['ShamPreAnode'],condition_accuracy_thresh['Anode'])

    rtdiff_results={
        'sham': {'coh': {}, 'intercept':{}},
        'anode': {'coh': {}, 'intercept':{}},
        'cathode': {'coh': {}, 'intercept':{}}
    }
    for condition in ['Sham','Anode','Cathode']:
        coh=[]
        rt_diff=[]
        for coherence in condition_coherence_rt_diff[condition]:
            for diff in condition_coherence_rt_diff[condition][coherence]:
                coh.append(coherence)
                rt_diff.append(diff)
        data=pd.DataFrame({
            'coh': coh,
            'rt_diff': rt_diff
        })
        data['intercept']=1.0
        lr = sm.GLM(data['rt_diff'], data[['coh','intercept']])
        result = lr.fit()
        for param in ['coh','intercept']:
            rtdiff_results[condition.lower()][param]['x']=result.params[param]
            rtdiff_results[condition.lower()][param]['t']=result.tvalues[param]
            rtdiff_results[condition.lower()][param]['p']=result.pvalues[param]

    if print_stats:
        print('Accuracy Threshold')
        for condition, results in thresh_results.iteritems():
            print('%s: x=%.4f, p=%.4f' % (condition, results['x'],results['p']))

        print('')
        print('RT Diff')
        for condition, results in rtdiff_results.iteritems():
            print('%s, coherence: x=%.4f, t=%.4f, p=%.4f, intercept: x=%.4f, t=%.4f, p=%.4f' % (condition,
                                                                                                results['coh']['x'],
                                                                                                results['coh']['t'],
                                                                                                results['coh']['p'],
                                                                                                results['intercept']['x'],
                                                                                                results['intercept']['t'],
                                                                                                results['intercept']['p']))

    return thresh_results, rtdiff_results


def analyze_choice_hysteresis(subjects, itis='all', plot=True, print_stats=True):
    condition_coherence_choices={
        'L*': {},
        'R*': {}
    }
    condition_sigmoid_offsets={
        'L*': {},
        'R*': {}
    }
    condition_logistic_params={
        'a1': {},
        'a2': {}
    }

    # For each subject
    for subj_id in subjects:
        subject=subjects[subj_id]
        subj_condition_coherence_choices, subj_condition_sigmoid_offsets, subj_condition_logistic_params=analyze_subject_choice_hysteresis(subject, plot=False, itis=itis)

        for condition in conditions:
            if not condition in condition_coherence_choices['L*']:
                condition_coherence_choices['L*'][condition]={}
                condition_coherence_choices['R*'][condition]={}
                condition_sigmoid_offsets['L*'][condition]=[]
                condition_sigmoid_offsets['R*'][condition]=[]
                condition_logistic_params['a1'][condition]=[]
                condition_logistic_params['a2'][condition]=[]

            condition_sigmoid_offsets['L*'][condition].append(subj_condition_sigmoid_offsets['L*'][condition])
            condition_sigmoid_offsets['R*'][condition].append(subj_condition_sigmoid_offsets['R*'][condition])

            condition_logistic_params['a1'][condition].append(subj_condition_logistic_params['a1'][condition])
            condition_logistic_params['a2'][condition].append(subj_condition_logistic_params['a2'][condition])

            for coherence in subj_condition_coherence_choices['L*'][condition]:
                if not coherence in condition_coherence_choices['L*'][condition]:
                    condition_coherence_choices['L*'][condition][coherence]=[]
                condition_coherence_choices['L*'][condition][coherence].append(np.mean(subj_condition_coherence_choices['L*'][condition][coherence]))

            for coherence in subj_condition_coherence_choices['R*'][condition]:
                if not coherence in condition_coherence_choices['R*'][condition]:
                    condition_coherence_choices['R*'][condition][coherence]=[]
                condition_coherence_choices['R*'][condition][coherence].append(np.mean(subj_condition_coherence_choices['R*'][condition][coherence]))

    if plot:
        plot_indifference(colors, condition_sigmoid_offsets)

        plot_choice_probability(colors, condition_coherence_choices)

        plot_logistic_parameter_ratio(colors, condition_logistic_params)

    indec_results={
        'sham': {},
        'anode': {},
        'cathode': {},
    }
    (indec_results['sham']['x'],indec_results['sham']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['ShamPreCathode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreCathode']),
        np.array(condition_sigmoid_offsets['L*']['ShamPreAnode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreAnode']))
    (indec_results['cathode']['x'],indec_results['cathode']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['ShamPreCathode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreCathode']),
        np.array(condition_sigmoid_offsets['L*']['Cathode'])-np.array(condition_sigmoid_offsets['R*']['Cathode']))
    (indec_results['anode']['x'],indec_results['anode']['p'])=wilcoxon(np.array(condition_sigmoid_offsets['L*']['ShamPreAnode'])-np.array(condition_sigmoid_offsets['R*']['ShamPreAnode']),
        np.array(condition_sigmoid_offsets['L*']['Anode'])-np.array(condition_sigmoid_offsets['R*']['Anode']))

    log_results={
        'sham': {},
        'anode': {},
        'cathode': {},
    }
    sham_anode_ratio=np.array(condition_logistic_params['a2']['ShamPreAnode'])/np.array(condition_logistic_params['a1']['ShamPreAnode'])
    sham_cathode_ratio=np.array(condition_logistic_params['a2']['ShamPreCathode'])/np.array(condition_logistic_params['a1']['ShamPreCathode'])
    (log_results['sham']['x'],log_results['sham']['p'])=wilcoxon(sham_anode_ratio, sham_cathode_ratio)

    anode_ratio=np.array(condition_logistic_params['a2']['Anode'])/np.array(condition_logistic_params['a1']['Anode'])
    (log_results['anode']['x'],log_results['anode']['p'])=wilcoxon(sham_anode_ratio, anode_ratio)

    cathode_ratio=np.array(condition_logistic_params['a2']['Cathode'])/np.array(condition_logistic_params['a1']['Cathode'])
    (log_results['cathode']['x'],log_results['cathode']['p'])=wilcoxon(sham_cathode_ratio, cathode_ratio)

    if print_stats:
        print('')
        print('Indecision Points')
        for condition, results in indec_results.iteritems():
            print('%s, x=%.4f, p=%.6f' % (condition,results['x'],results['p']))

        print('')
        print('Logistic Regression')
        print('ShamPreAnode, median=%.4f' % np.median(sham_anode_ratio))
        print('Anode, median=%.4f' % np.median(anode_ratio))
        print('ShamPreCathode, median=%.4f' % np.median(sham_cathode_ratio))
        print('Cathode, median=%.4f' % np.median(cathode_ratio))
        for condition, results in log_results.iteritems():
            print('%s, x=%.4f, p=%.4f' % (condition, results['x'], results['p']))

    return indec_results, log_results


def plot_indifference(colors, condition_sigmoid_offsets):
    fig = plt.figure()
    limits = [-.2, .2]
    ax = fig.add_subplot(1, 1, 1, aspect='equal',adjustable='box-forced')
    for stim_condition in ['Anode', 'Cathode']:
        condition = 'ShamPre%s' % stim_condition
        ellipse_x, ellipse_y=get_twod_confidence_interval(condition_sigmoid_offsets['L*'][condition],condition_sigmoid_offsets['R*'][condition])
        ax.plot(ellipse_x,ellipse_y,'%s-' % colors[stim_condition])
        ax.plot(condition_sigmoid_offsets['L*'][condition],condition_sigmoid_offsets['R*'][condition],'o%s' % colors[stim_condition], label=condition)
    ax.plot(limits, limits, '--k')
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel('Indifference Point for L* Trials')
    ax.set_ylabel('Indifference Point for R* Trials')
    ax.legend(loc='best')

    fig = plt.figure()
    limits = [-.25, .25]
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        ax = fig.add_subplot(1, 2, cond_idx + 1, aspect='equal',adjustable='box-forced')
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            ellipse_x, ellipse_y=get_twod_confidence_interval(condition_sigmoid_offsets['L*'][condition],condition_sigmoid_offsets['R*'][condition])
            ax.plot(ellipse_x,ellipse_y,'%s-' % colors[condition])
            ax.plot(condition_sigmoid_offsets['L*'][condition], condition_sigmoid_offsets['R*'][condition],
                'o%s' % colors[condition], label=condition)
        ax.plot(limits, limits, '--k')
        ax.set_xlim(limits)
        ax.set_ylim(limits)
        ax.set_xlabel('Indifference Point for L* Trials')
        ax.set_ylabel('Indifference Point for R* Trials')
        ax.legend(loc='best')


def plot_choice_accuracy(colors, condition_coherence_accuracy, plot_err=False):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for stim_condition in ['Anode', 'Cathode']:
        condition='ShamPre%s' % stim_condition
        coherences = sorted(condition_coherence_accuracy[condition].keys())
        mean_accuracy=[]
        stderr_accuracy=[]
        for coherence in coherences:
            mean_accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
            if plot_err:
                stderr_accuracy.append(np.std(condition_coherence_accuracy[condition][coherence])/np.sqrt(len(condition_coherence_accuracy[condition][coherence])))
        acc_fit = FitWeibull(coherences, mean_accuracy, guess=[0.0, 0.2], display=0)
        smoothInt = np.arange(.01, 1.0, 0.001)
        smoothResp = acc_fit.eval(smoothInt)
        ax.semilogx(smoothInt, smoothResp, colors[stim_condition], label=condition)
        if plot_err:
            ax.errorbar(coherences, mean_accuracy, yerr=stderr_accuracy, fmt='o%s' % colors[stim_condition])
        else:
            ax.plot(coherences, mean_accuracy, 'o%s' % colors[stim_condition])
        thresh=acc_fit.inverse(0.8)
        ax.plot([thresh,thresh],[0.5,1],'--%s' % colors[stim_condition])
    ax.legend(loc='best')
    ax.set_xlim([0.01,1.0])
    ax.set_ylim([0.5,1.0])
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% Correct')

    fig = plt.figure()
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        ax = fig.add_subplot(1, 2, cond_idx + 1,adjustable='box-forced')
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            coherences = sorted(condition_coherence_accuracy[condition].keys())
            mean_accuracy = []
            stderr_accuracy = []
            for coherence in coherences:
                mean_accuracy.append(np.mean(condition_coherence_accuracy[condition][coherence]))
                if plot_err:
                    stderr_accuracy.append(np.std(condition_coherence_accuracy[condition][coherence])/np.sqrt(len(condition_coherence_accuracy[condition][coherence])))
            acc_fit = FitWeibull(coherences, mean_accuracy, guess=[0.0, 0.2], display=0)
            smoothInt = np.arange(.01, 1.0, 0.001)
            smoothResp = acc_fit.eval(smoothInt)
            ax.semilogx(smoothInt, smoothResp, colors[condition], label=condition)
            if plot_err:
                ax.errorbar(coherences, mean_accuracy, yerr=stderr_accuracy, fmt='o%s' % colors[condition])
            else:
                ax.plot(coherences, mean_accuracy, 'o%s' % colors[condition])
            thresh=acc_fit.inverse(0.8)
            ax.plot([thresh,thresh],[0.5,1],'--%s' % colors[condition])
        ax.set_xlim([0.01,1.0])
        ax.set_ylim([0.5,1.0])
        ax.legend(loc='best')
        ax.set_xlabel('Coherence')
        ax.set_ylabel('% Correct')


def plot_choice_rt(colors, condition_coherence_rt):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for stim_condition in ['Anode', 'Cathode']:
        condition='ShamPre%s' % stim_condition
        coherences = sorted(condition_coherence_rt[condition].keys())
        mean_rt=[]
        stderr_rt=[]
        for coherence in coherences:
            mean_rt.append(np.mean(condition_coherence_rt[condition][coherence]))
            stderr_rt.append(np.std(condition_coherence_rt[condition][coherence])/np.sqrt(len(condition_coherence_rt[condition][coherence])))
        rt_fit = FitRT(coherences, mean_rt, guess=[1,1,1], display=0)
        smoothInt = np.arange(min(coherences), max(coherences), 0.001)
        smoothRT = rt_fit.eval(smoothInt)
        ax.semilogx(smoothInt, smoothRT, colors[stim_condition], label=condition)
        ax.errorbar(coherences, mean_rt, yerr=stderr_rt,fmt='o%s' % colors[stim_condition])
    ax.set_xlim([0.02,1])
    ax.set_ylim([490, 680])
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT')

    fig = plt.figure()
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        ax = fig.add_subplot(1, 2, cond_idx + 1)
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            coherences = sorted(condition_coherence_rt[condition].keys())
            mean_rt = []
            stderr_rt = []
            for coherence in coherences:
                mean_rt.append(np.mean(condition_coherence_rt[condition][coherence]))
                stderr_rt.append(np.std(condition_coherence_rt[condition][coherence])/np.sqrt(len(condition_coherence_rt[condition][coherence])))
            rt_fit = FitRT(coherences, mean_rt, guess=[1,1,1], display=0)
            smoothInt = np.arange(min(coherences), max(coherences), 0.001)
            smoothRT = rt_fit.eval(smoothInt)
            ax.semilogx(smoothInt, smoothRT, colors[condition], label=condition)
            ax.errorbar(coherences, mean_rt, yerr=stderr_rt, fmt='o%s' % colors[condition])
        ax.legend(loc='best')
        ax.set_xlabel('Coherence')
        ax.set_ylabel('RT')
        ax.set_xlim([0.02,1])
        ax.set_ylim([490, 680])


def plot_choice_rt_diff(colors, condition_coherence_rt_diff, plot_err=False):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    coherences = np.array(sorted(condition_coherence_rt_diff['Sham'].keys()))
    mean_diff=[]
    stderr_diff=[]
    for coherence in coherences:
        mean_diff.append(np.mean(condition_coherence_rt_diff['Sham'][coherence]))
        if plot_err:
            stderr_diff.append(np.std(condition_coherence_rt_diff['Sham'][coherence])/np.sqrt(len(condition_coherence_rt_diff['Sham'][coherence])))
    mean_diff=np.array(mean_diff)

    clf = LinearRegression()
    clf.fit(np.expand_dims(coherences,axis=1),np.expand_dims(mean_diff,axis=1))
    a = clf.coef_[0][0]
    b = clf.intercept_[0]
    r_sqr=clf.score(np.expand_dims(coherences,axis=1), np.expand_dims(mean_diff,axis=1))
    ax.plot([np.min(coherences), np.max(coherences)], [a * np.min(coherences) + b, a * np.max(coherences) + b], '--b',
        label='r^2=%.3f' % r_sqr)

    if plot_err:
        ax.errorbar(coherences, mean_diff, yerr=stderr_diff, fmt='ob')
    else:
        ax.plot(coherences, mean_diff, 'ob')

    ax.set_xlim([0,0.55])
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT Difference')


    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for stim_condition in ['Anode', 'Cathode']:
        coherences = np.array(sorted(condition_coherence_rt_diff[stim_condition].keys()))
        mean_diff=[]
        stderr_diff=[]
        for coherence in coherences:
            mean_diff.append(np.mean(condition_coherence_rt_diff[stim_condition][coherence]))
            if plot_err:
                stderr_diff.append(np.std(condition_coherence_rt_diff[stim_condition][coherence])/np.sqrt(len(condition_coherence_rt_diff[stim_condition][coherence])))
        mean_diff=np.array(mean_diff)

        clf = LinearRegression()
        clf.fit(np.expand_dims(coherences,axis=1),np.expand_dims(mean_diff,axis=1))
        a = clf.coef_[0][0]
        b = clf.intercept_[0]
        r_sqr=clf.score(np.expand_dims(coherences,axis=1), np.expand_dims(mean_diff,axis=1))
        ax.plot([np.min(coherences), np.max(coherences)], [a * np.min(coherences) + b, a * np.max(coherences) + b], '--%s' % colors[stim_condition],
            label='r^2=%.3f' % r_sqr)

        if plot_err:
            ax.errorbar(coherences, mean_diff, yerr=stderr_diff, fmt='o%s' % colors[stim_condition], label=stim_condition)
        else:
            ax.plot(coherences, mean_diff, 'o%s' % colors[stim_condition], label=stim_condition)
    ax.legend(loc='best')
    ax.set_xlim([0,0.55])
    ax.set_xlabel('Coherence')
    ax.set_ylabel('RT Difference')

def plot_choice_probability(colors, condition_coherence_choices):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for stim_condition in ['Anode', 'Cathode']:
        condition='ShamPre%s' % stim_condition
        full_coherences = sorted(condition_coherence_choices['L*'][condition].keys())
        left_choice_probs = []
        right_choice_probs = []
        for coherence in full_coherences:
            left_choice_probs.append(np.mean(condition_coherence_choices['L*'][condition][coherence]))
            right_choice_probs.append(np.mean(condition_coherence_choices['R*'][condition][coherence]))
        plot_condition_choice_probability(ax, colors[stim_condition], condition, full_coherences, left_choice_probs, right_choice_probs)

    fig = plt.figure()
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        ax = fig.add_subplot(1, 2, cond_idx + 1)
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            full_coherences = sorted(condition_coherence_choices['L*'][condition].keys())
            left_choice_probs = []
            right_choice_probs = []
            for coherence in full_coherences:
                left_choice_probs.append(np.mean(condition_coherence_choices['L*'][condition][coherence]))
                right_choice_probs.append(np.mean(condition_coherence_choices['R*'][condition][coherence]))
            plot_condition_choice_probability(ax, colors[condition], condition, full_coherences, left_choice_probs, right_choice_probs)


def plot_condition_choice_probability(ax, color, condition, full_coherences, left_choice_probs, right_choice_probs):
    acc_fit = FitSigmoid(full_coherences, left_choice_probs, guess=[0.0, 0.2], display=0)
    smoothInt = np.arange(min(full_coherences), max(full_coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, '--%s' % color, label='L* %s' % condition)
    ax.plot(full_coherences, left_choice_probs, 'o%s' % color)
    acc_fit = FitSigmoid(full_coherences, right_choice_probs, guess=[0.0, 0.2], display=0)
    smoothInt = np.arange(min(full_coherences), max(full_coherences), 0.001)
    smoothResp = acc_fit.eval(smoothInt)
    ax.plot(smoothInt, smoothResp, color, label='R* %s' % condition)
    ax.plot(full_coherences, right_choice_probs, 'o%s' % color)
    ax.legend(loc='best')
    ax.set_xlabel('Coherence')
    ax.set_ylabel('% of Right Choices')


def plot_logistic_parameter_ratio(colors, condition_logistic_params):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    binwidth=.02
    for stim_condition in ['Anode', 'Cathode']:
        condition = 'ShamPre%s' % stim_condition
        ratio=np.array(condition_logistic_params['a2'][condition]) / np.array(condition_logistic_params['a1'][condition])
        bins=np.arange(min(ratio), max(ratio) + binwidth, binwidth)
        ax.hist(ratio, normed=1, label=condition, color=colors[stim_condition], bins=bins, alpha=.75)
        (mu, sigma) = norm.fit(ratio)
        y = normpdf( np.arange(-.1,.2,0.001), mu, sigma)
        ax.plot(np.arange(-.1,.2,0.001), y, '%s--' % colors[stim_condition], linewidth=2)
    ax.legend(loc='best')
    ax.set_xlim([-.1, .2])
    ax.set_ylim([0, 18])
    ax.set_xlabel('a2/a1')
    ax.set_ylabel('% subjects')

    fig = plt.figure()
    for cond_idx, stim_condition in enumerate(['Anode', 'Cathode']):
        ax = fig.add_subplot(1, 2, cond_idx+1)
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            ratio=np.array(condition_logistic_params['a2'][condition]) / np.array(condition_logistic_params['a1'][condition])
            bins=np.arange(min(ratio), max(ratio) + binwidth, binwidth)
            ax.hist(ratio, normed=1, label=condition, color=colors[condition], bins=bins, alpha=.75)
            (mu, sigma) = norm.fit(ratio)
            y = normpdf( np.arange(-.1,.2,0.001), mu, sigma)
            ax.plot(np.arange(-.1,.2,0.001), y, '%s--' % colors[condition], linewidth=2)
        ax.set_xlim([-.1, .2])
        ax.set_ylim([0,18])
        ax.legend(loc='best')
        ax.set_xlabel('a2/a1')
        ax.set_ylabel('% subjects')


def plot_accuracy_thresh(colors, condition_accuracy_thresh):
    fig=plt.figure()
    ax=fig.add_subplot(1,1,1)
    for stim_condition in ['Anode','Cathode']:
        condition='ShamPre%s' % stim_condition
        ax.hist(condition_accuracy_thresh[condition], label=condition, color=colors[stim_condition], bins=10, alpha=0.75)
    ax.set_xlabel('Threshold')
    ax.set_ylabel('# Subjects')

    fig=plt.figure()
    for cond_idx, stim_condition in enumerate(['Anode','Cathode']):
        ax=fig.add_subplot(1,2,cond_idx+1)
        for condition in ['ShamPre%s' % stim_condition, stim_condition]:
            ax.hist(condition_accuracy_thresh[condition], label=condition, color=colors[condition], bins=10, alpha=0.75)
        ax.legend(loc='best')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('# Subjects')

if __name__=='__main__':
    data_dir='../../data/stim2'
    #analyze_single_subj_choice_prob('RIR','../../data/stim2')
    #analyze_single_subj_choice_prob('LZ',data_dir,plot=True)

    excluded_subjects=[]
    subjects=read_subjects(data_dir, filter=True, collapse_sham=False, response_only=True)
    filtered_subjects=exclude_subjects(subjects, excluded_subjects=excluded_subjects)
    analyze_choice_hysteresis(filtered_subjects, itis='all')
    print('')
    analyze_accuracy_rt(filtered_subjects)
    plt.show()

