from jinja2 import Environment, FileSystemLoader
import os
from datetime import datetime
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon, mannwhitneyu
import matplotlib.pyplot as plt
from psychopy import data
import pylab
from rdmd.utils import make_report_dirs, Struct, FitRT, save_to_png, TEMPLATE_DIR, mdm_outliers

subjects={
    'AG': [
        ['control','control','control'],
        ['control','sham - pre - cathode','cathode'],
        ['control','sham - pre - anode','anode']
    ],
    'AN': [
        ['control','control','control'],
        ['sham - pre - anode','anode','sham - post - anode'],
        ['control','sham - pre - cathode','cathode']
    ],
    'CCM': [
        ['control','control','control'],
        ['control','sham - pre - cathode','cathode'],
        ['sham - pre - anode','anode','sham - post - anode']
    ],
    'CT': [
        ['control','control','control'],
        ['control','sham - pre - anode','anode'],
        ['control','sham - pre - cathode','cathode']
    ],
    'EZ': [
        ['control','control','control'],
        ['sham - pre - cathode','cathode','sham - post - cathode'],
        ['sham - pre - anode','anode','sham - post - anode']
    ],
    #'IL': [
    #    ['control','control','control'],
    #    ['sham - pre - anode','anode','sham - post - anode'],
    #    ['control','sham - pre - cathode','cathode'],
    #],
    'JH': [
        ['control','control','control'],
        #['sham - pre - anode','anode','sham - post - anode'],
        ['sham - pre - cathode','cathode','sham - post - cathode'],
        #['control','sham - pre - cathode','cathode']
        ['control','sham - pre - anode','anode']
    ],
    'LZ': [
        ['control','control','control'],
        ['control','sham - pre - anode','anode'],
        ['sham - pre - cathode','cathode','sham - post - cathode']
    ],
    'ME': [
        ['control','control','control'],
        ['control','sham - pre - cathode','cathode'],
        ['control','sham - pre - anode','anode']
    ],
    'MF': [
        ['control','control','control'],
        #['sham - pre - anode','anode','sham - post - anode'],
        ['sham - pre - cathode','cathode','sham - post - cathode'],
        #['sham - pre - cathode','cathode','sham - post - cathode']
        ['sham - pre - anode','anode','sham - post - anode']
    ],
    'NH': [
        ['control','control','control'],
        ['sham - pre - cathode','cathode','sham - post - cathode'],
        ['control','sham - pre - anode','anode']
    ],
    'NM': [
        ['control','control','control'],
        ['sham - pre - anode','anode','sham - post - anode'],
        ['sham - pre - cathode','cathode','sham - post - cathode']
    ],
    'RIR': [
        ['control','control','control'],
        ['control','sham - pre - anode','anode'],
        ['control','sham - pre - cathode','cathode']
    ],
    'RR': [
        ['control','control','control'],
        ['control','sham - pre - cathode','cathode'],
        ['sham - pre - anode','anode','sham - post - anode']
    ],
    'SB': [
        ['control','control','control'],
        ['sham - pre - anode','anode','sham - post - anode'],
        ['control','sham - pre - cathode','cathode']
    ],
    'SM': [
        ['control','control','control'],
        ['control','sham - pre - anode','anode'],
        ['sham - pre - cathode','cathode','sham - post - cathode']
    ],
    'YK': [
        ['control','control','control'],
        #['sham - pre - anode','anode','sham - post - anode'],
        ['sham - pre - cathode','cathode','sham - post - cathode'],
        #['sham - pre - cathode','cathode','sham - post - cathode']
        ['sham - pre - anode','anode','sham - post - anode']
    ],
    'YY': [
        ['control','control','control'],
        #['control','sham - pre - cathode','cathode'],
        ['control','sham - pre - anode','anode'],
        #['sham - pre - anode','anode','sham - post - anode']
        ['sham - pre - cathode','cathode','sham - post - cathode']
    ]
}

condition_colors={'control':'b','anode':'r','cathode':'g','sham - pre - anode':'r','sham - pre - cathode': 'g', 'sham - post - anode':'k', 'sham - post - cathode':'m'}
condition_styles={'control':'-','anode':'-','cathode':'-','sham - pre - anode':'--','sham - pre - cathode': '--', 'sham - post - anode':'--', 'sham - post - cathode':'--'}
condition_alphas={'control':0.75,'anode':0.75,'cathode':0.75,'sham - pre - anode':0.5,'sham - pre - cathode':0.5, 'sham - post - anode':0.5, 'sham - post - cathode':0.5}


class ConditionAggregatedReport:
    def __init__(self):
        self.conditions=[]
        self.coherent_responses={}
        self.difficult_rts={}
        self.rts={}
        self.coherence_responses={}
        self.coherence_rts={}
        self.mean_rt={}
        self.std_rt={}
        self.coherence_perc_correct={}
        self.perc_correct={}
        self.difficult_mean_rt={}
        self.all_coherence_levels={}
        self.urls={}

    def aggregate_condition_stats(self):
        self.perc_correct={}
        self.difficult_mean_rt={}
        self.all_coherence_levels={}
        self.mean_rt={}
        self.std_rt={}
        self.coherence_perc_correct={}
        for condition in self.conditions:
            self.perc_correct[condition]=np.mean(self.coherent_responses[condition])*100.0
            self.difficult_mean_rt[condition]=np.mean(self.difficult_rts[condition])
            self.all_coherence_levels[condition]=sorted(self.coherence_responses[condition].keys())
            if not condition in self.mean_rt:
                self.mean_rt[condition]=[]
                self.std_rt[condition]=[]
                self.coherence_perc_correct[condition]=[]
            for coherence in self.all_coherence_levels[condition]:
                self.mean_rt[condition].append(np.mean(self.coherence_rts[condition][coherence]))
                self.std_rt[condition].append(np.std(self.coherence_rts[condition][coherence])/float(len(self.coherence_rts[condition][coherence])))
                if coherence>0:
                    self.coherence_perc_correct[condition].append(np.mean(self.coherence_responses[condition][coherence]))
        self.rt_fit={}
        self.a={}
        self.k={}
        self.tr={}
        self.acc_fit={}
        self.alpha={}
        self.beta={}
        self.thresh={}
        for condition in self.conditions:
            self.rt_fit[condition] = FitRT(self.all_coherence_levels[condition], self.mean_rt[condition], guess=[1,1,1])
            self.a[condition]=self.rt_fit[condition].params[0]
            self.k[condition]=self.rt_fit[condition].params[1]
            self.tr[condition]=self.rt_fit[condition].params[2]

            self.acc_fit[condition] = data.FitWeibull(self.all_coherence_levels[condition][1:],
                self.coherence_perc_correct[condition], guess=[0.2, 0.5])
            self.alpha[condition]=self.acc_fit[condition].params[0]
            self.beta[condition]=self.acc_fit[condition].params[1]
            self.thresh[condition]=np.max([0,self.acc_fit[condition].inverse(0.8)])

    def aggregate_low_level_stats(self, lower_level_obj):
        for condition in lower_level_obj.coherence_responses:
            if not condition in self.conditions:
                self.conditions.append(condition)
                self.coherence_responses[condition]={}
                self.coherence_rts[condition]={}
            for coherence in lower_level_obj.coherence_responses[condition]:
                if not coherence in self.coherence_responses[condition]:
                    self.coherence_responses[condition][coherence]=[]
                self.coherence_responses[condition][coherence].extend(lower_level_obj.coherence_responses[condition][coherence])
            for coherence in lower_level_obj.coherence_rts[condition]:
                if not coherence in self.coherence_rts[condition]:
                    self.coherence_rts[condition][coherence]=[]
                self.coherence_rts[condition][coherence].extend(lower_level_obj.coherence_rts[condition][coherence])
            if not condition in self.rts:
                self.rts[condition]=[]
            self.rts[condition].extend(lower_level_obj.rts[condition])

    def plot_stats(self, report_dir, suffix):
        condition_rt_fits=[]
        condition_all_coherence_levels=[]
        condition_mean_rts=[]
        condition_std_rts=[]
        condition_labels=[]
        condition_acc_fits=[]
        condition_perc_correct_coherence_levels=[]
        condition_perc_correct=[]
        condition_rts=[]
        colors=[]
        styles=[]
        alphas=[]
        for condition in self.conditions:
            if not condition=='control' and not condition.startswith('sham - post'):
                condition_rt_fits.append(self.rt_fit[condition])
                condition_all_coherence_levels.append(self.all_coherence_levels[condition])
                condition_mean_rts.append(self.mean_rt[condition])
                condition_std_rts.append(self.std_rt[condition])
                condition_labels.append(condition)
                condition_acc_fits.append(self.acc_fit[condition])
                condition_perc_correct_coherence_levels.append(self.all_coherence_levels[condition][1:])
                condition_perc_correct.append(self.coherence_perc_correct[condition])
                condition_rts.append(self.rts[condition])
                colors.append(condition_colors[condition])
                styles.append(condition_styles[condition])
                alphas.append(condition_alphas[condition])

        furl='img/mean_rt_%s' % suffix
        self.urls['mean_rt_%s' % suffix]='%s.png' % furl
        plot_coherence_rt(furl, report_dir, condition_rt_fits, condition_all_coherence_levels, condition_mean_rts,
            condition_std_rts, colors, styles, condition_labels)

        furl='img/mean_perc_correct_%s' % suffix
        self.urls['mean_perc_correct_%s' % suffix]='%s.png' % furl
        plot_coherence_perc_correct(furl, report_dir, condition_acc_fits, condition_perc_correct_coherence_levels,
            condition_perc_correct, colors, styles, condition_labels)

        furl='img/mean_rt_dist_%s' % suffix
        self.urls['mean_rt_dist_%s' % suffix]='%s.png' % furl
        plot_rt_dist(furl, report_dir, condition_rts, colors, alphas, condition_labels)

        self.urls['coherence_mean_rt_%s' % suffix]={}
        for coherence in self.all_coherence_levels[self.all_coherence_levels.keys()[0]]:
            condition_coherence_rts=[]
            for condition in self.conditions:
                if not condition=='control' and not condition.startswith('sham - post'):
                    condition_coherence_rts.append(self.coherence_rts[condition][coherence])
            furl='img/mean_rt_dist_%s_%.4f' % (suffix,coherence)
            self.urls['coherence_mean_rt_%s' % suffix][coherence]='%s.png' % furl
            plot_coherence_rt_dist(furl, report_dir, coherence, condition_coherence_rts, colors, alphas,
                condition_labels)

        if 'anode' in self.conditions and 'cathode' in self.conditions:
            furl='img/diff_mean_rt_%s' % suffix
            fname=os.path.join(report_dir, furl)
            self.urls['diff_mean_rt_%s' % suffix]='%s.png' % furl
            fig=plt.figure()
            plt.errorbar(self.all_coherence_levels['anode'],np.array(self.mean_rt['anode'])-np.array(self.mean_rt['sham - pre - anode']),yerr=self.std_rt['anode'],fmt='or')
            plt.errorbar(self.all_coherence_levels['cathode'],np.array(self.mean_rt['cathode'])-np.array(self.mean_rt['sham - pre - cathode'])  ,yerr=self.std_rt['cathode'],fmt='og')
            plt.legend(loc='best')
            plt.xscale('log')
            plt.xlabel('Coherence')
            plt.ylabel('Decision time (s)')
            save_to_png(fig, '%s.png' % fname)
            plt.close(fig)

class Experiment(ConditionAggregatedReport):
    def __init__(self, name):
        ConditionAggregatedReport.__init__(self)
        self.name=name
        self.subjects=[]

    def plot_stats(self, reports_dir, suffix):
        ConditionAggregatedReport.plot_stats(self, reports_dir, suffix)
    
        furl='img/a_dist_%s' % suffix
        self.urls['a_dist_%s' % suffix]='%s.png' % furl
        plot_param_dist(furl, reports_dir, self.a_vals, 'A')
    
        furl='img/k_dist_%s' % suffix
        self.urls['k_dist_%s' % suffix]='%s.png' % furl
        plot_param_dist(furl, reports_dir, self.k_vals, 'K')
    
        furl='img/tr_dist_%s' % suffix
        self.urls['tr_dist_%s' % suffix]='%s.png' % furl
        plot_param_dist(furl, reports_dir, self.tr_vals, 'TR')
    
        furl='img/alpha_dist_%s' % suffix
        self.urls['alpha_dist_%s' % suffix]='%s.png' % furl
        plot_param_dist(furl, reports_dir, self.alpha_vals, 'Alpha')
    
        furl='img/beta_dist_%s' % suffix
        self.urls['beta_dist_%s' % suffix]='%s.png' % furl
        plot_param_dist(furl, reports_dir, self.beta_vals, 'Beta')

        furl='img/thresh_dist_%s' % suffix
        self.urls['thresh_dist_%s' % suffix]='%s.png' % furl
        plot_param_dist(furl, reports_dir, self.thresh_vals, 'Thresh')
        
    def filter(self):
        for condition in ['sham - pre - anode','sham - pre - cathode']:
            a_vals=[]
            k_vals=[]
            alpha_vals=[]
            beta_vals=[]
            thresh_vals=[]
            for idx,subject in enumerate(self.subjects):
                a_vals.append(subject.a[condition])
                k_vals.append(subject.k[condition])
                alpha_vals.append(subject.alpha[condition])
                beta_vals.append(subject.beta[condition])
                thresh_vals.append(subject.thresh[condition])

            thresh_outliers=mdm_outliers(np.array(thresh_vals))
            for outlier_idx in thresh_outliers:
                self.subjects[outlier_idx].excluded=True
                
    def aggregate_condition_stats(self):
        self.conditions=[]
        self.coherent_responses={}
        self.difficult_rts={}
        self.rts={}
        self.coherence_responses={}
        self.coherence_rts={}
        all_coherent_responses=[]
        all_difficult_rts=[]
        self.a_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        self.k_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        self.tr_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        self.alpha_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        self.beta_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        self.thresh_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        self.perc_correct_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        for subject in self.subjects:
            if not subject.excluded:
                for condition in self.a_vals:
                    self.a_vals[condition].append(subject.a[condition])
                    self.k_vals[condition].append(subject.k[condition])
                    self.tr_vals[condition].append(subject.tr[condition])
                    self.alpha_vals[condition].append(subject.alpha[condition])
                    self.beta_vals[condition].append(subject.beta[condition])
                    self.thresh_vals[condition].append(subject.thresh[condition])
                    self.perc_correct_vals[condition].append(subject.perc_correct[condition])
                self.aggregate_low_level_stats(subject)
                for session in subject.sessions:
                    for run in session.runs:
                        if not run.excluded:
                            if not run.condition in self.coherent_responses:
                                self.coherent_responses[run.condition]=[]
                                self.difficult_rts[run.condition]=[]
                            for idx,trial in enumerate(run.trials):
                                if trial.coherence>0:
                                    self.coherent_responses[run.condition].append(trial.response)
                                    all_coherent_responses.append(trial.response)
                                else:
                                    self.difficult_rts[run.condition].append(trial.rt)
                                    all_difficult_rts.append(trial.rt)

        self.all_perc_correct=np.mean(all_coherent_responses)*100.0
        self.all_difficult_mean_rt=np.mean(all_difficult_rts)

        ConditionAggregatedReport.aggregate_condition_stats(self)
                    
    def create_report(self, data_dir, reports_dir):
        make_report_dirs(reports_dir)

        for subj_id in subjects:
            subject=Subject(subj_id)
            subject.create_report(data_dir, os.path.join(reports_dir,subj_id))
            self.subjects.append(subject)

        self.aggregate_condition_stats()
        self.plot_stats(reports_dir, 'pre')
        self.filter()
        self.aggregate_condition_stats()
        self.plot_stats(reports_dir, 'post')

        self.alpha_means={}
        self.alpha_stats={}
        self.beta_means={}
        self.beta_stats={}
        self.thresh_means={}
        self.thresh_stats={}
        self.a_means={}
        self.a_stats={}
        self.k_means={}
        self.k_stats={}
        self.tr_means={}
        self.tr_stats={}
        self.coherence_rt_means={}
        self.coherence_rt_stats={}
        self.perc_correct_means={}
        self.perc_correct_stats={}
        self.rt_means={}
        self.rt_stats={}
        for condition in self.thresh_vals:
            self.thresh_vals[condition]=np.array(self.thresh_vals[condition])
            self.thresh_means[condition]=np.mean(self.thresh_vals[condition])
            self.thresh_stats[condition]=shapiro(self.thresh_vals[condition])

            self.a_vals[condition]=np.array(self.a_vals[condition])
            self.a_means[condition]=np.mean(self.a_vals[condition])
            self.a_stats[condition]=shapiro(self.a_vals[condition])

            self.k_vals[condition]=np.array(self.k_vals[condition])
            self.k_means[condition]=np.mean(self.k_vals[condition])
            self.k_stats[condition]=shapiro(self.k_vals[condition])
            
            self.tr_vals[condition]=np.array(self.tr_vals[condition])
            self.tr_means[condition]=np.mean(self.tr_vals[condition])
            self.tr_stats[condition]=shapiro(self.tr_vals[condition])

            self.alpha_vals[condition]=np.array(self.alpha_vals[condition])
            self.alpha_means[condition]=np.mean(self.alpha_vals[condition])
            self.alpha_stats[condition]=shapiro(self.alpha_vals[condition])

            self.beta_vals[condition]=np.array(self.beta_vals[condition])
            self.beta_means[condition]=np.mean(self.beta_vals[condition])
            self.beta_stats[condition]=shapiro(self.beta_vals[condition])

            self.perc_correct[condition]=np.array(self.perc_correct_vals[condition])
            self.perc_correct_means[condition]=np.mean(self.perc_correct_vals[condition])
            self.perc_correct_stats[condition]=shapiro(self.perc_correct_vals[condition])

            self.rts[condition]=np.array(self.rts[condition])
            self.rt_means[condition]=np.mean(self.rts[condition])
            self.rt_stats[condition]=shapiro(self.rts[condition])


        self.anode_perc_correct_stats=ttest_rel(self.perc_correct['anode'],self.perc_correct['sham - pre - anode'])
        self.cathode_perc_correct_stats=ttest_rel(self.perc_correct['cathode'],self.perc_correct['sham - pre - cathode'])
        
        self.anode_alpha_stats=ttest_rel(self.alpha_vals['anode'],self.alpha_vals['sham - pre - anode'])
        self.cathode_alpha_stats=ttest_rel(self.alpha_vals['cathode'],self.alpha_vals['sham - pre - cathode'])

        self.anode_beta_stats=wilcoxon(self.beta_vals['anode'],self.beta_vals['sham - pre - anode'])
        self.cathode_beta_stats=wilcoxon(self.beta_vals['cathode'],self.beta_vals['sham - pre - cathode'])

        self.anode_thresh_stats=ttest_rel(self.thresh_vals['anode'],self.thresh_vals['sham - pre - anode'])
        self.cathode_thresh_stats=ttest_rel(self.thresh_vals['cathode'],self.thresh_vals['sham - pre - cathode'])

        self.anode_rt_stats=mannwhitneyu(self.rts['anode'], self.rts['sham - pre - anode'])
        self.cathode_rt_stats=mannwhitneyu(self.rts['cathode'], self.rts['sham - pre - cathode'])

        self.anode_a_stats=wilcoxon(self.a_vals['anode'],self.a_vals['sham - pre - anode'])
        self.cathode_a_stats=wilcoxon(self.a_vals['cathode'],self.a_vals['sham - pre - cathode'])

        self.anode_k_stats=ttest_rel(self.k_vals['anode'],self.k_vals['sham - pre - anode'])
        self.cathode_k_stats=ttest_rel(self.k_vals['cathode'],self.k_vals['sham - pre - cathode'])
        
        self.anode_tr_stats=ttest_rel(self.tr_vals['anode'],self.tr_vals['sham - pre - anode'])
        self.cathode_tr_stats=ttest_rel(self.tr_vals['cathode'],self.tr_vals['sham - pre - cathode'])

        #create report
        template_file='experiment.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='experiment.html'
        fname=os.path.join(reports_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)
        
        
class Subject(ConditionAggregatedReport):
    def __init__(self, id):
        ConditionAggregatedReport.__init__(self)
        self.id=id
        self.sessions=[]
        self.excluded=False

    def aggregate_condition_stats(self):
        all_coherent_responses=[]
        all_difficult_rts=[]
        for session in self.sessions:
            self.aggregate_low_level_stats(session)
            for run in session.runs:
                if not run.excluded:
                    if not run.condition in self.coherent_responses:
                        self.coherent_responses[run.condition]=[]
                        self.difficult_rts[run.condition]=[]
                    for idx,trial in enumerate(run.trials):
                        if trial.coherence>0:
                            self.coherent_responses[run.condition].append(trial.response)
                            all_coherent_responses.append(trial.response)
                        else:
                            self.difficult_rts[run.condition].append(trial.rt)
                            all_difficult_rts.append(trial.rt)

        self.all_perc_correct=np.mean(all_coherent_responses)*100.0
        self.all_difficult_mean_rt=np.mean(all_difficult_rts)

        ConditionAggregatedReport.aggregate_condition_stats(self)

    def plot_stats(self, report_dir, suffix):

        ConditionAggregatedReport.plot_stats(self, report_dir, suffix)

        linestyles=['-','--','.']
        colors=['r','g','b']

        session_rt_fits=[]
        session_acc_fits=[]
        session_all_coherence_levels=[]
        session_mean_rts=[]
        session_std_rts=[]
        session_perc_correct_coherence_levels=[]
        session_perc_correct=[]
        session_rts=[]
        session_colors=[]
        session_alphas=[]
        session_styles=[]
        session_labels=[]
        for session_idx,session in enumerate(self.sessions):
            for run_idx,run in enumerate(session.runs):
                if not run.excluded:
                    session_rt_fits.append(run.rt_fit)
                    session_acc_fits.append(run.acc_fit)
                    session_all_coherence_levels.append(run.all_coherence_levels)
                    session_mean_rts.append(run.mean_rt)
                    session_std_rts.append(run.std_rt)
                    session_perc_correct_coherence_levels.append(run.all_coherence_levels[1:])
                    session_perc_correct.append(run.coherence_perc_correct)
                    rts=[]
                    for run in session.runs:
                        rts.extend(run.rts)
                    session_rts.append(rts)
                    session_colors.append(colors[run_idx])
                    session_styles.append(linestyles[session_idx])
                    session_labels.append('session %d, run %d - %s' % ((session_idx+1),(run_idx+1),run.condition))
                    session_alphas.append(1)

        furl='img/rt_%s' % suffix
        self.urls['rt_%s' % suffix]='%s.png' % furl
        plot_coherence_rt(furl, report_dir, session_rt_fits, session_all_coherence_levels, session_mean_rts,
            session_std_rts, session_colors, session_styles, session_labels)

        furl='img/perc_correct_%s' % suffix
        self.urls['perc_correct_%s' % suffix]='%s.png' % furl
        plot_coherence_perc_correct(furl, report_dir, session_acc_fits, session_perc_correct_coherence_levels,
            session_perc_correct, session_colors, session_styles, session_labels)

        furl='img/rt_dist_%s' % suffix
        self.urls['rt_dist_%s' % suffix]='%s.png' % furl
        plot_rt_dist(furl, report_dir, session_rts, session_colors, session_alphas, session_labels)


    def create_report(self, data_dir, report_dir):
        print(self.id)
        make_report_dirs(report_dir)

        session_dict={}
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith('.csv'):
                file_name_parts=file_name.split('.')
                if file_name_parts[0].upper()==self.id:
                    session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')
                    if not session_date in session_dict:
                        session_dict[session_date]=Session(self.id, session_date)
        self.sessions = sorted(session_dict.values(), key=lambda x: x.date)
        for idx,session in enumerate(self.sessions):
            session.idx=idx+1
            session_report_dir=os.path.join(report_dir,str(idx+1))
            session.create_report(data_dir, session_report_dir)

        self.aggregate_condition_stats()

        self.plot_stats(report_dir, 'post')

        #create report
        template_file='subject.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='subject_%s.html' % self.id
        fname=os.path.join(report_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)


class Session(ConditionAggregatedReport):
    def __init__(self, subj_id, date):
        ConditionAggregatedReport.__init__(self)
        self.subj_id=subj_id
        self.date=date
        self.runs=[]

    def aggregate_condition_stats(self):
        all_coherent_responses=[]
        all_difficult_rts=[]
        for run in self.runs:
            if not run.excluded:
                if not run.condition in self.coherence_responses:
                    self.coherence_responses[run.condition]={}
                    self.coherence_rts[run.condition]={}
                if not run.condition in self.coherent_responses:
                    self.coherent_responses[run.condition]=[]
                    self.difficult_rts[run.condition]=[]
                if not run.condition in self.rts:
                    self.rts[run.condition]=[]
                self.rts[run.condition].extend(run.rts)
                for trial in run.trials:
                    if not trial.excluded:
                        if not trial.coherence in self.coherence_responses[run.condition]:
                            self.coherence_responses[run.condition][trial.coherence]=[]
                        self.coherence_responses[run.condition][trial.coherence].append(trial.response)
                        if trial.coherence==0 or trial.response:
                            if not trial.coherence in self.coherence_rts[run.condition]:
                                self.coherence_rts[run.condition][trial.coherence]=[]
                            self.coherence_rts[run.condition][trial.coherence].append(trial.rt)
                        if trial.coherence>0:
                            self.coherent_responses[run.condition].append(trial.response)
                            all_coherent_responses.append(trial.response)
                        else:
                            self.difficult_rts[run.condition].append(trial.rt)
                            all_difficult_rts.append(trial.rt)
        self.all_perc_correct=np.mean(all_coherent_responses)*100.0
        self.all_difficult_mean_rt=np.mean(all_difficult_rts)

        ConditionAggregatedReport.aggregate_condition_stats(self)

    def plot_stats(self, report_dir, suffix):
        ConditionAggregatedReport.plot_stats(self, report_dir, suffix)

        colors=['r','g','b']
        styles=['-','-','-']
        alphas=[1,1,1]

        run_rt_fits=[]
        run_acc_fits=[]
        run_all_coherence_levels=[]
        run_perc_correct_coherence_levels=[]
        run_perc_correct=[]
        run_rts=[]
        run_mean_rts=[]
        run_std_rts=[]
        run_labels=[]
        for idx,run in enumerate(self.runs):
            if not run.excluded:
                run_acc_fits.append(run.acc_fit)
                run_rt_fits.append(run.rt_fit)
                run_all_coherence_levels.append(run.all_coherence_levels)
                run_perc_correct_coherence_levels.append(run.all_coherence_levels[1:])
                run_perc_correct.append(run.coherence_perc_correct)
                run_rts.append(run.rts)
                run_mean_rts.append(run.mean_rt)
                run_std_rts.append(run.std_rt)
                run_labels.append('run %d - %s' % ((idx+1),run.condition))

        furl='img/rt_%s' % suffix
        self.urls['rt_%s' % suffix]='%s.png' % furl
        plot_coherence_rt(furl, report_dir, run_rt_fits, run_all_coherence_levels, run_mean_rts, run_std_rts, colors,
            styles, run_labels)

        furl='img/perc_correct_%s' % suffix
        self.urls['perc_correct_%s' % suffix]='%s.png' % furl
        plot_coherence_perc_correct(furl, report_dir, run_acc_fits, run_perc_correct_coherence_levels, run_perc_correct,
            colors, styles, run_labels)

        furl='img/rt_dist_%s' % suffix
        self.urls['rt_dist_%s' % suffix]='%s.png' % furl
        plot_rt_dist(furl, report_dir, run_rts, colors, alphas, run_labels)

    def create_report(self, data_dir, report_dir):
        make_report_dirs(report_dir)
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith('.csv'):
                file_name_parts=file_name.split('.')
                if file_name_parts[0].upper()==self.subj_id:
                    session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')
                    if session_date==self.date and not file_name_parts[2]=='training':
                        run_num=int(file_name_parts[2])
                        run=Run(self.subj_id, self.idx, self.date, run_num, subjects[self.subj_id][self.idx-1][run_num-1])
                        run_report_dir=os.path.join(report_dir,str(run_num))
                        run.create_report(data_dir, file_name, run_report_dir)
                        if run.thresh>1.0:
                            run.excluded=True
                        self.runs.append(run)
        self.runs=sorted(self.runs, key=lambda x: x.run_num)

        for run in self.runs:
            if not run.condition in self.conditions:
                self.conditions.append(run.condition)

        self.aggregate_condition_stats()
        self.plot_stats(report_dir, 'post')

        #create report
        template_file='session.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='session_%d.html' % self.idx
        fname=os.path.join(report_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)


class Run:
    def __init__(self, subj_id, session_idx, session_date, run_num, condition):
        self.subj_id=subj_id
        self.session_idx=session_idx
        self.session_date=session_date
        self.run_num=run_num
        self.condition=condition
        self.trials=[]
        self.coherence_rts={}
        self.excluded=False
        self.urls={}

    def filter(self):
        coherence_trial_idx={}
        coherence_rts={}
        for idx, trial in enumerate(self.trials):
            if trial.response:
                if not trial.coherence in coherence_trial_idx:
                    coherence_trial_idx[trial.coherence]=[]
                    coherence_rts[trial.coherence]=[]
                coherence_trial_idx[trial.coherence].append(idx)
                coherence_rts[trial.coherence].append(trial.rt)

        for coherence,rts in coherence_rts.iteritems():
            outliers=mdm_outliers(np.array(rts))
            for outlier_idx in outliers:
                self.trials[coherence_trial_idx[coherence][outlier_idx]].excluded=True

    def aggregate_stats(self):
        # Compute % correct and mean RT in 0% coherence trials
        coherent_responses=[]
        difficult_rts=[]
        for trial in self.trials:
            if not trial.excluded:
                if trial.coherence>0:
                    coherent_responses.append(trial.response)
                else:
                    difficult_rts.append(trial.rt)
        self.perc_correct=np.mean(coherent_responses)*100.0
        self.difficult_mean_rt=np.mean(difficult_rts)

        # All RTs
        self.rts=[]
        for trial in self.trials:
            if not trial.excluded and (trial.coherence==0 or trial.response):
                self.rts.append(trial.rt)

        self.coherence_responses={}
        self.coherence_rts={}
        for trial in self.trials:
            if not trial.excluded:
                if not trial.coherence in self.coherence_responses:
                    self.coherence_responses[trial.coherence]=[]
                self.coherence_responses[trial.coherence].append(trial.response)
                if trial.coherence==0 or trial.response:
                    if not trial.coherence in self.coherence_rts:
                        self.coherence_rts[trial.coherence]=[]
                    self.coherence_rts[trial.coherence].append(trial.rt)

        self.all_coherence_levels=sorted(self.coherence_responses.keys())

        self.mean_rt=[]
        self.std_rt=[]
        self.coherence_perc_correct=[]
        for coherence in self.all_coherence_levels:
            self.mean_rt.append(np.mean(self.coherence_rts[coherence]))
            self.std_rt.append(np.std(self.coherence_rts[coherence])/float(len(self.coherence_rts[coherence])))
            if coherence>0:
                self.coherence_perc_correct.append(np.mean(self.coherence_responses[coherence]))

    def plot_stats(self, report_dir, prefix):
        furl='img/rt_%s' % prefix
        self.urls['rt_%s' % prefix]='%s.png' % furl
        self.rt_fit = FitRT(self.all_coherence_levels, self.mean_rt, guess=[1,1,1])
        plot_coherence_rt(furl, report_dir, [self.rt_fit], [self.all_coherence_levels], [self.mean_rt], [self.std_rt],
            ['k'],['-'],[None])
        self.a=self.rt_fit.params[0]
        self.k=self.rt_fit.params[1]
        self.tr=self.rt_fit.params[2]

        furl='img/perc_correct_%s' % prefix
        self.urls['perc_correct_%s' % prefix]='%s.png' % furl
        self.acc_fit = data.FitWeibull(self.all_coherence_levels[1:], self.coherence_perc_correct, guess=[0.2, 0.5])
        plot_coherence_perc_correct(furl, report_dir, [self.acc_fit], [self.all_coherence_levels[1:]],
            [self.coherence_perc_correct], ['k'], ['-'], [None])
        self.alpha=self.acc_fit.params[0]
        self.beta=self.acc_fit.params[1]
        self.thresh = np.max([0,self.acc_fit.inverse(0.8)])

        furl='img/rt_dist_%s' % prefix
        self.urls['rt_dist_%s' % prefix]='%s.png' % furl
        plot_rt_dist(furl, report_dir, [self.rts],['b'],[1],[None])

        self.urls['coherence_rt_dist_%s' % prefix]={}
        for coherence,rts in self.coherence_rts.iteritems():
            furl='img/rt_dist_%0.4f_%s' % (coherence,prefix)
            self.urls['coherence_rt_dist_%s' % prefix][coherence]='%s.png' % furl
            plot_coherence_rt_dist(furl, report_dir, coherence, [rts], ['b'], [1], [None])

    def create_report(self, data_dir, file_name, report_dir):
        make_report_dirs(report_dir)

        file=open(os.path.join(data_dir,file_name),'r')
        for idx,line in enumerate(file):
            if idx>0:
                cols=line.split(',')
                direction=int(cols[0])
                coherence=float(cols[1])
                resp=int(cols[2])
                rt=float(cols[3])*1000.0
                trialIdx=idx-1
                trial=Trial(trialIdx,coherence,resp,rt)
                self.trials.append(trial)

        self.aggregate_stats()

        self.plot_stats(report_dir, 'pre')

        self.filter()

        self.aggregate_stats()

        self.plot_stats(report_dir, 'post')

        #create report
        template_file='run.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='run_%d.html' % self.run_num
        fname=os.path.join(report_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

class Trial:
    def __init__(self, idx, coherence, response, rt):
        self.idx=idx
        self.coherence=coherence
        self.response=response
        self.rt=rt
        self.excluded=False


def run_analysis(data_dir, reports_dir):
    experiment=Experiment('Random dot motion discrimination task')
    experiment.create_report(data_dir, reports_dir)


def plot_coherence_perc_correct(furl, report_dir, acc_fit_list, coherence_levels_list, perc_correct_list, colors,
                                styles, labels):
    fname=os.path.join(report_dir, furl)
    fig=plt.figure()

    for acc_fit, coherence_levels, perc_correct, color, style, label in zip(acc_fit_list, coherence_levels_list,
        perc_correct_list, colors, styles, labels):
        # Fit Weibull function
        thresh = np.max([0,acc_fit.inverse(0.8)])

        smoothInt = pylab.arange(0.0, max(coherence_levels), 0.001)
        smoothResp = acc_fit.eval(smoothInt)

        plt.plot(smoothInt, smoothResp, '%s%s' % (style,color), label=label)
        plt.plot(coherence_levels, perc_correct, 'o%s' % color)
        plt.plot([thresh,thresh],[0.4,1.0],'%s%s' % (style,color))
    plt.ylim([0.4,1])
    plt.legend(loc='best')
    plt.xlabel('Coherence')
    plt.ylabel('% Correct')
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)


def plot_coherence_rt(furl, report_dir, rt_fit_list, coherence_levels_list, mean_rt_list, std_rt_list, colors, styles,
                      labels):
    fname=os.path.join(report_dir, furl)
    fig=plt.figure()

    for rt_fit, coherence_levels, mean_rt, std_rt, color, style, label in zip(rt_fit_list, coherence_levels_list,
        mean_rt_list, std_rt_list, colors, styles, labels):

        smoothInt = pylab.arange(0.01, max(coherence_levels), 0.001)
        smoothResp = rt_fit.eval(smoothInt)

        plt.errorbar(coherence_levels, mean_rt,yerr=std_rt,fmt='o%s' % color)
        plt.plot(smoothInt, smoothResp, '%s%s' % (style,color), label=label)
    plt.legend(loc='best')
    plt.xscale('log')
    #plt.ylim([500, 1000])
    plt.xlabel('Coherence')
    plt.ylabel('Decision time (s)')
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)

def plot_rt_dist(furl, report_dir, rts_list, colors, alphas, labels):
    fname=os.path.join(report_dir,furl)
    fig=plt.figure()

    for rts, color, alpha, label in zip(rts_list,colors,alphas,labels):
        rt_hist,rt_bins=np.histogram(np.array(rts), bins=10, density=True)
        bin_width=rt_bins[1]-rt_bins[0]
        bars=plt.bar(rt_bins[:-1], rt_hist, width=bin_width, label=label)
        for bar in bars:
            bar.set_color(color)
            bar.set_alpha(alpha)
    plt.legend(loc='best')
    plt.xlabel('RT')
    plt.ylabel('Proportion of trials')
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)

def plot_coherence_rt_dist(furl, report_dir, coherence, rts_list, colors, alphas, labels):
    fname=os.path.join(report_dir,furl)
    fig=plt.figure()
    for rts, color, alpha, label in zip(rts_list, colors, alphas, labels):
        rt_hist,rt_bins=np.histogram(np.array(rts), bins=10, density=True)
        bin_width=rt_bins[1]-rt_bins[0]
        bars=plt.bar(rt_bins[:-1],rt_hist,width=bin_width, label=label)
        for bar in bars:
            bar.set_color(color)
            bar.set_alpha(alpha)
    plt.legend(loc='best')
    plt.xlabel('RT')
    plt.ylabel('Proportion of trials')
    plt.title('Coherence=%.4f' % coherence)
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)

def plot_param_dist(furl, report_dir, vals, param_name):
    fname=os.path.join(report_dir,furl)
    fig=plt.figure()
    for condition in vals:
        rt_hist,rt_bins=np.histogram(np.array(vals[condition]), bins=10, density=True)
        bin_width=rt_bins[1]-rt_bins[0]
        bars=plt.bar(rt_bins[:-1], rt_hist, width=bin_width, label=condition)
        for bar in bars:
            bar.set_color(condition_colors[condition])
            bar.set_alpha(condition_alphas[condition])
    plt.legend(loc='best')
    plt.xlabel(param_name)
    plt.ylabel('Proportion of subjects')
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)
    
if __name__=='__main__':
    #run_analysis('../../data/stim2','../../data/reports2')
    run_analysis('../../data/stim2','../../data/report_thresh_filtering')
