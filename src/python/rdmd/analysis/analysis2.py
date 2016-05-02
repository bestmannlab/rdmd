from jinja2 import Environment, FileSystemLoader
import os
from datetime import datetime
import numpy as np
from scipy.stats import shapiro, ttest_rel, wilcoxon
import matplotlib.pyplot as plt
from psychopy import data
import pylab
from sklearn.linear_model import LinearRegression
from rdmd.analysis.subject_info import subject_sessions
from rdmd.utils import make_report_dirs, FitRT, save_to_png, TEMPLATE_DIR, mdm_outliers, twoway_interaction, save_to_eps, sd_outliers, movingaverage

condition_colors={'control':'b','anode':'r','cathode':'g','sham - pre - anode':'b','sham - pre - cathode': 'b', 'sham - post - anode':'k', 'sham - post - cathode':'m'}
condition_styles={'control':'-','anode':'-','cathode':'-','sham - pre - anode':'--','sham - pre - cathode': '--', 'sham - post - anode':'--', 'sham - post - cathode':'--'}
condition_alphas={'control':0.75,'anode':0.75,'cathode':0.75,'sham - pre - anode':0.5,'sham - pre - cathode':0.5, 'sham - post - anode':0.5, 'sham - post - cathode':0.5}


class ConditionAggregatedReport:
    def __init__(self):
        self.conditions=[]
        self.coherent_responses={}
        self.coherent_rts={}
        self.difficult_rts={}
        self.rts={}
        self.speeds={}
        self.coherence_responses={}
        self.coherence_rts={}
        self.coherence_speeds={}
        self.mean_rt={}
        self.std_rt={}
        self.mean_speed={}
        self.std_speed={}
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
        self.mean_speed={}
        self.std_speed={}
        self.coherence_perc_correct={}
        for condition in self.conditions:
            self.perc_correct[condition]=np.mean(self.coherent_responses[condition])*100.0
            self.difficult_mean_rt[condition]=np.mean(self.difficult_rts[condition])
            self.all_coherence_levels[condition]=sorted(self.coherence_responses[condition].keys())
            if not condition in self.mean_rt:
                self.mean_rt[condition]=[]
                self.std_rt[condition]=[]
                self.coherence_perc_correct[condition]=[]
                self.mean_speed[condition]=[]
                self.std_speed[condition]=[]
            for coherence in self.all_coherence_levels[condition]:
                self.mean_rt[condition].append(np.mean(self.coherence_rts[condition][coherence]))
                self.std_rt[condition].append(np.std(self.coherence_rts[condition][coherence])/float(np.sqrt(len(self.coherence_rts[condition][coherence]))))
                self.mean_speed[condition].append(np.mean(self.coherence_speeds[condition][coherence]))
                self.std_speed[condition].append(np.std(self.coherence_speeds[condition][coherence])/float(np.sqrt(len(self.coherence_speeds[condition][coherence]))))
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
                self.coherence_speeds[condition]={}
            for coherence in lower_level_obj.coherence_responses[condition]:
                if not coherence in self.coherence_responses[condition]:
                    self.coherence_responses[condition][coherence]=[]
                #self.coherence_responses[condition][coherence].extend(lower_level_obj.coherence_responses[condition][coherence])
                self.coherence_responses[condition][coherence].append(np.mean(lower_level_obj.coherence_responses[condition][coherence]))
            for coherence in lower_level_obj.coherence_rts[condition]:
                if not coherence in self.coherence_rts[condition]:
                    self.coherence_rts[condition][coherence]=[]
                    self.coherence_speeds[condition][coherence]=[]
                #self.coherence_rts[condition][coherence].extend(lower_level_obj.coherence_rts[condition][coherence])
                self.coherence_rts[condition][coherence].append(np.mean(lower_level_obj.coherence_rts[condition][coherence]))
                #self.coherence_speeds[condition][coherence].extend(lower_level_obj.coherence_speeds[condition][coherence])
                self.coherence_speeds[condition][coherence].append(np.mean(lower_level_obj.coherence_speeds[condition][coherence]))
            if not condition in self.rts:
                self.rts[condition]=[]
                self.speeds[condition]=[]
            #self.rts[condition].extend(lower_level_obj.rts[condition])
            self.rts[condition].append(np.mean(lower_level_obj.rts[condition]))
            #self.speeds[condition].extend(lower_level_obj.speeds[condition])
            self.speeds[condition].append(np.mean(lower_level_obj.speeds[condition]))

    def plot_stats(self, report_dir, suffix, regenerate_plots):
        condition_rt_fits=[]
        condition_all_coherence_levels=[]
        condition_mean_rts=[]
        condition_std_rts=[]
        condition_mean_speeds=[]
        condition_std_speeds=[]
        condition_labels=[]
        condition_acc_fits=[]
        condition_perc_correct_coherence_levels=[]
        condition_perc_correct=[]
        condition_rts=[]
        condition_speeds=[]
        condition_coherent_responses=[]
        condition_coherent_rts=[]
        colors=[]
        styles=[]
        alphas=[]
        for condition in self.conditions:
            if not condition=='control' and not condition.startswith('sham - post'):
                condition_rt_fits.append(self.rt_fit[condition])
                condition_all_coherence_levels.append(self.all_coherence_levels[condition])
                condition_mean_rts.append(self.mean_rt[condition])
                condition_std_rts.append(self.std_rt[condition])
                condition_mean_speeds.append(self.mean_speed[condition])
                condition_std_speeds.append(self.std_speed[condition])
                condition_labels.append(condition)
                condition_acc_fits.append(self.acc_fit[condition])
                condition_perc_correct_coherence_levels.append(self.all_coherence_levels[condition][1:])
                condition_perc_correct.append(self.coherence_perc_correct[condition])
                condition_rts.append(self.rts[condition])
                condition_speeds.append(self.speeds[condition])
                condition_coherent_responses.append(self.coherent_responses[condition])
                condition_coherent_rts.append(self.coherent_rts[condition])
                colors.append(condition_colors[condition])
                styles.append(condition_styles[condition])
                alphas.append(condition_alphas[condition])

        furl='img/mean_rt_%s' % suffix
        self.urls['mean_rt_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_coherence_rt(furl, report_dir, condition_rt_fits, condition_all_coherence_levels, condition_mean_rts,
                condition_std_rts, colors, styles, condition_labels)

        furl='img/mean_perc_correct_%s' % suffix
        self.urls['mean_perc_correct_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_coherence_perc_correct(furl, report_dir, condition_acc_fits, condition_perc_correct_coherence_levels,
                condition_perc_correct, colors, styles, condition_labels)

        furl='img/mean_rt_dist_%s' % suffix
        self.urls['mean_rt_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_rt_dist(furl, report_dir, condition_rts, colors, alphas, condition_labels)

        furl='img/mean_speed_dist_%s' % suffix
        self.urls['mean_speed_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_speed_dist(furl, report_dir, condition_speeds, colors, alphas, condition_labels)

        self.urls['coherence_mean_rt_%s' % suffix]={}
        for coherence in self.all_coherence_levels[self.all_coherence_levels.keys()[0]]:
            condition_coherence_rts=[]
            for condition in self.conditions:
                if not condition=='control' and not condition.startswith('sham - post'):
                    condition_coherence_rts.append(self.coherence_rts[condition][coherence])
            furl='img/mean_rt_dist_%s_%.4f' % (suffix,coherence)
            self.urls['coherence_mean_rt_%s' % suffix][coherence]='%s.png' % furl
            if regenerate_plots:
                plot_coherence_rt_dist(furl, report_dir, coherence, condition_coherence_rts, colors, alphas,
                    condition_labels)

        self.urls['coherence_mean_speed_%s' % suffix]={}
        for coherence in self.all_coherence_levels[self.all_coherence_levels.keys()[0]]:
            condition_coherence_speeds=[]
            for condition in self.conditions:
                if not condition=='control' and not condition.startswith('sham - post'):
                    condition_coherence_speeds.append(self.coherence_speeds[condition][coherence])
            furl='img/mean_speed_dist_%s_%.4f' % (suffix,coherence)
            self.urls['coherence_mean_speed_%s' % suffix][coherence]='%s.png' % furl
            if regenerate_plots:
                plot_coherence_speed_dist(furl, report_dir, coherence, condition_coherence_speeds, colors, alphas,
                    condition_labels)

        furl='img/sat_%s' % suffix
        self.urls['sat_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_sat(furl, report_dir, condition_coherent_responses, condition_coherent_rts, colors, condition_labels)

        self.urls['coherence_sat_%s' % suffix]={}
        for coherence in self.all_coherence_levels[self.all_coherence_levels.keys()[0]][1:]:
            condition_coherent_rts=[]
            condition_coherent_responses=[]
            for condition in self.conditions:
                if not condition=='control' and not condition.startswith('sham - post'):
                    condition_coherent_rts.append(self.coherence_rts[condition][coherence])
                    condition_coherent_responses.append(self.coherence_responses[condition][coherence])
            furl='img/coherence_sat_%.04f_%s' % (coherence,suffix)
            self.urls['coherence_sat_%s' % suffix][coherence]='%s.png' % furl
            if regenerate_plots:
                plot_coherence_sat(furl, report_dir, coherence, condition_coherent_rts, condition_coherent_responses,
                    colors, condition_labels)

        if 'anode' in self.conditions and 'cathode' in self.conditions:
            furl='img/diff_mean_rt_%s' % suffix
            self.urls['diff_mean_rt_%s' % suffix]='%s.png' % furl
            if regenerate_plots:
                plot_diff_mean_rt(furl, report_dir, self.all_coherence_levels, self.mean_rt, self.std_rt)
                
            furl='img/diff_mean_perc_correct_%s' % suffix
            fname=os.path.join(report_dir, furl)
            self.urls['diff_mean_perc_correct_%s' % suffix]='%s.png' % furl
            if regenerate_plots:
                plot_diff_mean_perc_correct(furl, report_dir, self.all_coherence_levels, self.coherence_perc_correct)
            
            furl='img/diff_mean_speed_%s' % suffix
            fname=os.path.join(report_dir, furl)
            self.urls['diff_mean_speed_%s' % suffix]='%s.png' % furl
            if regenerate_plots:
                fig=plt.figure()
                plt.errorbar(self.all_coherence_levels['anode'],np.array(self.mean_speed['anode'])-np.array(self.mean_speed['sham - pre - anode']),yerr=self.std_speed['anode'],fmt='or')
                plt.errorbar(self.all_coherence_levels['cathode'],np.array(self.mean_speed['cathode'])-np.array(self.mean_speed['sham - pre - cathode']),yerr=self.std_speed['cathode'],fmt='og')
                plt.legend(loc='best')
                plt.xscale('log')
                plt.xlabel('Coherence')
                plt.ylabel('Speed')
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)

class Experiment(ConditionAggregatedReport):
    def __init__(self, name):
        ConditionAggregatedReport.__init__(self)
        self.name=name
        self.subjects=[]
        self.num_subjects=0
        self.num_post_filter_subjects=0

    def export_csv(self, reports_dir, filename):
        csv_file=open(os.path.join(reports_dir, filename),'w')
        csv_file.write('Subject,AccThreshAnode,AccThreshCathode,AccThreshShamPreAnode,AccThreshShamPreCathode')
        conditions=['anode','cathode','sham - pre - anode','sham - pre - cathode']
        for condition in conditions:
            for coherence_level in self.all_coherence_levels[condition]:
                csv_file.write(',RT%s%0.3f' % (condition.title().replace(' - ',''),coherence_level))
        for coherence_level in self.all_coherence_levels['control']:
            csv_file.write(',RTDiffAnode%0.3f' % coherence_level)
        for coherence_level in self.all_coherence_levels['control']:
            csv_file.write(',RTDiffCathode%0.3f' % coherence_level)
        csv_file.write('\n')
        for subject in self.subjects:
            if not subject.excluded:
                csv_file.write('%s,%.4f,%.4f,%.4f,%.4f' %
                               (subject.id,
                                subject.thresh['anode'],
                                subject.thresh['cathode'],
                                subject.thresh['sham - pre - anode'],
                                subject.thresh['sham - pre - cathode']))

                for condition in conditions:
                    for coherence_level in self.all_coherence_levels[condition]:
                        rts=[]
                        for session in subject.sessions:
                            for run in session.runs:
                                if not run.excluded and run.condition==condition:
                                    for trial in run.trials:
                                        if not trial.excluded and trial.coherence==coherence_level and trial.response>0:
                                            rts.append(trial.rt)

                        csv_file.write(',%0.4f' % np.mean(rts))

                subj_rts={}
                for coherence_level in self.all_coherence_levels['control']:
                    subj_rts[coherence_level]={}
                    for condition in conditions:
                        subj_rts[coherence_level][condition]=[]
                        for session in subject.sessions:
                            for run in session.runs:
                                if not run.excluded and run.condition==condition:
                                    for trial in run.trials:
                                        if not trial.excluded and trial.coherence==coherence_level and trial.response>0:
                                            subj_rts[coherence_level][condition].append(trial.rt)

                for condition in ['anode','cathode']:
                    diffs=[]
                    for coherence_level in self.all_coherence_levels['control']:
                        diff=np.mean(subj_rts[coherence_level][condition])-\
                             np.mean(subj_rts[coherence_level]['sham - pre - %s' % condition])
                        diffs.append(diff)
                    smoothed_rt_diff=movingaverage(diffs,3)
                    for diff in smoothed_rt_diff:
                        csv_file.write(',%0.4f' % diff)

                csv_file.write('\n')
        csv_file.close()

    def plot_stats(self, reports_dir, suffix, regenerate_plots):
        ConditionAggregatedReport.plot_stats(self, reports_dir, suffix, regenerate_plots)
    
        furl='img/a_dist_%s' % suffix
        self.urls['a_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_param_dist(furl, reports_dir, self.a_vals, 'A')
    
        furl='img/k_dist_%s' % suffix
        self.urls['k_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_param_dist(furl, reports_dir, self.k_vals, 'K')
    
        furl='img/tr_dist_%s' % suffix
        self.urls['tr_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_param_dist(furl, reports_dir, self.tr_vals, 'TR')
    
        furl='img/alpha_dist_%s' % suffix
        self.urls['alpha_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_param_dist(furl, reports_dir, self.alpha_vals, 'Alpha')
    
        furl='img/beta_dist_%s' % suffix
        self.urls['beta_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_param_dist(furl, reports_dir, self.beta_vals, 'Beta')

        furl='img/thresh_dist_%s' % suffix
        self.urls['thresh_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_param_dist(furl, reports_dir, self.thresh_vals, 'Thresh')

        self.urls['coherence_diff_speed_%s' % suffix]={}
        all_anode_speed_diffs=[]
        all_cathode_speed_diffs=[]
        mean_anode_speed_diffs=[]
        mean_cathode_speed_diffs=[]
        for coherence in self.all_coherence_levels[self.all_coherence_levels.keys()[0]]:
            anode_speed_diffs=[]
            cathode_speed_diffs=[]
            for subject in self.subjects:
                if not subject.excluded:
                    anode_speed_diffs.append(np.mean(subject.coherence_speeds['anode'][coherence])-np.mean(subject.coherence_speeds['sham - pre - anode'][coherence]))
                    cathode_speed_diffs.append(np.mean(subject.coherence_speeds['cathode'][coherence])-np.mean(subject.coherence_speeds['sham - pre - cathode'][coherence]))
            all_anode_speed_diffs.extend(anode_speed_diffs)
            all_cathode_speed_diffs.extend(cathode_speed_diffs)
            mean_anode_speed_diffs.append(np.mean(anode_speed_diffs))
            mean_cathode_speed_diffs.append(np.mean(cathode_speed_diffs))
            furl='img/diff_mean_speed_%s_%.4f' % (suffix,coherence)
            self.urls['coherence_diff_speed_%s' % suffix][coherence]='%s.png' % furl
            if regenerate_plots:
                fname=os.path.join(reports_dir,furl)
                fig=plt.figure()
                speed_hist,speed_bins=np.histogram(np.array(anode_speed_diffs), bins=10)
                bin_width=speed_bins[1]-speed_bins[0]
                bars=plt.bar(speed_bins[:-1],speed_hist/float(len(anode_speed_diffs)),width=bin_width, label='anode')
                for bar in bars:
                    bar.set_color('r')
                speed_hist,speed_bins=np.histogram(np.array(cathode_speed_diffs), bins=10)
                bin_width=speed_bins[1]-speed_bins[0]
                bars=plt.bar(speed_bins[:-1],speed_hist/float(len(cathode_speed_diffs)),width=bin_width, label='cathode')
                for bar in bars:
                    bar.set_color('g')
                plt.legend(loc='best')
                plt.xlabel('Speed Diff')
                plt.ylabel('Proportion of trials')
                plt.title('Coherence=%.4f' % coherence)
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)
            
        furl='img/speed_diff_%s' % suffix
        self.urls['speed_diff_%s' % suffix]='%s.png' % furl
        fname=os.path.join(reports_dir,furl)
        if regenerate_plots:
            fig=plt.figure()
            speed_hist,speed_bins=np.histogram(np.array(all_anode_speed_diffs), bins=10)
            bin_width=speed_bins[1]-speed_bins[0]
            bars=plt.bar(speed_bins[:-1],speed_hist/float(len(all_anode_speed_diffs)),width=bin_width, label='anode')
            for bar in bars:
                bar.set_color('r')
            speed_hist,speed_bins=np.histogram(np.array(all_cathode_speed_diffs), bins=10)
            bin_width=speed_bins[1]-speed_bins[0]
            bars=plt.bar(speed_bins[:-1],speed_hist/float(len(all_cathode_speed_diffs)),width=bin_width, label='cathode')
            for bar in bars:
                bar.set_color('g')
            plt.legend(loc='best')
            plt.xlabel('Speed Diff')
            plt.ylabel('Proportion of trials')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        furl='img/mean_speed_diff_%s' % suffix
        self.urls['mean_speed_diff_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            fname=os.path.join(reports_dir,furl)
            fig=plt.figure()
            speed_hist,speed_bins=np.histogram(np.array(mean_anode_speed_diffs), bins=5)
            bin_width=speed_bins[1]-speed_bins[0]
            bars=plt.bar(speed_bins[:-1],speed_hist/float(len(mean_anode_speed_diffs)),width=bin_width, label='anode')
            for bar in bars:
                bar.set_color('r')
            speed_hist,speed_bins=np.histogram(np.array(mean_cathode_speed_diffs), bins=5)
            bin_width=speed_bins[1]-speed_bins[0]
            bars=plt.bar(speed_bins[:-1],speed_hist/float(len(mean_cathode_speed_diffs)),width=bin_width, label='cathode')
            for bar in bars:
                bar.set_color('g')
            plt.legend(loc='best')
            plt.xlabel('Mean Speed Diff')
            plt.ylabel('Proportion of subjects')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        self.urls['coherence_diff_rt_%s' % suffix]={}
        all_anode_rt_diffs=[]
        all_cathode_rt_diffs=[]
        for coherence in self.all_coherence_levels[self.all_coherence_levels.keys()[0]]:
            anode_rt_diffs=[]
            cathode_rt_diffs=[]
            for subject in self.subjects:
                if not subject.excluded:
                    anode_rt_diffs.append(np.mean(subject.coherence_rts['anode'][coherence])-np.mean(subject.coherence_rts['sham - pre - anode'][coherence]))
                    cathode_rt_diffs.append(np.mean(subject.coherence_rts['cathode'][coherence])-np.mean(subject.coherence_rts['sham - pre - cathode'][coherence]))
            all_anode_rt_diffs.extend(anode_rt_diffs)
            all_cathode_rt_diffs.extend(cathode_rt_diffs)
            furl='img/diff_mean_rt_%s_%.4f' % (suffix,coherence)
            self.urls['coherence_diff_rt_%s' % suffix][coherence]='%s.png' % furl
            if regenerate_plots:
                fname=os.path.join(reports_dir,furl)
                fig=plt.figure()
                rt_hist,rt_bins=np.histogram(np.array(anode_rt_diffs), bins=10)
                bin_width=rt_bins[1]-rt_bins[0]
                bars=plt.bar(rt_bins[:-1],rt_hist/float(len(anode_rt_diffs)),width=bin_width, label='anode')
                for bar in bars:
                    bar.set_color('r')
                rt_hist,rt_bins=np.histogram(np.array(cathode_rt_diffs), bins=10)
                bin_width=rt_bins[1]-rt_bins[0]
                bars=plt.bar(rt_bins[:-1],rt_hist/float(len(cathode_rt_diffs)),width=bin_width, label='cathode')
                for bar in bars:
                    bar.set_color('g')
                plt.legend(loc='best')
                plt.xlabel('RT Diff')
                plt.ylabel('Proportion of trials')
                plt.title('Coherence=%.4f' % coherence)
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)

        furl='img/rt_diff_%s' % suffix
        self.urls['rt_diff_%s' % suffix]='%s.png' % furl
        fname=os.path.join(reports_dir,furl)
        if regenerate_plots:
            fig=plt.figure()
            rt_hist,rt_bins=np.histogram(np.array(all_anode_rt_diffs), bins=10)
            bin_width=rt_bins[1]-rt_bins[0]
            bars=plt.bar(rt_bins[:-1],rt_hist/float(len(all_anode_rt_diffs)),width=bin_width, label='anode')
            for bar in bars:
                bar.set_color('r')
            rt_hist,rt_bins=np.histogram(np.array(all_cathode_rt_diffs), bins=10)
            bin_width=rt_bins[1]-rt_bins[0]
            bars=plt.bar(rt_bins[:-1],rt_hist/float(len(all_cathode_rt_diffs)),width=bin_width, label='cathode')
            for bar in bars:
                bar.set_color('g')
            plt.legend(loc='best')
            plt.xlim([-200,200])
            plt.xlabel('RT Diff')
            plt.ylabel('Proportion of trials')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

        mean_anode_rt_diffs=[]
        mean_cathode_rt_diffs=[]
        for subject in self.subjects:
            if not subject.excluded:
                anode_rt_diffs=[]
                cathode_rt_diffs=[]
                for coherence in self.all_coherence_levels[self.all_coherence_levels.keys()[0]]:
                    anode_rt_diffs.append(np.array(subject.coherence_rts['anode'][coherence])-np.array(subject.coherence_rts['sham - pre - anode'][coherence]))
                    cathode_rt_diffs.append(np.array(subject.coherence_rts['cathode'][coherence])-np.array(subject.coherence_rts['sham - pre - cathode'][coherence]))
                mean_anode_rt_diffs.append(np.mean(anode_rt_diffs))
                mean_cathode_rt_diffs.append(np.mean(cathode_rt_diffs))
        furl='img/mean_rt_diff_%s' % suffix
        self.urls['mean_rt_diff_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            fname=os.path.join(reports_dir,furl)
            fig=plt.figure()
            rt_hist,rt_bins=np.histogram(np.array(mean_cathode_rt_diffs), bins=5)
            bin_width=rt_bins[1]-rt_bins[0]
            bars=plt.bar(rt_bins[:-1],rt_hist/float(len(mean_cathode_rt_diffs)),width=bin_width, label='cathode')
            for bar in bars:
                bar.set_color('g')
            rt_hist,rt_bins=np.histogram(np.array(mean_anode_rt_diffs), bins=10)
            bin_width=rt_bins[1]-rt_bins[0]
            bars=plt.bar(rt_bins[:-1],rt_hist/float(len(mean_anode_rt_diffs)),width=bin_width, label='anode')
            for bar in bars:
                bar.set_color('r')
            plt.legend(loc='best')
            #plt.xlim([-40,40])
            plt.xlabel('Mean RT Diff')
            plt.ylabel('Proportion of subjects')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)


        if 'anode' in self.conditions and 'cathode' in self.conditions:
            furl='img/diff_mean_rt_%s' % suffix
            fname=os.path.join(reports_dir, furl)
            self.urls['diff_mean_rt_%s' % suffix]='%s.png' % furl
            if regenerate_plots:
                mean_rt_diff={'anode':[], 'cathode': []}
                std_rt_diff={'anode':[], 'cathode': []}
                for coherence in self.all_coherence_levels['anode']:
                    coherence_mean_rt_diffs={'anode':[], 'cathode': []}
                    for subject in self.subjects:
                        subj_mean_rt_diffs={}
                        if not subject.excluded:
                            for session in subject.sessions:
                                for run in session.runs:
                                    if not run.excluded and (run.condition=='anode' or run.condition=='cathode' or run.condition=='sham - pre - anode' or run.condition=='sham - pre - cathode'):
                                        if not run.condition in subj_mean_rt_diffs:
                                            subj_mean_rt_diffs[run.condition]=[]
                                        for trial in run.trials:
                                            if not trial.excluded and trial.coherence==coherence and trial.response>0:
                                                subj_mean_rt_diffs[run.condition].append(trial.rt)
                            coherence_mean_rt_diffs['anode'].append(np.mean(subj_mean_rt_diffs['anode'])-np.mean(subj_mean_rt_diffs['sham - pre - anode']))
                            coherence_mean_rt_diffs['cathode'].append(np.mean(subj_mean_rt_diffs['cathode'])-np.mean(subj_mean_rt_diffs['sham - pre - cathode']))
                    for condition in coherence_mean_rt_diffs:
                        mean_rt_diff[condition].append(np.mean(coherence_mean_rt_diffs[condition]))
                        std_rt_diff[condition].append(np.std(coherence_mean_rt_diffs[condition])/np.sqrt(len(coherence_mean_rt_diffs[condition])))

                smoothed_mean_rt_diff={
                    'anode': movingaverage(mean_rt_diff['anode'],3),
                    'cathode': movingaverage(mean_rt_diff['cathode'],3)
                }
                min_x=self.all_coherence_levels['anode'][1]
                max_x=self.all_coherence_levels['anode'][-1]
                fig=plt.figure()
                clf = LinearRegression()
                clf.fit(np.reshape(np.array(self.all_coherence_levels['anode'][1:]),
                    (len(self.all_coherence_levels['anode'][1:]),1)), np.reshape(np.array(smoothed_mean_rt_diff['anode'][1:]),
                    (len(smoothed_mean_rt_diff['anode'][1:]),1)))
                anode_a = clf.coef_[0][0]
                anode_b = clf.intercept_[0]
                anode_r_sqr=clf.score(np.reshape(np.array(self.all_coherence_levels['anode'][1:]),
                    (len(self.all_coherence_levels['anode'][1:]),1)), np.reshape(np.array(smoothed_mean_rt_diff['anode'][1:]),
                    (len(smoothed_mean_rt_diff['anode'][1:]),1)))
                plt.plot([min_x, max_x], [anode_a * min_x + anode_b, anode_a * max_x + anode_b], '--r',
                    label='r^2=%.3f' % anode_r_sqr)
                clf = LinearRegression()
                clf.fit(np.reshape(np.array(self.all_coherence_levels['cathode'][1:]),(len(self.all_coherence_levels['cathode'][1:]),1)),
                    np.reshape(np.array(smoothed_mean_rt_diff['cathode'][1:]),(len(smoothed_mean_rt_diff['cathode'][1:]),1)))
                cathode_a = clf.coef_[0][0]
                cathode_b = clf.intercept_[0]
                cathode_r_sqr=clf.score(np.reshape(np.array(self.all_coherence_levels['cathode'][1:]),
                    (len(self.all_coherence_levels['cathode'][1:]),1)), np.reshape(np.array(smoothed_mean_rt_diff['cathode'][1:]),
                    (len(smoothed_mean_rt_diff['cathode'][1:]),1)))
                plt.plot([min_x, max_x], [cathode_a * min_x + cathode_b, cathode_a * max_x + cathode_b], '--g',
                    label='r^2=%.3f' % cathode_r_sqr)
                plt.errorbar(self.all_coherence_levels['anode'],smoothed_mean_rt_diff['anode'],yerr=std_rt_diff['anode'],fmt='or')
                plt.errorbar(self.all_coherence_levels['cathode'],smoothed_mean_rt_diff['cathode'],yerr=std_rt_diff['cathode'],fmt='og')
                plt.ylim([-50,50])
                plt.legend(loc='best')
                plt.xscale('log')
                plt.xlabel('Coherence')
                plt.ylabel('Decision time (ms)')
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)

            furl='img/diff_mean_speed_%s' % suffix
            fname=os.path.join(reports_dir, furl)
            self.urls['diff_mean_speed_%s' % suffix]='%s.png' % furl
            if regenerate_plots:
                mean_speed_diff={'anode':[], 'cathode': []}
                std_speed_diff={'anode':[], 'cathode': []}
                for coherence in self.all_coherence_levels['anode']:
                    coherence_mean_speed_diffs={'anode':[], 'cathode': []}
                    for subject in self.subjects:
                        subj_mean_speed_diffs={}
                        if not subject.excluded:
                            for session in subject.sessions:
                                for run in session.runs:
                                    if not run.excluded and (run.condition=='anode' or run.condition=='cathode' or run.condition=='sham - pre - anode' or run.condition=='sham - pre - cathode'):
                                        if not run.condition in subj_mean_speed_diffs:
                                            subj_mean_speed_diffs[run.condition]=[]
                                        for trial in run.trials:
                                            if not trial.excluded and trial.coherence==coherence and trial.response>0:
                                                subj_mean_speed_diffs[run.condition].append(trial.speed)
                            coherence_mean_speed_diffs['anode'].append(np.mean(subj_mean_speed_diffs['anode'])-np.mean(subj_mean_speed_diffs['sham - pre - anode']))
                            coherence_mean_speed_diffs['cathode'].append(np.mean(subj_mean_speed_diffs['cathode'])-np.mean(subj_mean_speed_diffs['sham - pre - cathode']))
                    for condition in coherence_mean_speed_diffs:
                        mean_speed_diff[condition].append(np.mean(coherence_mean_speed_diffs[condition]))
                        std_speed_diff[condition].append(np.std(coherence_mean_speed_diffs[condition])/float(len(coherence_mean_speed_diffs[condition])))

                fig=plt.figure()
                plt.errorbar(self.all_coherence_levels['anode'],mean_speed_diff['anode'],yerr=std_speed_diff['anode'],fmt='or')
                plt.errorbar(self.all_coherence_levels['cathode'],mean_speed_diff['cathode'],yerr=std_speed_diff['cathode'],fmt='og')
                plt.legend(loc='best')
                plt.xscale('log')
                plt.xlabel('Coherence')
                plt.ylabel('Speed')
                save_to_png(fig, '%s.png' % fname)
                save_to_eps(fig, '%s.eps' % fname)
                plt.close(fig)

        furl='img/mean_perc_correct_%s' % suffix
        self.urls['mean_perc_correct_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            fname=os.path.join(reports_dir, furl)
            fig=plt.figure()
            mean_perc_correct={}
            std_perc_correct={}
            for coherence in self.all_coherence_levels['control'][1:]:
                coherence_perc_correct={}
                for subject in self.subjects:
                    subject_perc_correct={}
                    if not subject.excluded:
                        for session in subject.sessions:
                            for run in session.runs:
                                if not run.excluded and (run.condition=='anode' or run.condition=='cathode' or run.condition=='sham - pre - anode' or run.condition=='sham - pre - cathode'):
                                    if not run.condition in subject_perc_correct:
                                        subject_perc_correct[run.condition]=[]
                                    for trial in run.trials:
                                        if not trial.excluded and trial.coherence==coherence:
                                            subject_perc_correct[run.condition].append(trial.response)
                        for condition in subject_perc_correct:
                            if not condition in coherence_perc_correct:
                                coherence_perc_correct[condition]=[]
                            coherence_perc_correct[condition].append(np.mean(subject_perc_correct[condition]))
                for condition in coherence_perc_correct:
                    if not condition in mean_perc_correct:
                        mean_perc_correct[condition]=[]
                        std_perc_correct[condition]=[]
                    mean_perc_correct[condition].append(np.mean(coherence_perc_correct[condition]))
                    std_perc_correct[condition].append(np.std(coherence_perc_correct[condition])/np.sqrt(len(coherence_perc_correct[condition])))

            for condition in mean_perc_correct:
                fit=data.FitWeibull(self.all_coherence_levels[condition][1:], mean_perc_correct[condition], guess=[0.2, 0.5])
                smoothInt = pylab.arange(0.01, max(self.all_coherence_levels[condition][1:]), 0.001)
                smoothResp = fit.eval(smoothInt)

                plt.errorbar(self.all_coherence_levels[condition][1:], mean_perc_correct[condition],yerr=std_perc_correct[condition],fmt='o%s' % condition_colors[condition])
                plt.plot(smoothInt, smoothResp, '%s%s' % (condition_styles[condition],condition_colors[condition]), label=condition)
                thresh = np.max([0,fit.inverse(0.8)])
                plt.plot([thresh,thresh],[0.4,1.0],'%s%s' % (condition_styles[condition],condition_colors[condition]))
            plt.legend(loc='best')
            plt.xscale('log')
            #plt.ylim([500, 1000])
            plt.xlabel('Coherence')
            plt.ylabel('% Correct')
            save_to_png(fig, '%s.png' % fname)
            save_to_eps(fig, '%s.eps' % fname)
            plt.close(fig)

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
        self.coherent_rts={}
        self.difficult_rts={}
        self.rts={}
        self.speeds={}
        self.coherence_responses={}
        self.coherence_rts={}
        self.coherence_speeds={}
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
        self.rt_mean_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        self.speed_mean_vals={
            'sham - pre - anode': [],
            'sham - pre - cathode': [],
            'anode': [],
            'cathode': []
        }
        self.rt_diff_vals={
            'anode': [],
            'cathode': []
        }
        self.speed_diff_vals={
            'anode': [],
            'cathode': []
        }
        self.num_subjects=0
        self.num_post_filter_subjects=0
        for subject in self.subjects:
            self.num_subjects+=1
            if not subject.excluded:
                self.num_post_filter_subjects+=1
                for condition in self.a_vals:
                    self.a_vals[condition].append(subject.a[condition])
                    self.k_vals[condition].append(subject.k[condition])
                    self.tr_vals[condition].append(subject.tr[condition])
                    self.alpha_vals[condition].append(subject.alpha[condition])
                    self.beta_vals[condition].append(subject.beta[condition])
                    self.thresh_vals[condition].append(subject.thresh[condition])
                    self.perc_correct_vals[condition].append(subject.perc_correct[condition])
                    self.rt_mean_vals[condition].append(np.mean(subject.rts[condition]))
                    self.speed_mean_vals[condition].append(np.mean(subject.speeds[condition]))
                self.rt_diff_vals['anode'].append(np.mean(subject.rts['anode']-np.mean(subject.rts['sham - pre - anode'])))
                self.rt_diff_vals['cathode'].append(np.mean(subject.rts['cathode']-np.mean(subject.rts['sham - pre - cathode'])))
                self.speed_diff_vals['anode'].append(np.mean(subject.speeds['anode']-np.mean(subject.speeds['sham - pre - anode'])))
                self.speed_diff_vals['cathode'].append(np.mean(subject.speeds['cathode']-np.mean(subject.speeds['sham - pre - cathode'])))
                self.aggregate_low_level_stats(subject)
                for session in subject.sessions:
                    for run in session.runs:
                        if not run.excluded:
                            if not run.condition in self.coherent_responses:
                                self.coherent_responses[run.condition]=[]
                                self.coherent_rts[run.condition]=[]
                                self.difficult_rts[run.condition]=[]
                            for idx,trial in enumerate(run.trials):
                                if trial.coherence>0:
                                    self.coherent_responses[run.condition].append(trial.response)
                                    self.coherent_rts[run.condition].append(trial.rt)
                                    all_coherent_responses.append(trial.response)
                                else:
                                    self.difficult_rts[run.condition].append(trial.rt)
                                    all_difficult_rts.append(trial.rt)

        self.all_perc_correct=np.mean(all_coherent_responses)*100.0
        self.all_difficult_mean_rt=np.mean(all_difficult_rts)

        ConditionAggregatedReport.aggregate_condition_stats(self)
                    
    def create_report(self, data_dir, reports_dir, regenerate_plots=True, regenerate_subject_plots=True, regenerate_session_plots=True, regenerate_run_plots=True):
        make_report_dirs(reports_dir)

        for subj_id in subject_sessions:
            subject=Subject(subj_id)
            subject.create_report(data_dir, os.path.join(reports_dir,subj_id), regenerate_plots=regenerate_subject_plots, regenerate_session_plots=regenerate_session_plots,regenerate_run_plots=regenerate_run_plots)
            self.subjects.append(subject)

        self.aggregate_condition_stats()
        self.plot_stats(reports_dir, 'pre', regenerate_plots)
        self.filter()
        self.aggregate_condition_stats()
        self.plot_stats(reports_dir, 'post', regenerate_plots)

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
        self.rt_means_mean={}
        self.rt_stats={}
        self.speed_means_mean={}
        self.speed_stats={}
        self.rt_diff_mean={}
        self.rt_diff_stats={}
        self.speed_diff_mean={}
        self.speed_diff_stats={}
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

            self.rt_mean_vals[condition]=np.array(self.rt_mean_vals[condition])
            self.rt_means_mean[condition]=np.mean(self.rt_mean_vals[condition])
            self.rt_stats[condition]=shapiro(self.rt_mean_vals[condition])

            self.speed_mean_vals[condition]=np.array(self.speed_mean_vals[condition])
            self.speed_means_mean[condition]=np.mean(self.speed_mean_vals[condition])
            self.speed_stats[condition]=shapiro(self.speed_mean_vals[condition])

        self.rt_diff_vals['anode']=np.array(self.rt_diff_vals['anode'])
        self.rt_diff_mean['anode']=np.mean(self.rt_diff_vals['anode'])
        self.rt_diff_stats['anode']=shapiro(self.rt_diff_vals['anode'])

        self.rt_diff_vals['cathode']=np.array(self.rt_diff_vals['cathode'])
        self.rt_diff_mean['cathode']=np.mean(self.rt_diff_vals['cathode'])
        self.rt_diff_stats['cathode']=shapiro(self.rt_diff_vals['cathode'])
        
        self.speed_diff_vals['anode']=np.array(self.speed_diff_vals['anode'])
        self.speed_diff_mean['anode']=np.mean(self.speed_diff_vals['anode'])
        self.speed_diff_stats['anode']=shapiro(self.speed_diff_vals['anode'])

        self.speed_diff_vals['cathode']=np.array(self.speed_diff_vals['cathode'])
        self.speed_diff_mean['cathode']=np.mean(self.speed_diff_vals['cathode'])
        self.speed_diff_stats['cathode']=shapiro(self.speed_diff_vals['cathode'])


        self.anode_perc_correct_stats=ttest_rel(self.perc_correct['anode'],self.perc_correct['sham - pre - anode'])
        self.cathode_perc_correct_stats=ttest_rel(self.perc_correct['cathode'],self.perc_correct['sham - pre - cathode'])
        
        self.anode_alpha_stats=ttest_rel(self.alpha_vals['anode'],self.alpha_vals['sham - pre - anode'])
        self.cathode_alpha_stats=ttest_rel(self.alpha_vals['cathode'],self.alpha_vals['sham - pre - cathode'])

        self.anode_beta_stats=wilcoxon(self.beta_vals['anode'],self.beta_vals['sham - pre - anode'])
        self.cathode_beta_stats=wilcoxon(self.beta_vals['cathode'],self.beta_vals['sham - pre - cathode'])

        self.anode_thresh_stats=ttest_rel(self.thresh_vals['anode'],self.thresh_vals['sham - pre - anode'])
        self.cathode_thresh_stats=ttest_rel(self.thresh_vals['cathode'],self.thresh_vals['sham - pre - cathode'])

        self.anode_rt_stats=ttest_rel(self.rt_mean_vals['anode'], self.rt_mean_vals['sham - pre - anode'])
        self.cathode_rt_stats=ttest_rel(self.rt_mean_vals['cathode'], self.rt_mean_vals['sham - pre - cathode'])

        self.anode_speed_stats=ttest_rel(self.speed_mean_vals['anode'], self.speed_mean_vals['sham - pre - anode'])
        self.cathode_speed_stats=ttest_rel(self.speed_mean_vals['cathode'], self.speed_mean_vals['sham - pre - cathode'])

        self.anode_a_stats=wilcoxon(self.a_vals['anode'],self.a_vals['sham - pre - anode'])
        self.cathode_a_stats=wilcoxon(self.a_vals['cathode'],self.a_vals['sham - pre - cathode'])

        self.anode_k_stats=ttest_rel(self.k_vals['anode'],self.k_vals['sham - pre - anode'])
        self.cathode_k_stats=ttest_rel(self.k_vals['cathode'],self.k_vals['sham - pre - cathode'])
        
        self.anode_tr_stats=ttest_rel(self.tr_vals['anode'],self.tr_vals['sham - pre - anode'])
        self.cathode_tr_stats=ttest_rel(self.tr_vals['cathode'],self.tr_vals['sham - pre - cathode'])

        anode_rt_groups=[]
        anode_speed_groups=[]
        anode_perc_correct_groups=[]
        for condition in ['anode','sham - pre - anode']:
            condition_rts=[]
            condition_speeds=[]
            condition_perc_correct=[]
            for coherence in self.all_coherence_levels['anode']:
                coherence_rts=[]
                coherence_speeds=[]
                coherence_perc_correct=[]
                for subject in self.subjects:
                    if not subject.excluded:
                        coherence_rts.append(np.mean(subject.coherence_rts[condition][coherence]))
                        coherence_speeds.append(np.mean(subject.coherence_speeds[condition][coherence]))
                        coherence_perc_correct.append(np.mean(subject.coherence_responses[condition][coherence])*100.0)
                condition_rts.append(coherence_rts)
                condition_speeds.append(coherence_speeds)
                condition_perc_correct.append(coherence_perc_correct)
            anode_rt_groups.append(condition_rts)
            anode_speed_groups.append(condition_speeds)
            anode_perc_correct_groups.append(condition_perc_correct)
        
        self.anode_rt_mean_anova = twoway_interaction(anode_rt_groups, 'condition', 'coherence', "html")
        self.anode_speed_mean_anova = twoway_interaction(anode_speed_groups, 'condition', 'coherence', 'html')
        self.anode_perc_correct_anova = twoway_interaction(anode_perc_correct_groups, 'condition', 'coherence', 'html')

        cathode_rt_groups=[]
        cathode_speed_groups=[]
        cathode_perc_correct_groups=[]
        for condition in ['cathode','sham - pre - cathode']:
            condition_rts=[]
            condition_speeds=[]
            condition_perc_correct=[]
            for coherence in self.all_coherence_levels['cathode']:
                coherence_rts=[]
                coherence_speeds=[]
                coherence_perc_correct=[]
                for subject in self.subjects:
                    if not subject.excluded:
                        coherence_rts.append(np.mean(subject.coherence_rts[condition][coherence]))
                        coherence_speeds.append(np.mean(subject.coherence_speeds[condition][coherence]))
                        coherence_perc_correct.append(np.mean(subject.coherence_responses[condition][coherence])*100.0)
                condition_rts.append(coherence_rts)
                condition_speeds.append(coherence_speeds)
                condition_perc_correct.append(coherence_perc_correct)
            cathode_rt_groups.append(condition_rts)
            cathode_speed_groups.append(condition_speeds)
            cathode_perc_correct_groups.append(condition_perc_correct)
        self.cathode_rt_mean_anova = twoway_interaction(cathode_rt_groups, 'condition', 'coherence', "html")
        self.cathode_speed_mean_anova = twoway_interaction(cathode_speed_groups, 'condition', 'coherence', 'html')
        self.cathode_perc_correct_anova = twoway_interaction(cathode_perc_correct_groups, 'condition', 'coherence', 'html')


        rt_diff_groups=[]
        speed_diff_groups=[]

        anode_rt_diffs=[]
        anode_speed_diffs=[]
        for idx,coherence in enumerate(self.all_coherence_levels['anode']):
            coherence_rt_diffs=[]
            coherence_speed_diffs=[]
            for subject in self.subjects:
                if not subject.excluded:
                    #smoothed_mean_rt_diff=movingaverage(np.array(subject.mean_rt['anode'])-np.array(subject.mean_rt['sham - pre - anode']),3)
                    #coherence_rt_diffs.append(smoothed_mean_rt_diff[idx])
                    coherence_rt_diffs.append(np.mean(subject.coherence_rts['anode'][coherence])-np.mean(subject.coherence_rts['sham - pre - anode'][coherence]))
                    coherence_speed_diffs.append(np.mean(subject.coherence_speeds['anode'][coherence])-np.mean(subject.coherence_speeds['sham - pre - anode'][coherence]))
            anode_rt_diffs.append(coherence_rt_diffs)
            anode_speed_diffs.append(coherence_speed_diffs)
        rt_diff_groups.append(anode_rt_diffs)
        speed_diff_groups.append(anode_speed_diffs)

        cathode_rt_diffs=[]
        cathode_speed_diffs=[]
        for idx,coherence in enumerate(self.all_coherence_levels['cathode']):
            coherence_rt_diffs=[]
            coherence_speed_diffs=[]
            for subject in self.subjects:
                if not subject.excluded:
                    #smoothed_mean_rt_diff=movingaverage(np.array(subject.mean_rt['cathode'])-np.array(subject.mean_rt['sham - pre - cathode']),3)
                    #coherence_rt_diffs.append(smoothed_mean_rt_diff[idx])
                    coherence_rt_diffs.append(np.mean(subject.coherence_rts['cathode'][coherence])-np.mean(subject.coherence_rts['sham - pre - cathode'][coherence]))
                    coherence_speed_diffs.append(np.mean(subject.coherence_speeds['cathode'][coherence])-np.mean(subject.coherence_speeds['sham - pre - cathode'][coherence]))
            cathode_rt_diffs.append(coherence_rt_diffs)
            cathode_speed_diffs.append(coherence_speed_diffs)
        rt_diff_groups.append(cathode_rt_diffs)
        speed_diff_groups.append(cathode_speed_diffs)

        self.rt_diff_anova=twoway_interaction(rt_diff_groups, 'condition', 'coherence','html')
        self.speed_diff_anova=twoway_interaction(speed_diff_groups, 'condition', 'coherence','html')


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
                        self.coherent_rts[run.condition]=[]
                        self.difficult_rts[run.condition]=[]
                    for idx,trial in enumerate(run.trials):
                        if trial.coherence>0:
                            self.coherent_responses[run.condition].append(trial.response)
                            self.coherent_rts[run.condition].append(trial.rt)
                            all_coherent_responses.append(trial.response)
                        else:
                            self.difficult_rts[run.condition].append(trial.rt)
                            all_difficult_rts.append(trial.rt)

        self.all_perc_correct=np.mean(all_coherent_responses)*100.0
        self.all_difficult_mean_rt=np.mean(all_difficult_rts)

        ConditionAggregatedReport.aggregate_condition_stats(self)

    def plot_stats(self, report_dir, suffix, regenerate_plots):

        ConditionAggregatedReport.plot_stats(self, report_dir, suffix, regenerate_plots)

        linestyles=['-','--','.']
        colors=['r','g','b']

        session_rt_fits=[]
        session_acc_fits=[]
        session_all_coherence_levels=[]
        session_mean_rts=[]
        session_std_rts=[]
        session_mean_speeds=[]
        session_std_speeds=[]
        session_perc_correct_coherence_levels=[]
        session_perc_correct=[]
        session_rts=[]
        session_speeds=[]
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
                    session_mean_speeds.append(run.mean_speed)
                    session_std_speeds.append(run.std_speed)
                    session_perc_correct_coherence_levels.append(run.all_coherence_levels[1:])
                    session_perc_correct.append(run.coherence_perc_correct)
                    rts=[]
                    speeds=[]
                    for run in session.runs:
                        rts.extend(run.rts)
                        speeds.extend(run.speeds)
                    session_rts.append(rts)
                    session_speeds.append(speeds)
                    session_colors.append(colors[run_idx])
                    session_styles.append(linestyles[session_idx])
                    session_labels.append('session %d, run %d - %s' % ((session_idx+1),(run_idx+1),run.condition))
                    session_alphas.append(1)

        furl='img/rt_%s' % suffix
        self.urls['rt_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_coherence_rt(furl, report_dir, session_rt_fits, session_all_coherence_levels, session_mean_rts,
                session_std_rts, session_colors, session_styles, session_labels)

        furl='img/perc_correct_%s' % suffix
        self.urls['perc_correct_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_coherence_perc_correct(furl, report_dir, session_acc_fits, session_perc_correct_coherence_levels,
                session_perc_correct, session_colors, session_styles, session_labels)

        furl='img/rt_dist_%s' % suffix
        self.urls['rt_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_rt_dist(furl, report_dir, session_rts, session_colors, session_alphas, session_labels)

        furl='img/speed_dist_%s' % suffix
        self.urls['speed_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_speed_dist(furl, report_dir, session_speeds, session_colors, session_alphas, session_labels)


    def create_report(self, data_dir, report_dir, regenerate_plots=True, regenerate_session_plots=True, regenerate_run_plots=True):
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
            session.create_report(data_dir, session_report_dir, regenerate_plots=regenerate_session_plots, regenerate_run_plots=regenerate_run_plots)

        self.aggregate_condition_stats()

        self.plot_stats(report_dir, 'post', regenerate_plots)

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
                    self.coherence_speeds[run.condition]={}
                if not run.condition in self.coherent_responses:
                    self.coherent_responses[run.condition]=[]
                    self.coherent_rts[run.condition]=[]
                    self.difficult_rts[run.condition]=[]
                if not run.condition in self.rts:
                    self.rts[run.condition]=[]
                    self.speeds[run.condition]=[]
                self.rts[run.condition].extend(run.rts)
                self.speeds[run.condition].extend(run.speeds)
                for trial in run.trials:
                    if not trial.excluded:
                        if not trial.coherence in self.coherence_responses[run.condition]:
                            self.coherence_responses[run.condition][trial.coherence]=[]
                        self.coherence_responses[run.condition][trial.coherence].append(trial.response)
                        if trial.coherence==0 or trial.response:
                            if not trial.coherence in self.coherence_rts[run.condition]:
                                self.coherence_rts[run.condition][trial.coherence]=[]
                                self.coherence_speeds[run.condition][trial.coherence]=[]
                            self.coherence_rts[run.condition][trial.coherence].append(trial.rt)
                            self.coherence_speeds[run.condition][trial.coherence].append(trial.speed)
                        if trial.coherence>0:
                            self.coherent_responses[run.condition].append(trial.response)
                            self.coherent_rts[run.condition].append(trial.rt)
                            all_coherent_responses.append(trial.response)
                        else:
                            self.difficult_rts[run.condition].append(trial.rt)
                            all_difficult_rts.append(trial.rt)
        self.all_perc_correct=np.mean(all_coherent_responses)*100.0
        self.all_difficult_mean_rt=np.mean(all_difficult_rts)

        ConditionAggregatedReport.aggregate_condition_stats(self)

    def plot_stats(self, report_dir, suffix, regenerate_plots):
        ConditionAggregatedReport.plot_stats(self, report_dir, suffix, regenerate_plots)

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
        run_speeds=[]
        run_labels=[]

        for idx,run in enumerate(self.runs):
            if not run.excluded:
                run_acc_fits.append(run.acc_fit)
                run_rt_fits.append(run.rt_fit)
                run_all_coherence_levels.append(run.all_coherence_levels)
                run_perc_correct_coherence_levels.append(run.all_coherence_levels[1:])
                run_perc_correct.append(run.coherence_perc_correct)
                run_rts.append(run.rts)
                run_speeds.append(run.speeds)
                run_mean_rts.append(run.mean_rt)
                run_std_rts.append(run.std_rt)
                run_labels.append('run %d - %s' % ((idx+1),run.condition))

        furl='img/rt_%s' % suffix
        self.urls['rt_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_coherence_rt(furl, report_dir, run_rt_fits, run_all_coherence_levels, run_mean_rts, run_std_rts, colors,
                styles, run_labels)

        furl='img/perc_correct_%s' % suffix
        self.urls['perc_correct_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_coherence_perc_correct(furl, report_dir, run_acc_fits, run_perc_correct_coherence_levels, run_perc_correct,
                colors, styles, run_labels)

        furl='img/rt_dist_%s' % suffix
        self.urls['rt_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_rt_dist(furl, report_dir, run_rts, colors, alphas, run_labels)

        furl='img/speed_dist_%s' % suffix
        self.urls['speed_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_speed_dist(furl, report_dir, run_speeds, colors, alphas, run_labels)


    def create_report(self, data_dir, report_dir, regenerate_plots=True, regenerate_run_plots=True):
        make_report_dirs(report_dir)
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith('.csv'):
                file_name_parts=file_name.split('.')
                if file_name_parts[0].upper()==self.subj_id:
                    session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')
                    if session_date==self.date and not file_name_parts[2]=='training':
                        run_num=int(file_name_parts[2])
                        run=Run(self.subj_id, self.idx, self.date, run_num, subject_sessions[self.subj_id][self.idx-1][run_num-1])
                        run_report_dir=os.path.join(report_dir,str(run_num))
                        run.create_report(data_dir, file_name, run_report_dir, regenerate_plots=regenerate_run_plots)
                        if run.thresh>1.0:
                            run.excluded=True
                        self.runs.append(run)
        self.runs=sorted(self.runs, key=lambda x: x.run_num)

        for run in self.runs:
            if not run.condition in self.conditions:
                self.conditions.append(run.condition)

        self.aggregate_condition_stats()
        self.plot_stats(report_dir, 'post', regenerate_plots)

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
            if idx<len(self.trials)/3.0:
                trial.excluded=True
            elif trial.response:
                if not trial.coherence in coherence_trial_idx:
                    coherence_trial_idx[trial.coherence]=[]
                    coherence_rts[trial.coherence]=[]
                coherence_trial_idx[trial.coherence].append(idx)
                coherence_rts[trial.coherence].append(trial.rt)
            else:
                trial.excluded=False

        for coherence,rts in coherence_rts.iteritems():
            outliers=mdm_outliers(np.array(rts))
            #outliers=sd_outliers(np.array(rts),1.0)
            for outlier_idx in outliers:
                self.trials[coherence_trial_idx[coherence][outlier_idx]].excluded=True
            for idx,rt in enumerate(rts):
                if rt<300:
                    self.trials[coherence_trial_idx[coherence][idx]].excluded=True
                #elif rt>900:
                #    self.trials[coherence_trial_idx[coherence][idx]].excluded=True
#            new_rts=[]
#            new_rts_idx=[]
#            for i in range(len(rts)):
#                if not i in outliers:
#                    new_rts.append(rts[i])
#                    new_rts_idx.append(i)
#            outliers=mdm_outliers(np.array(new_rts))
#            #outliers=two_sd_outliers(np.array(rts))
#            for outlier_idx in outliers:
#                self.trials[coherence_trial_idx[coherence][new_rts_idx[outlier_idx]]].excluded=True


    def aggregate_stats(self):
        # Compute % correct and mean RT in 0% coherence trials
        self.coherent_responses=[]
        self.coherent_rts=[]
        difficult_rts=[]
        for trial in self.trials:
            if not trial.excluded:
                if trial.coherence>0:
                    self.coherent_responses.append(trial.response)
                    self.coherent_rts.append(trial.rt)
                else:
                    difficult_rts.append(trial.rt)
        self.perc_correct=np.mean(self.coherent_responses)*100.0
        self.difficult_mean_rt=np.mean(difficult_rts)

        # All RTs
        self.rts=[]
        self.speeds=[]
        for trial in self.trials:
            if not trial.excluded and (trial.coherence==0 or trial.response):
                self.rts.append(trial.rt)
                self.speeds.append(trial.speed)

        self.coherence_responses={}
        self.coherence_rts={}
        self.coherence_speeds={}
        for trial in self.trials:
            if not trial.excluded:
                if not trial.coherence in self.coherence_responses:
                    self.coherence_responses[trial.coherence]=[]
                self.coherence_responses[trial.coherence].append(trial.response)
                if trial.coherence==0 or trial.response:
                    if not trial.coherence in self.coherence_rts:
                        self.coherence_rts[trial.coherence]=[]
                        self.coherence_speeds[trial.coherence]=[]
                    self.coherence_rts[trial.coherence].append(trial.rt)
                    self.coherence_speeds[trial.coherence].append(trial.speed)

        self.all_coherence_levels=sorted(self.coherence_responses.keys())

        self.mean_rt=[]
        self.std_rt=[]
        self.mean_speed=[]
        self.std_speed=[]
        self.coherence_perc_correct=[]
        for coherence in self.all_coherence_levels:
            self.mean_rt.append(np.mean(self.coherence_rts[coherence]))
            self.std_rt.append(np.std(self.coherence_rts[coherence])/float(np.sqrt(len(self.coherence_rts[coherence]))))
            self.mean_speed.append(np.mean(self.coherence_speeds[coherence]))
            self.std_speed.append(np.std(self.coherence_speeds[coherence])/float(np.sqrt(len(self.coherence_speeds[coherence]))))
            if coherence>0:
                self.coherence_perc_correct.append(np.mean(self.coherence_responses[coherence]))

    def plot_stats(self, report_dir, suffix, regenerate_plots):
        furl='img/rt_%s' % suffix
        self.urls['rt_%s' % suffix]='%s.png' % furl
        self.rt_fit = FitRT(self.all_coherence_levels, self.mean_rt, guess=[1,1,1])
        if regenerate_plots:
            plot_coherence_rt(furl, report_dir, [self.rt_fit], [self.all_coherence_levels], [self.mean_rt], [self.std_rt],
                ['k'],['-'],[None])
        self.a=self.rt_fit.params[0]
        self.k=self.rt_fit.params[1]
        self.tr=self.rt_fit.params[2]

        furl='img/perc_correct_%s' % suffix
        self.urls['perc_correct_%s' % suffix]='%s.png' % furl
        self.acc_fit = data.FitWeibull(self.all_coherence_levels[1:], self.coherence_perc_correct, guess=[0.2, 0.5])
        if regenerate_plots:
            plot_coherence_perc_correct(furl, report_dir, [self.acc_fit], [self.all_coherence_levels[1:]],
                [self.coherence_perc_correct], ['k'], ['-'], [None])
        self.alpha=self.acc_fit.params[0]
        self.beta=self.acc_fit.params[1]
        self.thresh = np.max([0,self.acc_fit.inverse(0.8)])

        furl='img/rt_dist_%s' % suffix
        self.urls['rt_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_rt_dist(furl, report_dir, [self.rts],['b'],[1],[None])

        furl='img/speed_dist_%s' % suffix
        self.urls['speed_dist_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_speed_dist(furl, report_dir, [self.speeds],['b'],[1],[None])

        self.urls['coherence_rt_dist_%s' % suffix]={}
        for coherence,rts in self.coherence_rts.iteritems():
            furl='img/rt_dist_%0.4f_%s' % (coherence,suffix)
            self.urls['coherence_rt_dist_%s' % suffix][coherence]='%s.png' % furl
            if regenerate_plots:
                plot_coherence_rt_dist(furl, report_dir, coherence, [rts], ['b'], [1], [None])

        furl='img/sat_%s' % suffix
        self.urls['sat_%s' % suffix]='%s.png' % furl
        if regenerate_plots:
            plot_sat(furl, report_dir, [self.coherent_responses], [self.coherent_rts], ['b'], [None])

        self.urls['coherence_sat_%s' % suffix]={}
        for coherence, rts in self.coherence_rts.iteritems():
            if coherence>0:
                furl='img/coherence_sat_%.04f_%s' % (coherence,suffix)
                self.urls['coherence_sat_%s' % suffix][coherence]='%s.png' % furl
                if regenerate_plots:
                    plot_coherence_sat(furl, report_dir, coherence, [rts], [self.coherence_responses[coherence]], ['b'], [None])

    def create_report(self, data_dir, file_name, report_dir, regenerate_plots=True):
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

        self.plot_stats(report_dir, 'pre', regenerate_plots)

        self.filter()

        self.aggregate_stats()

        self.plot_stats(report_dir, 'post', regenerate_plots)

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
        self.speed=1.0/rt
        self.excluded=False


def run_plasticity_analysis(data_dir):
    run_accuracy=np.zeros((len(subject_sessions),3))
    for subj_idx,subj_id in enumerate(subject_sessions):
        session_dict={}
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith('.csv'):
                file_name_parts=file_name.split('.')
                if file_name_parts[0].upper()==subj_id:
                    session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')
                    if not session_date in session_dict:
                        session_dict[session_date]=Session(subj_id, session_date)
        sessions = sorted(session_dict.values(), key=lambda x: x.date)
        first_session=sessions[0]
        for file_name in os.listdir(data_dir):
            if file_name.lower().endswith('.csv'):
                file_name_parts=file_name.split('.')
                if file_name_parts[0].upper()==subj_id:
                    session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')
                    if session_date==first_session.date and not file_name_parts[2]=='training':
                        run_responses=[]
                        run_num=int(file_name_parts[2])
                        file=open(os.path.join(data_dir,file_name),'r')
                        for idx,line in enumerate(file):
                            if idx>0:
                                cols=line.split(',')
                                correct=float(cols[2])
                                run_responses.append(correct)
                        run_accuracy[subj_idx,run_num-1]=np.mean(run_responses)
    mean_accuracy=np.mean(run_accuracy,axis=0)
    stderr_accuracy=np.std(run_accuracy,axis=0)/np.sqrt(len(subject_sessions))
    for i in range(3):
        print('Run %d = %.4f +/- %.4f' % ((i+1), mean_accuracy[i],stderr_accuracy[i]))

def run_analysis(data_dir, reports_dir):
    experiment=Experiment('Random dot motion discrimination task')
    experiment.create_report(data_dir, reports_dir, regenerate_plots=True, regenerate_subject_plots=False,
        regenerate_session_plots=False, regenerate_run_plots=False)
    experiment.export_csv(reports_dir, 'behavioral_data.csv')


def plot_coherence_perc_correct(furl, report_dir, acc_fit_list, coherence_levels_list, perc_correct_list, colors,
                                styles, labels):
    fname=os.path.join(report_dir, furl)
    fig=plt.figure()

    for acc_fit, coherence_levels, perc_correct, color, style, label in zip(acc_fit_list, coherence_levels_list,
        perc_correct_list, colors, styles, labels):
        # Fit Weibull function
        thresh = np.max([0,acc_fit.inverse(0.8)])

        smoothInt = pylab.arange(0.01, max(coherence_levels), 0.001)
        smoothResp = acc_fit.eval(smoothInt)

        plt.plot(smoothInt, smoothResp, '%s%s' % (style,color), label=label)
        plt.plot(coherence_levels, perc_correct, 'o%s' % color)
        plt.plot([thresh,thresh],[0.4,1.0],'%s%s' % (style,color))
    plt.ylim([0.4,1])
    plt.xscale('log')
    plt.legend(loc='best')
    plt.xlabel('Coherence')
    plt.ylabel('% Correct')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
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
    #plt.ylim([450, 650])
    plt.xlabel('Coherence')
    plt.ylabel('Decision time (ms)')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_rt_dist(furl, report_dir, rts_list, colors, alphas, labels):
    fname=os.path.join(report_dir,furl)
    fig=plt.figure()

    for rts, color, alpha, label in zip(rts_list,colors,alphas,labels):
        rt_hist,rt_bins=np.histogram(np.array(rts), bins=10)
        bin_width=rt_bins[1]-rt_bins[0]
        bars=plt.bar(rt_bins[:-1], rt_hist/float(len(rts)), width=bin_width, label=label)
        for bar in bars:
            bar.set_color(color)
            bar.set_alpha(alpha)
    plt.legend(loc='best')
    plt.xlabel('RT')
    plt.ylabel('Proportion of trials')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_speed_dist(furl, report_dir, speeds_list, colors, alphas, labels):
    fname=os.path.join(report_dir,furl)
    fig=plt.figure()

    for rts, color, alpha, label in zip(speeds_list,colors,alphas,labels):
        rt_hist,rt_bins=np.histogram(np.array(rts), bins=10)
        bin_width=rt_bins[1]-rt_bins[0]
        bars=plt.bar(rt_bins[:-1], rt_hist/float(len(rts)), width=bin_width, label=label)
        for bar in bars:
            bar.set_color(color)
            bar.set_alpha(alpha)
    plt.legend(loc='best')
    plt.xlabel('Speed')
    plt.ylabel('Proportion of trials')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_coherence_rt_dist(furl, report_dir, coherence, rts_list, colors, alphas, labels):
    fname=os.path.join(report_dir,furl)
    fig=plt.figure()
    for rts, color, alpha, label in zip(rts_list, colors, alphas, labels):
        rt_hist,rt_bins=np.histogram(np.array(rts), bins=10)
        bin_width=rt_bins[1]-rt_bins[0]
        bars=plt.bar(rt_bins[:-1],rt_hist/float(len(rts)),width=bin_width, label=label)
        for bar in bars:
            bar.set_color(color)
            bar.set_alpha(alpha)
    plt.legend(loc='best')
    plt.xlabel('RT')
    plt.ylabel('Proportion of trials')
    plt.title('Coherence=%.4f' % coherence)
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_coherence_speed_dist(furl, report_dir, coherence, speeds_list, colors, alphas, labels):
    fname=os.path.join(report_dir,furl)
    fig=plt.figure()
    for speeds, color, alpha, label in zip(speeds_list, colors, alphas, labels):
        speed_hist,speed_bins=np.histogram(np.array(speeds), bins=10)
        bin_width=speed_bins[1]-speed_bins[0]
        bars=plt.bar(speed_bins[:-1],speed_hist/float(len(speeds)),width=bin_width, label=label)
        for bar in bars:
            bar.set_color(color)
            bar.set_alpha(alpha)
    plt.legend(loc='best')
    plt.xlabel('Speed')
    plt.ylabel('Proportion of trials')
    plt.title('Coherence=%.4f' % coherence)
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_param_dist(furl, report_dir, vals, param_name):
    fname=os.path.join(report_dir,furl)
    fig=plt.figure()
    for condition in vals:
        rt_hist,rt_bins=np.histogram(np.array(vals[condition]), bins=10)
        bin_width=rt_bins[1]-rt_bins[0]
        bars=plt.bar(rt_bins[:-1], rt_hist/float(len(vals[condition])), width=bin_width, label=condition)
        for bar in bars:
            bar.set_color(condition_colors[condition])
            bar.set_alpha(condition_alphas[condition])
    plt.legend(loc='best')
    plt.xlabel(param_name)
    plt.ylabel('Proportion of subjects')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_sat(furl, report_dir, coherent_responses_list, coherent_rts_list, colors, labels):
    fname=os.path.join(report_dir, furl)
    fig=plt.figure()
    for coherent_responses, coherent_rts, color, label in zip(coherent_responses_list, coherent_rts_list, colors, labels):
        resp_array=np.array(coherent_responses)
        rt_array=np.array(coherent_rts)
        hist,bins=np.histogram(rt_array, bins=10)
        plot_rts=[]
        plot_perc_correct=[]
        for i in range(10):
            trials=np.where((rt_array>=bins[i]) & (rt_array<bins[i+1]))[0]
            plot_rts.append(bins[i]+.5*(bins[i+1]-bins[i]))
            plot_perc_correct.append((1.0-np.mean(resp_array[trials])))

        plt.plot(plot_rts, plot_perc_correct, 'o%s' % color, label=label)
    plt.legend(loc='best')
    #plt.ylim([500, 1000])
    plt.xlabel('RT')
    plt.ylabel('Error Rate')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_coherence_sat(furl, report_dir, coherence, rts_list, coherence_responses_list, colors, labels):
    fname=os.path.join(report_dir, furl)
    fig=plt.figure()
    for rts, coherence_responses, color, label in zip(rts_list,coherence_responses_list,colors,labels):
        rts=np.array(rts)
        responses=np.array(coherence_responses)
        hist,bins=np.histogram(rts, bins=5)
        plot_rts=[]
        plot_perc_correct=[]
        for i in range(5):
            trials=np.where((rts>=bins[i]) & (rts<bins[i+1]))[0]
            plot_rts.append(bins[i]+.5*(bins[i+1]-bins[i]))
            plot_perc_correct.append((1.0-np.mean(responses[trials])))

        plt.plot(plot_rts, plot_perc_correct, 'o%s' % color, label=label)
    plt.legend(loc='best')
    #plt.ylim([500, 1000])
    plt.xlabel('RT')
    plt.ylabel('Error Rate')
    plt.title('Coherence=%.4f' % coherence)
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_diff_mean_rt(furl, report_dir, coherence, mean_rt, std_rt):
    fname=os.path.join(report_dir, furl)
    mean_rt_diff={
        'anode': np.array(mean_rt['anode'])-np.array(mean_rt['sham - pre - anode']),
        'cathode':np.array(mean_rt['cathode'])-np.array(mean_rt['sham - pre - cathode'])
    }
    smoothed_mean_rt_diff={
        'anode': movingaverage(mean_rt_diff['anode'],3),
        'cathode': movingaverage(mean_rt_diff['cathode'],3)
    }
    min_x=coherence['anode'][1]
    max_x=coherence['anode'][-1]
    fig=plt.figure()
    clf = LinearRegression()
    clf.fit(np.reshape(np.array(coherence['anode'][1:]),
        (len(coherence['anode'][1:]),1)), np.reshape(np.array(smoothed_mean_rt_diff['anode'][1:]),
        (len(smoothed_mean_rt_diff['anode'][1:]),1)))
    anode_a = clf.coef_[0][0]
    anode_b = clf.intercept_[0]
    anode_r_sqr=clf.score(np.reshape(np.array(coherence['anode'][1:]),
        (len(coherence['anode'][1:]),1)), np.reshape(np.array(smoothed_mean_rt_diff['anode'][1:]),
        (len(smoothed_mean_rt_diff['anode'][1:]),1)))
    plt.plot([min_x, max_x], [anode_a * min_x + anode_b, anode_a * max_x + anode_b], '--r',
        label='r^2=%.3f' % anode_r_sqr)
    clf = LinearRegression()
    clf.fit(np.reshape(np.array(coherence['cathode'][1:]),(len(coherence['cathode'][1:]),1)),
        np.reshape(np.array(smoothed_mean_rt_diff['cathode'][1:]),(len(smoothed_mean_rt_diff['cathode'][1:]),1)))
    cathode_a = clf.coef_[0][0]
    cathode_b = clf.intercept_[0]
    cathode_r_sqr=clf.score(np.reshape(np.array(coherence['cathode'][1:]),
        (len(coherence['cathode'][1:]),1)), np.reshape(np.array(smoothed_mean_rt_diff['cathode'][1:]),
        (len(smoothed_mean_rt_diff['cathode'][1:]),1)))
    plt.plot([min_x, max_x], [cathode_a * min_x + cathode_b, cathode_a * max_x + cathode_b], '--g',
        label='r^2=%.3f' % cathode_r_sqr)
    plt.errorbar(coherence['anode'],smoothed_mean_rt_diff['anode'],yerr=std_rt['anode'],fmt='or')
    plt.errorbar(coherence['cathode'],smoothed_mean_rt_diff['cathode'],yerr=std_rt['cathode'],fmt='og')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.xlabel('Coherence')
    plt.ylabel('Decision time (ms)')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

def plot_diff_mean_perc_correct(furl, report_dir, coherence, condition_perc_correct):
    fname=os.path.join(report_dir, furl)
    mean_perc_correct_diff={
        'anode': np.array(condition_perc_correct['anode'])-np.array(condition_perc_correct['sham - pre - anode']),
        'cathode': np.array(condition_perc_correct['cathode'])-np.array(condition_perc_correct['sham - pre - cathode']),
    }
    smoothed_mean_perc_correct_diff={
        'anode': movingaverage(mean_perc_correct_diff['anode'],3),
        'cathode': movingaverage(mean_perc_correct_diff['cathode'],3)
    }
    min_x=coherence['anode'][1]
    max_x=coherence['anode'][-1]
    fig=plt.figure()
    clf = LinearRegression()
    clf.fit(np.reshape(np.array(coherence['anode'][1:]),
        (len(coherence['anode'][1:]),1)), np.reshape(np.array(smoothed_mean_perc_correct_diff['anode']),
        (len(smoothed_mean_perc_correct_diff['anode']),1)))
    anode_a = clf.coef_[0][0]
    anode_b = clf.intercept_[0]
    anode_r_sqr=clf.score(np.reshape(np.array(coherence['anode'][1:]),
        (len(coherence['anode'][1:]),1)), np.reshape(np.array(smoothed_mean_perc_correct_diff['anode']),
        (len(smoothed_mean_perc_correct_diff['anode']),1)))
    plt.plot([min_x, max_x], [anode_a * min_x + anode_b, anode_a * max_x + anode_b], '--r',
        label='r^2=%.3f' % anode_r_sqr)
    clf = LinearRegression()
    clf.fit(np.reshape(np.array(coherence['cathode'][1:]),(len(coherence['cathode'][1:]),1)),
        np.reshape(np.array(smoothed_mean_perc_correct_diff['cathode']),(len(smoothed_mean_perc_correct_diff['cathode']),1)))
    cathode_a = clf.coef_[0][0]
    cathode_b = clf.intercept_[0]
    cathode_r_sqr=clf.score(np.reshape(np.array(coherence['cathode'][1:]),
        (len(coherence['cathode'][1:]),1)), np.reshape(np.array(smoothed_mean_perc_correct_diff['cathode']),
        (len(smoothed_mean_perc_correct_diff['cathode']),1)))
    plt.plot([min_x, max_x], [cathode_a * min_x + cathode_b, cathode_a * max_x + cathode_b], '--g',
        label='r^2=%.3f' % cathode_r_sqr)
    plt.errorbar(coherence['anode'][1:],smoothed_mean_perc_correct_diff['anode'],fmt='or')
    plt.errorbar(coherence['cathode'][1:],smoothed_mean_perc_correct_diff['cathode'],fmt='og')
    plt.legend(loc='best')
    plt.xscale('log')
    plt.xlabel('Coherence')
    plt.ylabel('% Correct')
    save_to_png(fig, '%s.png' % fname)
    save_to_eps(fig, '%s.eps' % fname)
    plt.close(fig)

if __name__=='__main__':
    run_analysis('../../data/stim2','../../data/report_thresh_filtering4')
    #run_plasticity_analysis('../../data/stim2')
