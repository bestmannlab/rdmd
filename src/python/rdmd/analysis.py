import os
import numpy as np
from datetime import datetime
from scipy.optimize import curve_fit
from jinja2 import Environment, FileSystemLoader
from psychopy import data
from psychopy.misc import fromFile
import pylab
from rdmd.utils import TEMPLATE_DIR, make_report_dirs, rt_function, save_to_png, weibull, Struct, FitRT
import matplotlib.pyplot as plt

class Condition:
    def __init__(self, name):
        self.name=name
        self.subjects={}

        self.colors={'pre':'g','stim':'b','post':'r'}

        self.perc_correct=0.0
        self.difficult_mean_rt=0.0

        self.alpha_values={}
        self.beta_values={}
        self.a_values={}
        self.k_values={}
        self.tr_values={}

        self.rt_fits={}
        self.acc_fits={}

    def create_report(self, report_dir):
        make_report_dirs(report_dir)

        # Compute % correct and mean RT in 0% coherence trials
        coherent_responses=[]
        difficult_rts=[]
        # All RTs
        rts=[]
        for subj_id, subject in self.subjects.iteritems():
            subj_report_dir=os.path.join(report_dir,subj_id)
            subject.create_report(subj_report_dir)
            for session_id, session in subject.sessions.iteritems():
                for trial in session.trials:
                    rts.append(trial.rt)
                    if trial.coherence>0:
                        coherent_responses.append(trial.response)
                    else:
                        difficult_rts.append(trial.rt)
        self.perc_correct=np.mean(coherent_responses)*100.0
        self.difficult_mean_rt=np.mean(difficult_rts)

        session_coherence_responses={}
        session_coherence_rts={}
        all_coherence_levels=[]
        for subj_id, subject in self.subjects.iteritems():
            for session_id, session in subject.sessions.iteritems():
                if not session in session_coherence_responses:
                    session_coherence_responses[session_id]={}
                if not session in session_coherence_rts:
                    session_coherence_rts[session_id]={}
                for trial in session.trials:
                    if not trial.coherence in all_coherence_levels:
                        all_coherence_levels.append(trial.coherence)
                    if not trial.coherence in session_coherence_responses[session_id]:
                        session_coherence_responses[session_id][trial.coherence]=[]
                    session_coherence_responses[session_id][trial.coherence].append(trial.response)
                    if not trial.coherence in session_coherence_rts[session_id]:
                        session_coherence_rts[session_id][trial.coherence]=[]
                    session_coherence_rts[session_id][trial.coherence].append(trial.rt)

        all_coherence_levels=sorted(all_coherence_levels)

        self.session_mean_rt={}
        self.session_std_rt={}
        self.session_coherence_perc_correct={}
        for session in session_coherence_responses.keys():
            if not session in self.session_mean_rt:
                self.session_mean_rt[session]=[]
                self.session_std_rt[session]=[]
                self.session_coherence_perc_correct[session]=[]
            for coherence in all_coherence_levels:
                self.session_mean_rt[session].append(np.mean(session_coherence_rts[session][coherence]))
                self.session_std_rt[session].append(np.std(session_coherence_rts[session][coherence])/float(len(session_coherence_rts[session][coherence])))
                if coherence>0:
                    self.session_coherence_perc_correct[session].append(np.mean(session_coherence_responses[session][coherence]))

        furl='img/%s_rt' % self.name
        fname=os.path.join(report_dir, furl)
        self.rt_url='%s.png' % furl
        fig=plt.figure()
        for session_type in self.session_mean_rt.keys():
            plt.errorbar(all_coherence_levels,self.session_mean_rt[session_type],yerr=self.session_std_rt[session_type],
                fmt='o%s' % self.colors[session_type], label=session_type)
            self.rt_fits[session_type] = FitRT(all_coherence_levels, self.session_mean_rt[session_type], guess=[1,1,1])
            self.a_values[session_type]=self.rt_fits[session_type].params[0]
            self.k_values[session_type]=self.rt_fits[session_type].params[1]
            self.tr_values[session_type]=self.rt_fits[session_type].params[2]
            smoothInt = pylab.arange(0.01, max(all_coherence_levels), 0.001)
            smoothResp = self.rt_fits[session_type].eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '-%s' % self.colors[session_type])
        plt.legend(loc='best')
        plt.xscale('log')
        plt.ylim([500, 1000])
        plt.xlabel('Coherence')
        plt.ylabel('Decision time (s)')
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        furl='img/%s_perc_correct' % self.name
        fname=os.path.join(report_dir, furl)
        self.perc_correct_url='%s.png' % furl
        fig=plt.figure()
        for session_type in self.session_coherence_perc_correct.keys():
            self.acc_fits[session_type] = data.FitWeibull(all_coherence_levels[1:],
                self.session_coherence_perc_correct[session_type], guess=[0.2, 0.5])
            self.alpha_values[session_type]=self.acc_fits[session_type].params[0]
            self.beta_values[session_type]=self.acc_fits[session_type].params[1]
            smoothInt = pylab.arange(0.0, max(all_coherence_levels[1:]), 0.001)
            smoothResp = self.acc_fits[session_type].eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '-%s' % self.colors[session_type])
            plt.plot(all_coherence_levels[1:], self.session_coherence_perc_correct[session_type],
                'o%s' % self.colors[session_type], label=session_type)
        plt.ylim([0.4,1])
        plt.legend(loc='best')
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        furl='img/%s_rt_dist' % self.name
        fname=os.path.join(report_dir,furl)
        self.rt_dist_url='%s.png' % furl
        fig=plt.figure()
        rt_hist,rt_bins=np.histogram(np.array(rts), bins=10, density=True)
        bin_width=rt_bins[1]-rt_bins[0]
        plt.bar(rt_bins[:-1], rt_hist, width=bin_width)
        plt.xlabel('RT')
        plt.ylabel('Proportion of trials')
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        #create report
        template_file='condition.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='condition_%s.html' % self.name
        fname=os.path.join(report_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

class Subject:
    def __init__(self, id, condition):
        self.id=id
        self.condition=condition
        self.sessions={}
        self.perc_correct=0.0
        self.difficult_mean_rt=0.0
        self.colors={'pre':'g','stim':'b','post':'r'}

    def create_report(self, report_dir):
        make_report_dirs(report_dir)

        # Compute % correct and mean RT in 0% coherence trials
        coherent_responses=[]
        difficult_rts=[]
        # All RTs
        rts=[]
        for session_id, session in self.sessions.iteritems():
            session.create_report(report_dir)
            for trial in session.trials:
                rts.append(trial.rt)
                if trial.coherence>0:
                    coherent_responses.append(trial.response)
                else:
                    difficult_rts.append(trial.rt)
        self.perc_correct=np.mean(coherent_responses)*100.0
        self.difficult_mean_rt=np.mean(difficult_rts)

        furl='img/rt'
        fname=os.path.join(report_dir, furl)
        self.rt_url='%s.png' % furl
        fig=plt.figure()
        for session_type, session in self.sessions.iteritems():
            plt.errorbar(session.all_coherence_levels,session.mean_rt,yerr=session.std_rt,fmt='o%s' % self.colors[session_type],
                label=session_type)
            smoothInt = pylab.arange(0.01, max(session.all_coherence_levels), 0.001)
            smoothResp = session.rt_fit.eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '-%s' % self.colors[session_type])
        plt.legend(loc='best')
        plt.xlabel('Contrast')
        plt.ylabel('Decision time (s)')
        plt.xscale('log')
        plt.ylim([500, 1000])
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        furl='img/perc_correct'
        fname=os.path.join(report_dir, furl)
        self.perc_correct_url='%s.png' % furl
        fig=plt.figure()
        for session_type, session in self.sessions.iteritems():
            plt.plot(session.all_coherence_levels[1:],session.coherence_perc_correct,'%so' % self.colors[session_type])
            smooth_int=pylab.arange(0.0, max(session.all_coherence_levels[1:]), 0.001)
            smooth_resp=session.acc_fit.eval(smooth_int)
            plt.plot(smooth_int,smooth_resp,self.colors[session_type],label=session_type)
        plt.legend(loc='best')
        plt.ylim([0.4, 1])
        plt.xlabel('Coherence')
        plt.ylabel('% correct')
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        furl='img/rt_dist'
        fname=os.path.join(report_dir,furl)
        self.rt_dist_url='%s.png' % furl
        fig=plt.figure()
        rt_hist,rt_bins=np.histogram(np.array(rts), bins=10, density=True)
        bin_width=rt_bins[1]-rt_bins[0]
        plt.bar(rt_bins[:-1], rt_hist, width=bin_width)
        plt.xlabel('RT')
        plt.ylabel('Proportion of trials')
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        #create report
        template_file='subject.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='subject_%s.html' % self.id
        fname=os.path.join(report_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)
        
class Session:
    def __init__(self, subject_id, condition, type, session_time, file_name):
        self.subject_id=subject_id
        self.condition=condition
        self.type=type
        self.session_time=session_time
        self.file_name=file_name

        self.trials=[]

        # % Correct over all trials (except 0% coherence trials)
        self.perc_correct=0.0
        # Mean response time during 0% coherence trials
        self.difficult_mean_rt=0.0

        # accuracy function params
        self.alpha=0
        self.beta=0

    def read_data(self, data_dir):
        file=open(os.path.join(data_dir,self.file_name),'r')
        for idx,line in enumerate(file):
            if idx>0:
                cols=line.split(',')
                direction=int(cols[0])
                coherence=float(cols[1])
                resp=int(cols[2])
                rt=float(cols[3])*1000.0
                trialIdx=idx-1

                self.trials.append(Trial(trialIdx,coherence,resp,rt))

    def create_report(self, reports_dir):
        make_report_dirs(reports_dir)

        # Compute % correct and mean RT in 0% coherence trials
        coherent_responses=[]
        difficult_rts=[]
        for idx,trial in enumerate(self.trials):
            if trial.coherence>0:
                coherent_responses.append(trial.response)
            else:
                difficult_rts.append(trial.rt)
        self.perc_correct=np.mean(coherent_responses)*100.0
        self.difficult_mean_rt=np.mean(difficult_rts)

        # All RTs
        rts=[x.rt for x in self.trials]

        coherence_responses={}
        coherence_rts={}
        for trial in self.trials:
            if not trial.coherence in coherence_responses:
                coherence_responses[trial.coherence]=[]
            coherence_responses[trial.coherence].append(trial.response)
            if not trial.coherence in coherence_rts:
                coherence_rts[trial.coherence]=[]
            coherence_rts[trial.coherence].append(trial.rt)

        self.all_coherence_levels=sorted(coherence_responses.keys())

        self.mean_rt=[]
        self.std_rt=[]
        self.coherence_perc_correct=[]
        for coherence in self.all_coherence_levels:
            self.mean_rt.append(np.mean(coherence_rts[coherence]))
            self.std_rt.append(np.std(coherence_rts[coherence])/float(len(coherence_rts[coherence])))
            if coherence>0:
                self.coherence_perc_correct.append(np.mean(coherence_responses[coherence]))

        furl='img/%s_rt' % self.type
        fname=os.path.join(reports_dir, furl)
        self.rt_url='%s.png' % furl
        fig=plt.figure()
        plt.errorbar(self.all_coherence_levels,self.mean_rt,yerr=self.std_rt,fmt='ok')
        self.rt_fit = FitRT(self.all_coherence_levels, self.mean_rt, guess=[1,1,1])
        self.a=self.rt_fit.params[0]
        self.k=self.rt_fit.params[1]
        self.tr=self.rt_fit.params[2]
        smoothInt = pylab.arange(0.01, max(self.all_coherence_levels), 0.001)
        smoothResp = self.rt_fit.eval(smoothInt)
        plt.plot(smoothInt, smoothResp, '-k')
        plt.xscale('log')
        plt.ylim([500, 1000])
        plt.xlabel('Contrast')
        plt.ylabel('Decision time (s)')
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        furl='img/%s_perc_correct' % self.type
        fname=os.path.join(reports_dir, furl)
        self.perc_correct_url='%s.png' % furl
        fig=plt.figure()
        self.acc_fit = data.FitWeibull(self.all_coherence_levels[1:], self.coherence_perc_correct, guess=[0.2, 0.5])
        self.alpha=self.acc_fit.params[0]
        self.beta=self.acc_fit.params[1]
        smoothInt = pylab.arange(0.0, max(self.all_coherence_levels[1:]), 0.001)
        smoothResp = self.acc_fit.eval(smoothInt)
        self.thresh = self.acc_fit.inverse(0.8)
        plt.plot(smoothInt, smoothResp, '-k')
        plt.plot(self.all_coherence_levels[1:], self.coherence_perc_correct, 'ok')
        plt.ylim([0.4,1])
        plt.xlabel('Coherence')
        plt.ylabel('% Correct')
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        furl='img/%s_rt_dist' % self.type
        fname=os.path.join(reports_dir,furl)
        self.rt_dist_url='%s.png' % furl
        fig=plt.figure()
        rt_hist,rt_bins=np.histogram(np.array(rts), bins=10, density=True)
        bin_width=rt_bins[1]-rt_bins[0]
        plt.bar(rt_bins[:-1], rt_hist, width=bin_width)
        plt.xlabel('RT')
        plt.ylabel('Proportion of trials')
        save_to_png(fig, '%s.png' % fname)
        plt.close(fig)

        #create report
        template_file='subject_session.html'
        env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
        template=env.get_template(template_file)

        self.output_file='subject_%s.session_%s.html' % (self.subject_id,self.type)
        fname=os.path.join(reports_dir,self.output_file)
        stream=template.stream(rinfo=self)
        stream.dump(fname)

class Trial:
    def __init__(self, idx, coherence, response, rt):
        self.idx=idx
        self.coherence=coherence
        self.response=response
        self.rt=rt

#subjects={
#    'YD': Subject('YD','sham'),
#    'BT': Subject('BT','sham'),
#    'RS': Subject('RS','sham'),
#    'JO': Subject('JO','sham'),
#    'ES': Subject('ES','sham'),
#    'JM': Subject('JM','anodal'),
#    'JH': Subject('JH','anodal'),
#    'DP': Subject('DP','anodal'),
#    'HH': Subject('SH','anodal'),
#    'AK': Subject('AK','anodal'),
#    #'CK': Subject('CK','anodal'),
#    'JT': Subject('JT','sham'),
#    'RR': Subject('RR','sham'),
#    'SP': Subject('SP','sham'),
#    'PT': Subject('PT','sham'),
#}
subjects={
    'ME': Subject('ME','anodal'),
    'SM': Subject('SM','anodal'),
    'YD': Subject('YD','sham'),
    'BT': Subject('BT','sham'),
    'AW': Subject('AW','anodal'),
    'RS': Subject('RS','sham'),
    'JO': Subject('JO','sham'),
    'TA': Subject('TA','anodal'),
    'BB': Subject('BB','anodal'),
    'ES': Subject('ES','sham'),
    'JM': Subject('JM','anodal'),
    'JH': Subject('JH','anodal'),
    'DP': Subject('DP','anodal'),
    'HH': Subject('SH','anodal'),
    'AK': Subject('AK','anodal'),
    #'CK': Subject('CK','anodal'),
    'JT': Subject('JT','sham'),
    'RR': Subject('RR','sham'),
    'SP': Subject('SP','sham'),
    'PT': Subject('PT','sham'),
}
#subjects={
#    'JB': Subject('JB','sham')
#}

sessions=['pre','stim','post']

def run_analysis(data_dir, reports_dir):
    make_report_dirs(reports_dir)

    rinfo=Struct()
    rinfo.name='Random dot motion discrimination task'

    rinfo.conditions={}
    for file_name in os.listdir(data_dir):
        if file_name.lower().endswith('.csv'):
            file_name_parts=file_name.split('.')
            subj_id=file_name_parts[0].upper()
            session_time=datetime.strptime(file_name_parts[1],'%Y_%b_%d_%H%M')
            session=file_name_parts[2].lower()

            if subj_id in subjects:
                condition=subjects[subj_id].condition
                if not session=='training':
                    if session in sessions:
                        subjects[subj_id].sessions[session]=Session(subj_id, condition, session, session_time, file_name)
                        subjects[subj_id].sessions[session].read_data(data_dir)
                    else:
                        print('unknown session type: %s' % session)
            else:
                print('%s not found' % subj_id)

    for subj_id, subject in subjects.iteritems():
        if not subject.condition in rinfo.conditions:
            rinfo.conditions[subject.condition]=Condition(subject.condition)
        rinfo.conditions[subject.condition].subjects[subj_id]=subject

    for condition in rinfo.conditions.itervalues():
        condition.create_report(reports_dir)

    line_styles={
        'sham':'-',
        'anodal':'--'
    }

    furl='img/rt'
    fname=os.path.join(reports_dir, furl)
    rinfo.rt_url='%s.png' % furl
    fig=plt.figure()
    for condition_name, condition in rinfo.conditions.iteritems():
        for session_type in condition.session_mean_rt.keys():
            plt.errorbar(condition.all_coherence_levels,condition.session_mean_rt[session_type],
                yerr=condition.session_std_rt[session_type], fmt='o%s' % condition.colors[session_type],
                label='%s - %s' % (condition_name,session_type))
            smoothInt = pylab.arange(0.01, max(condition.all_coherence_levels), 0.001)
            smoothResp = condition.rt_fits[session_type].eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '%s%s' % (line_styles[condition_name],condition.colors[session_type]))
    plt.legend(loc='best')
    plt.xscale('log')
    plt.ylim([500, 1000])
    plt.xlabel('Coherence')
    plt.ylabel('Decision time (s)')
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)

    furl='img/perc_correct'
    fname=os.path.join(reports_dir, furl)
    rinfo.perc_correct_url='%s.png' % furl
    fig=plt.figure()
    for condition_name, condition in rinfo.conditions.iteritems():
        for session_type in condition.session_coherence_perc_correct.keys():
            plt.plot(condition.all_coherence_levels[1:], condition.session_coherence_perc_correct[session_type],
                'o%s' % condition.colors[session_type], label=session_type)
            smoothInt = pylab.arange(0.0, max(condition.all_coherence_levels[1:]), 0.001)
            smoothResp = condition.acc_fits[session_type].eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '%s%s' % (line_styles[condition_name],condition.colors[session_type]))
    plt.ylim([0.4,1])
    plt.legend(loc='best')
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)

    rts=[]
    for condition_name, condition in rinfo.conditions.iteritems():
        for subject_id, subject in condition.subjects.iteritems():
            for session_id, session in subject.sessions.iteritems():
                for trial in session.trials:
                    rts.append(trial.rt)
    furl='img/rt_dist'
    fname=os.path.join(reports_dir,furl)
    rinfo.rt_dist_url='%s.png' % furl
    fig=plt.figure()
    rt_hist,rt_bins=np.histogram(np.array(rts), bins=10, density=True)
    bin_width=rt_bins[1]-rt_bins[0]
    plt.bar(rt_bins[:-1], rt_hist, width=bin_width)
    plt.xlabel('RT')
    plt.ylabel('Proportion of trials')
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)

    #create report
    template_file='experiment.html'
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR))
    template=env.get_template(template_file)

    output_file='experiment.html'
    fname=os.path.join(reports_dir,output_file)
    stream=template.stream(rinfo=rinfo)
    stream.dump(fname)

if __name__=='__main__':
    run_analysis('../../data/stim','../../data/reports')