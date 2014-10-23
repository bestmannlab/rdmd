from datetime import datetime
import os
import numpy as np
from jinja2 import Environment, FileSystemLoader
import matplotlib.pyplot as plt
from psychopy import data
import pylab
from rdmd.analysis import Subject, Session, Condition, Trial
from rdmd.utils import make_report_dirs, Struct, save_to_png, TEMPLATE_DIR, FitRT

class PilotSubject(Subject):
    def __init__(self, id, condition):
        Subject.__init__(self, id, condition)
        self.colors={'1_1':'g','1_2':'b','1_3':'r','2_1':'m','2_2':'c','2_3':'k'}

class PilotCondition(Condition):
    def __init__(self, name):
        Condition.__init__(self, name)
        self.colors={'1_1':'g','1_2':'b','1_3':'r','2_1':'m','2_2':'c','2_3':'k'}

subjects={
    'WY': PilotSubject('WY','pilot'),
    'AW': PilotSubject('AW','pilot'),
    'SM': PilotSubject('SM','pilot')
}

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
                    subjects[subj_id].sessions[session]=Session(subj_id, condition, session, session_time, file_name)
                    subjects[subj_id].sessions[session].read_data(data_dir)
            else:
                print('%s not found' % subj_id)

    for subj_id, subject in subjects.iteritems():
        if not subject.condition in rinfo.conditions:
            rinfo.conditions[subject.condition]=PilotCondition(subject.condition)
        rinfo.conditions[subject.condition].subjects[subj_id]=subject

    for condition in rinfo.conditions.itervalues():
        condition.create_report(reports_dir)

    line_styles={
        'pilot':'-',
    }

    furl='img/rt'
    fname=os.path.join(reports_dir, furl)
    rinfo.rt_url='%s.png' % furl
    fig=plt.figure()
    for condition_name, condition in rinfo.conditions.iteritems():
        for session_type in condition.session_mean_rt.keys():
            plt.errorbar(condition.all_coherence_levels,condition.session_mean_rt[session_type],
                yerr=condition.session_std_rt[session_type], fmt='o%s' % condition.colors[session_type])
            smoothInt = pylab.arange(0.01, max(condition.all_coherence_levels), 0.001)
            smoothResp = condition.rt_fits[session_type].eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '%s%s' % (line_styles[condition_name],condition.colors[session_type]),
                label='%s - %s' % (condition_name,session_type))
    plt.legend(loc='best')
    plt.xscale('log')
    #plt.ylim([500, 1000])
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
                'o%s' % condition.colors[session_type])
            smoothInt = pylab.arange(0.0, max(condition.all_coherence_levels[1:]), 0.001)
            smoothResp = condition.acc_fits[session_type].eval(smoothInt)
            plt.plot(smoothInt, smoothResp, '%s%s' % (line_styles[condition_name],condition.colors[session_type]),
                label='%s - %s' % (condition_name,session_type))
    plt.ylim([0.4,1])
    plt.legend(loc='best')
    save_to_png(fig, '%s.png' % fname)
    plt.close(fig)

    rts=[]
    for condition_name, condition in rinfo.conditions.iteritems():
        for subject_id, subject in condition.subjects.iteritems():
            for session_id, session in subject.sessions.iteritems():
                for trial in session.trials:
                    if not trial.idx in session.excluded_trials:
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
    run_analysis('../../data/re-pilot','../../data/reports/re-pilot')
