from datetime import datetime
import os
import numpy as np
from rdmd.utils import mdm_outliers

LEFT=180
RIGHT=0

coherences=[0.0320, 0.0640, 0.1280, 0.2560, 0.5120]
conditions=['ShamPreCathode','Cathode','ShamPreAnode','Anode']

subject_sessions={
    'YK': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode']
    ],
    'CCM': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode']
    ],
    'JH': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'NH': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'NM': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
    'RR': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'RIR': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'TO': [
        ['control', 'control', 'control'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
    'LZ': [
        ['control', 'control', 'control'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    'CM': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'CG': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'AN': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'AT': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
    'AW': [
        ['control', 'control', 'control'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ],
    'EZ': [
        ['control', 'control', 'control'],
        ['sham - pre - anode', 'anode', 'sham - post - anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'CT': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'ME': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    'YY': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    'MF': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode']
    ],
    'MJ': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    'SM': [
        ['control', 'control', 'control'],
        ['sham - pre - cathode', 'cathode', 'sham - post - cathode'],
        ['sham - pre - anode', 'anode', 'sham - post - anode']
    ],
    'AG': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'SB': [
        ['control', 'control', 'control'],
        ['control - anode', 'sham - pre - anode', 'anode'],
        ['control - cathode', 'sham - pre - cathode', 'cathode']
    ],
    'SA': [
        ['control', 'control', 'control'],
        ['control - cathode', 'sham - pre - cathode', 'cathode'],
        ['control - anode', 'sham - pre - anode', 'anode']
    ]
}


class Subject:
    def __init__(self, data_dir, subj_id):
        self.data_dir=data_dir
        self.id=subj_id

    def read_data(self, filter=True, collapse_sham=False, response_only=False):
        self.session_data={}
        sessions=[]
        for file_name in os.listdir(self.data_dir):
            if file_name.lower().endswith('.csv'):
                file_name_parts=file_name.split('.')
                if file_name_parts[0].upper()==self.id:
                    session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')
                    if not session_date in sessions:
                        sessions.append(session_date)
        sessions = sorted(sessions)

        for idx,session in enumerate(sessions):
            for file_name in os.listdir(self.data_dir):
                if file_name.lower().endswith('.csv'):
                    file_name_parts=file_name.split('.')
                    if file_name_parts[0].upper()==self.id:
                        session_date=datetime.strptime(file_name_parts[1][:11],'%Y_%b_%d')
                        if session_date==session and not file_name_parts[2]=='training':
                            run_num=int(file_name_parts[2])
                            condition=subject_sessions[self.id][idx-1][run_num-1].replace('- ','').title().replace(' ','')
                            #if condition in conditions:
                            if not condition=='Control':
                                file=open(os.path.join(self.data_dir,file_name),'r')

                                trial_data=[]
                                last_resp=float('NaN')
                                last_rt=float('NaN')
                                for line_idx,line in enumerate(file):
                                    if line_idx>0:
                                        cols=line.split(',')
                                        direction=int(cols[0])
                                        if direction==LEFT:
                                            direction=-1
                                        elif direction==RIGHT:
                                            direction=1
                                        coherence=float(cols[1])
                                        correct=int(cols[2])
                                        resp=direction
                                        if correct<1:
                                            resp=direction*-1
                                        rt=float(cols[3])*1000.0
                                        trialIdx=line_idx-1
                                        iti=1000+1000-last_rt
                                        if coherence in coherences and (not response_only or rt<=982):
                                            trial_data.append([trialIdx, direction, coherence, correct, resp, last_resp, rt, iti])
                                        last_resp=resp
                                        last_rt=rt
                                trial_data=np.array(trial_data)
                                if filter:
                                    outliers=mdm_outliers(trial_data[:,6])
                                    trial_data=trial_data[np.setdiff1d(np.array(range(trial_data.shape[0])),np.array(outliers)),:]
                                if (condition=='ShamPreAnode' or condition=='ShamPreCathode') and collapse_sham:
                                    condition='Sham'
                                    if condition in self.session_data:
                                        self.session_data[condition]=np.append(self.session_data[condition], trial_data,axis=0)
                                    else:
                                        self.session_data[condition]=trial_data
                                else:
                                    self.session_data[condition]=trial_data



def read_subjects(data_dir, filter=True, collapse_sham=False, response_only=False):
    subjects={}
    for subj_id in subject_sessions:
        subjects[subj_id]=Subject(data_dir, subj_id)
        subjects[subj_id].read_data(filter=filter, collapse_sham=collapse_sham, response_only=response_only)
    return subjects

def exclude_subjects(subjects, excluded_subjects=[]):
    filtered_subjects={}
    for subj_id, subject in subjects.iteritems():
        if not subj_id in excluded_subjects:
            filtered_subjects[subj_id]=subject
    return filtered_subjects