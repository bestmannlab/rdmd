from psychopy import data, gui, core
from psychopy.data import TrialHandler
from rdmd.task import RDMDTask

def run_block(filename, expInfo, task, reps, coherence_levels):
    ##################################################################################################################
    #########          Training
    ##################################################################################################################
    #make a text file to save data
    data_file = open(filename + '.csv', 'w')#a simple text file with 'comma-separated-values'
    data_file.write('motionDirection,coherence,correct,rt\n')
    # trial list - each coherence level in both directions
    trial_list = []
    for coherence in coherence_levels:
        trial_list.append({'coherence': coherence, 'direction': 0})
        trial_list.append({'coherence': coherence, 'direction': 180})

    # Create trial handler to randomize conditions
    trial_handler = TrialHandler(trial_list, reps, extraInfo=expInfo, method='fullRandom')
    # Show instructions
    task.display_instructions()
    # Keep track of total correct
    total_correct=0
    # Keep track of mean RT on most difficult trial
    total_hard_rt=0
    # Keep track of total task time
    task_clock=core.Clock()

    # Run training trials
    for trial_idx,trial in enumerate(trial_handler):
        if (trial_idx+1) % int(len(trial_list)*reps/5)==0:
            task.display_break()

        # Run trial
        correct, rt = task.runTrial(trial['coherence'], trial['direction'])

        #add the data to the trial handler
        trial_handler.addData('response', correct)
        trial_handler.addData('rt', rt)
        data_file.write('%i,%.4f,%i,%.4f\n' % (trial['direction'], trial['coherence'], correct, rt))

        # Update block statistics
        total_correct+=correct
        if trial['coherence']==min(coherence_levels):
            total_hard_rt+=rt

    print('block took %0.2f minutes' % (task_clock.getTime()/60.0))
    # compute feedback stats and provide feedback
    correct_per_min=float(total_correct)/(task_clock.getTime()/60.0)
    perc_correct=float(total_correct)/float(len(trial_list)*reps)*100.0
    mean_hard_rt=(float(total_hard_rt)/(2.0*float(reps)))*1000.0

    task.display_feedback(perc_correct, mean_hard_rt, correct_per_min)

    # training has ended
    data_file.close()
    # special python binary file to save all the info
    trial_handler.saveAsPickle(fileName)

if __name__ == "__main__":

    # repetitions per condition
    reps = 25
    # training coherence levels
    coherence_levels = [0.0, .016, .032, .064, .096, .128, .256, .512]

    # experiment parameters
    expInfo = {
        'subject': '',
        'dateStr': data.getDateStr(),
        'condition': ''
    }

    #present a dialogue to change params
    dlg = gui.DlgFromDict(
        expInfo,
        title='RDMD',
        fixed=['dateStr']
    )

    # Create task
    # Fixation for 500ms
    fixation_duration=500.0
    # Display dots for max of 1s
    max_dot_duration=1000.0
    # Min ITI of 1s
    min_iti_duration=1000.0
    # break duration of 20s
    break_duration=20000.0
    task=RDMDTask(fixation_duration, max_dot_duration, min_iti_duration, break_duration)

    fileName = '%s.%s.%s' % (expInfo['subject'], expInfo['dateStr'], expInfo['condition'])
    run_block(fileName, expInfo, task, reps, coherence_levels)

    task.quit()
