import random
from psychopy import data, gui
from rdmd.task import RDMDTask

if __name__ == "__main__":

    # experiment parameters
    expInfo = {
        'subject': 'jjb',
        'hemifield': 'left', # Hemifield to show the dots in
        'startCoherence': 5, # Initial guess for 80% threshold
        'dateStr': data.getDateStr(),
        'condition': 'pre'  # pre, stim, 0minpost, 10minpost, or 20minpost
    }

    #present a dialogue to change params
    dlg = gui.DlgFromDict(
        expInfo,
        title='RDMD',
        fixed=['dateStr']
    )

    fileName = '%s.%s.threshold.%s' % (expInfo['subject'], expInfo['dateStr'], expInfo['condition'])

    #make a text file to save data
    dataFile = open(fileName+'.csv', 'w')#a simple text file with 'comma-separated-values'
    dataFile.write('motionDirection,coherence,correct,rt\n')

    task=RDMDTask(expInfo['hemifield'], 500.0, 2000.0, 1000.0)

    #create the quest handler
    quest = data.QuestHandler(
        expInfo['startCoherence'],
        50,
        extraInfo=expInfo,
        nTrials=50,
        minVal=0,
        maxVal=100.0,
        stepType='log'
    )

    # Step through the staircase
    for thisCoherence in quest:
        # set direction of moving dots
        # will be either 0(right) or 180(left)
        movementDirection = random.choice([0,180])

        thisResp, thisRT, thisAttnResp = task.runTrial(thisCoherence, movementDirection)

        #add the data to the staircase so it can calculate the next level
        quest.addData(thisResp)
        quest.addOtherData('rt',thisRT)
        dataFile.write('%i,%.4f,%i,%.4f\n' %(movementDirection, thisCoherence, thisResp, thisRT))

    # staircase has ended
    dataFile.close()
    # special python binary file to save all the info
    quest.saveAsPickle(fileName)

    task.quit()

