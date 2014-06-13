#This analysis script takes one or more staircase datafiles as input
#from a GUI. It then plots the staircases on top of each other on 
#the left and a combined psychometric function from the same data
#on the right

import numpy as np
from psychopy import data, gui, core
from psychopy.misc import fromFile
import pylab

if __name__ == "__main__":

    cond_colors={
        'pre': 'g',
        'stim': 'r',
        'post': 'b'
    }
    #Open a dialog box to select files from
    files = gui.fileOpenDlg('.')
    if not files:
        core.quit()

    coherence_resp={}
    coherence_rt={}
    #get the data from all the files
    allCoherenceLevels, allResponses, allRTs = {},{}, {}
    for thisFileName in files:
        thisDat = fromFile(thisFileName)

        condition=thisDat.extraInfo['condition'].lower()
        if not condition in coherence_resp:
            coherence_resp[condition]={}
            coherence_rt[condition]={}
            allCoherenceLevels[condition]=[]
            allResponses[condition]=[]
            allRTs[condition]=[]

        trialList=thisDat.trialList

        for i in range(thisDat.data['response'].shape[0]):
            for j in range(thisDat.data['response'].shape[1]):
                trialIdx=thisDat.sequenceIndices[i,j]

                coherence=thisDat.trialList[trialIdx]['coherence']
                allCoherenceLevels[condition].append(coherence)

                resp=thisDat.data['response'][i,j]
                allResponses[condition].append(resp)

                rt=thisDat.data['rt'][i,j]
                allRTs[condition].append(rt)

                if not coherence in coherence_resp[condition]:
                    coherence_resp[condition][coherence]=[]
                coherence_resp[condition][coherence].append(float(resp))

                if not coherence in coherence_rt[condition]:
                    coherence_rt[condition][coherence]=[]
                coherence_rt[condition][coherence].append(rt)

    perf_ax=pylab.subplot(121)
    rt_ax=pylab.subplot(122)
    for condition in coherence_resp:
        #get combined data
        combinedInten, combinedResp, combinedN =data.functionFromStaircase(allCoherenceLevels[condition], allResponses[condition], 10)
        #combinedInten=coherence_resp.keys()
        #combinedInten.sort()
        #combinedResp=[np.mean(coherence_resp[x]) for x in combinedInten]
        #combinedRT=[np.mean(coherence_rt[x]) for x in combinedInten]

        #fit curve - in this case using a Weibull function
        fit = data.FitWeibull(combinedInten, combinedResp, guess=[0.2, 0.5])
        #smoothInt = pylab.arange(min(combinedInten), max(combinedInten), 0.001)
        smoothInt = pylab.arange(0.0, max(combinedInten), 0.001)
        smoothResp = fit.eval(smoothInt)
        thresh = fit.inverse(0.8)
        print thresh

        #plot curve
        perf_ax.plot(smoothInt, smoothResp, '-%s' % cond_colors[condition],label=condition)
        #pylab.plot([thresh, thresh],[0,0.8],'--')
        #pylab.plot([0, thresh], [0.8,0.8],'--')
        #plot points
        perf_ax.plot(combinedInten, combinedResp, 'o%s' % cond_colors[condition],label=condition)
        pylab.title('threshold = %0.3f' % thresh)
        pylab.ylim([0,1])
        pylab.xlabel('Coherence')
        pylab.ylabel('% Correct')


        combinedInten, combinedRT, combinedN=data.functionFromStaircase(allCoherenceLevels[condition], allRTs[condition], 10)
        rt_ax.plot(combinedInten, combinedRT, 'o',label=condition)
        pylab.xlabel('Coherence')
        pylab.ylabel('RT (s)')

    perf_ax.legend(loc='best')
    rt_ax.legend(loc='best')
    pylab.show()
