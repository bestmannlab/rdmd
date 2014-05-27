#This analysis script takes one or more staircase datafiles as input
#from a GUI. It then plots the staircases on top of each other on 
#the left and a combined psychometric function from the same data
#on the right

from psychopy import data, gui, core
from psychopy.misc import fromFile
import pylab

if __name__ == "__main__":
    #Open a dialog box to select files from
    files = gui.fileOpenDlg('.')
    if not files:
        core.quit()

    #get the data from all the files
    allCoherenceLevels, allResponses, allRTs = [],[], []
    for thisFileName in files:
        thisDat = fromFile(thisFileName)
        allCoherenceLevels.append( thisDat.intensities )
        allResponses.append( thisDat.data )
        allRTs.append( thisDat.otherData['rt'])

    #plot each staircase
    pylab.subplot(131)
    colors = 'brgkcmbrgkcm'
    lines, names = [],[]
    for fileN, thisStair in enumerate(allCoherenceLevels):
        #lines.extend(pylab.plot(thisStair))
        #names = files[fileN]
        pylab.plot(thisStair, label=files[fileN])
    pylab.xlabel('Trial')
    pylab.ylabel('Coherence')
    #pylab.legend()

    #get combined data
    combinedInten, combinedResp, combinedN =data.functionFromStaircase(allCoherenceLevels, allResponses, 5)
    #fit curve - in this case using a Weibull function
    fit = data.FitWeibull(combinedInten, combinedResp, guess=[0.2, 0.5])
    #smoothInt = pylab.arange(min(combinedInten), max(combinedInten), 0.001)
    smoothInt = pylab.arange(0.0, max(combinedInten), 0.001)
    smoothResp = fit.eval(smoothInt)
    thresh = fit.inverse(0.8)
    print thresh

    #plot curve
    pylab.subplot(132)
    pylab.plot(smoothInt, smoothResp, '-')
    pylab.plot([thresh, thresh],[0,0.8],'--')
    pylab.plot([0, thresh], [0.8,0.8],'--')
    pylab.title('threshold = %0.3f' % thresh)
    #plot points
    pylab.plot(combinedInten, combinedResp, 'o')
    pylab.ylim([0,1])
    pylab.xlabel('Coherence')
    pylab.ylabel('% Correct')

    combinedInten, combinedRT, combinedN=data.functionFromStaircase(allCoherenceLevels, allRTs, 5)
    pylab.subplot(133)
    pylab.plot(combinedInten, combinedRT, 'o')
    pylab.xlabel('Coherence')
    pylab.ylabel('RT (s)')

    pylab.show()