from psychopy import visual, event, core
from rdmd.config import MONITOR, SCREEN

class RDMDTask():
    def __init__(self, fixation_duration, max_dot_duration, min_iti_duration, break_duration):
        '''
        RDMD task
        hemifield = hemifield to show dot stim in (right or left)
        fixation_duration = duration of fixation (ms)
        max_dot_duration = max duration of dot stimuli (ms)
        min_iti_duration = min inter-trial-interval (ms)
        break_duration = duration of break (ms)
        '''
        #create window and stimuli
        # Window to use
        self.wintype='pyglet' # use pyglet if possible, it's faster at event handling
        self.win = visual.Window(
            [1280,1024],
            monitor=MONITOR,
            screen=SCREEN,
            units="deg",
            fullscr=True,
            color=[-1,-1,-1],
            winType=self.wintype)
        self.win.setMouseVisible(False)
        event.clearEvents()

        # Measure frame rate
        self.mean_ms_per_frame, std_ms_per_frame, median_ms_per_frame=visual.getMsPerFrame(self.win, nFrames=60,
            showVisual=True)

        # Compute number of frames for fixation
        self.fixation_duration=fixation_duration
        self.fixation_frames=int(fixation_duration/self.mean_ms_per_frame)

        # Compute max number of frames for dot stimuli
        self.max_dot_duration=max_dot_duration
        self.max_dot_frames=int(max_dot_duration/self.mean_ms_per_frame)

        # Compute minimum inter-trial-interval frames (will be adjusted based on time of other stimuli)
        self.min_iti_duration=min_iti_duration
        self.min_iti_frames=int(min_iti_duration/self.mean_ms_per_frame)

        self.break_duration=break_duration
        self.break_duration_frames=int(break_duration/self.mean_ms_per_frame)

        # fixation stimulus
        self.fixation = visual.PatchStim(
            self.win,
            units='deg',
            tex=None,
            mask='circle',
            sf=0,
            size=0.5,
            name='fixation',
            autoLog=False,
            color=[1,1,1]
        )

        # dot stimulus
        self.dots = visual.DotStim(
            win=self.win,
            name='dots',
            nDots=200, # number of dots
            dotSize=2, # dot size in degrees
            speed=0.4, # 60Hz refresh - 16.7ms/frame, 4deg/s=.0668deg/frame
            dir=0.0, # 0=right, 180=left
            coherence=12.5, # percentage of dots moving in the same direction
            fieldPos=[0.0, 0.0], # centered on the screen
            fieldSize=10.0, # field is 10 deg wide
            fieldShape='circle', # circle shaped field
            signalDots='different', # are the signal and noise dots 'different' or 'same' popns (see Scase et al)
            noiseDots='direction', # do the noise dots follow random- 'walk', 'direction', or 'position'
            dotLife=3, # number of frames for each dot to be drawn
            color=[1.0,1.0,1.0], # white dots
            colorSpace='rgb',
            opacity=1, # fully opaque
            depth=-1.0
        )

        self.training_message = visual.TextStim(self.win, wrapWidth=30, pos=[0,3])

        # Clock for computing response time
        self.rt_clock = core.Clock()

    def display_instructions(self):
        #display instructions and wait
        instr_text='You will be shown a white dot in the center of the screen. Fixate on this dot for the duration ' \
        'of the trial. After %.2f seconds a field of smaller moving dots will appear.' \
        'Hit the left or right arrow key to indicate the direction that the ' \
        'majority of these dots are moving in. Please respond as quickly as possible given a high level of accuracy. ' \
        'For difficult displays, you may take some time to improve your judgment. For this experiment, we will give ' \
        'you a target mean response time for the most difficult displays of about 800 ms.' % \
                   (self.fixation_duration/1000.0)
        message1 = visual.TextStim(self.win, wrapWidth=30, pos=[0,0], text=instr_text)
        message2 = visual.TextStim(self.win, pos=[0,-8],text='Hit any key when ready.')
        message1.draw()
        message2.draw()
        self.win.flip()
        #pause until there's a keypress
        event.waitKeys()

    def display_feedback(self, perc_correct, mean_hard_rt, correct_per_min):
        message1 = visual.TextStim(self.win, wrapWidth=30, pos=[0,8], text='Block feedback')
        message2 = visual.TextStim(self.win, wrapWidth=30, pos=[0,6], text='correct responses: %.2f %%' % perc_correct)
        message3 = visual.TextStim(self.win, wrapWidth=30, pos=[0,4],
            text='correct responses/min: %.2f' % correct_per_min)
        message4 = visual.TextStim(self.win, wrapWidth=30, pos=[0,2],
            text='mean response time in difficult conditions: %.2f ms' % mean_hard_rt)
        message1.draw()
        message2.draw()
        message3.draw()
        message4.draw()
        self.win.flip()
        #pause until there's a keypress
        event.waitKeys()

    def display_break(self):
        message1 = visual.TextStim(self.win, wrapWidth=30, pos=[0,2], text='Take a short break')
        message2 = visual.TextStim(self.win, pos=[0,0],text='Hit any key when ready.')
        for f in range(self.break_duration_frames):
            message1.draw()
            self.win.flip()
        message1.draw()
        message2.draw()
        self.win.flip()
        event.waitKeys()

    def runTrial(self, coherence, direction, training=False):
        """
        Run one trial
        coherence = dot motion coherence level
        direction = direction of coherently moving dot
        show_feedback = show subject accuracy feedback
        """

        # Update dot direction and coherence
        self.dots.setDir(direction)
        self.dots.setFieldCoherence(coherence)

        # draw fixation
        for f in range(self.fixation_frames):
            self.fixation.draw()
            self.win.flip()

        # counter - number of times dots refreshed before response
        n_dot_frames=0

        # extra frames for ITI (to make trials same length)
        extra_iti_frames=0

        # clear any keystrokes before starting
        event.clearEvents()
        all_keys=[]

        # wait for a keypress
        while len(all_keys)==0 and n_dot_frames<self.max_dot_frames:

            # Draw stimuli
            self.dots.draw()
            self.fixation.draw()
            if training:
                if direction==180:
                    self.training_message.setText('Left')
                else:
                    self.training_message.setText('Right')
                self.training_message.draw()
            self.win.flip()

            # Reset RT clock after first presentation
            if not n_dot_frames:
                self.rt_clock.reset()

            # if timeStamped = a core.Clock object, it causes return of the tuple (key,time-elapsed-since-last-reset)
            all_keys=event.getKeys(timeStamped=self.rt_clock)
            n_dot_frames+=1

        # incorrect
        correct = 0
        rt=self.rt_clock.getTime()

        if len(all_keys):
            # if don't have pyglet, need to get the time explicitly
            if not self.wintype == 'pyglet':
                all_keys[0][1] = self.rt_clock.getTime()

            # unpack all_keys  taking the first keypress in the list
            thisKey=all_keys[0][0].upper()
            rt =all_keys[0][1]
            if thisKey=='LEFT' or thisKey=='RIGHT':
                # correct
                if (thisKey=='LEFT' and direction==180) or (thisKey=='RIGHT' and direction==0):
                    correct = 1
            # abort experiment
            elif thisKey in ['Q', 'ESCAPE']:
                core.quit()

            # must clear other (eg mouse) events - they clog the buffer
            event.clearEvents()

        # Update number of ITI frames so trials are same length
        extra_iti_frames+=self.max_dot_frames-n_dot_frames

        # blank screen
        for i in range(self.min_iti_frames+extra_iti_frames):
            self.win.flip()

        return correct, rt

    def quit(self):
        self.win.close()
        core.quit()