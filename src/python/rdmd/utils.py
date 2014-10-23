from glob import glob
import os
import numpy as np
from shutil import copytree, copyfile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from psychopy.data import _baseFunctionFit

TEMPLATE_DIR='../html'

class Struct():
    def __init__(self):
        pass

def mdm_outliers(dist):
    c=1.1926
    medians=[]
    for idx,x in enumerate(dist):
        medians.append(np.median(np.abs(x-dist)))
    mdm=c*np.median(medians)
    outliers=[]
    for idx,x in enumerate(dist):
        if np.median(np.abs(x-dist))/mdm>3:
            outliers.append(idx)
    return outliers

def save_to_png(fig, output_file):
    fig.set_facecolor("#FFFFFF")
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(output_file, dpi=72)

def make_report_dirs(output_dir):

    rdirs = ['img']
    if not os.path.exists(output_dir):
        try:
            os.mkdir(output_dir)
        except Exception:
            print 'Could not make directory %s' % output_dir

    for d in rdirs:
        dname = os.path.join(output_dir, d)
        if not os.path.exists(dname):
            try:
                os.mkdir(dname)
            except Exception:
                print 'Could not make directory %s' % dname

    dirs_to_copy = ['js', 'css']
    for d in dirs_to_copy:
        srcdir = os.path.join(TEMPLATE_DIR, d)
        destdir = os.path.join(output_dir, d)
        if not os.path.exists(destdir):
            try:
                copytree(srcdir, destdir)
            except Exception:
                print 'Problem copying %s to %s' % (srcdir, destdir)

    imgfiles = glob(os.path.join(TEMPLATE_DIR, '*.gif'))
    for ipath in imgfiles:
        [rootdir, ifile] = os.path.split(ipath)
        destfile = os.path.join(output_dir, ifile)
        if not os.path.exists(destfile):
            copyfile(ipath, destfile)


def weibull(x, alpha, beta):
    return 1.0-0.5*np.exp(-(x/alpha)**beta)


def rt_function(x, a, k, tr):
    #return a/(k*x)*np.tanh(a*k*x)+tr
    return a*np.tanh(k*x)+tr

class FitRT(_baseFunctionFit):
    """Fit a Weibull function (either 2AFC or YN)
    of the form::

        y = chance + (1.0-chance)*(1-exp( -(xx/alpha)**(beta) ))

    and with inverse::

        x = alpha * (-log((1.0-y)/(1-chance)))**(1.0/beta)

    After fitting the function you can evaluate an array of x-values
    with ``fit.eval(x)``, retrieve the inverse of the function with
    ``fit.inverse(y)`` or retrieve the parameters from ``fit.params``
    (a list with ``[alpha, beta]``)"""
    def eval(self, xx=None, params=None):
        if params==None:  params=self.params #so the user can set params for this particular eval
        a = params[0]
        k = params[1]
        tr = params[2]
        xx = np.asarray(xx)
        yy = a*np.tanh(k*xx)+tr
        return yy
    def inverse(self, yy, params=None):
        if params==None: params=self.params #so the user can set params for this particular inv
        a = params[0]
        k = params[1]
        tr = params[2]
        xx = np.arctanh((yy-tr)/a)/k
        return xx

