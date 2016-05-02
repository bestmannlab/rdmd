from glob import glob
import math
import os
import numpy as np
from shutil import copytree, copyfile
from matplotlib.backends.backend_agg import FigureCanvasAgg
from psychopy.data import _baseFunctionFit

TEMPLATE_DIR='../html'

class Struct():
    def __init__(self):
        pass

def sd_outliers(dist, c):
    mean=np.mean(dist)
    std=np.std(dist)
    outliers=[]
    for idx,x in enumerate(dist):
        if np.abs(x-mean)>=c*std:
            outliers.append(idx)
    return outliers

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

def movingaverage (values, window):
    weights = np.repeat(1.0, window)/window
    #sma = np.convolve(values, weights, 'valid')
    sma = np.convolve(values, weights, 'same')
    return sma

def save_to_png(fig, output_file):
    fig.set_facecolor("#FFFFFF")
    canvas = FigureCanvasAgg(fig)
    canvas.print_png(output_file, dpi=72)

def save_to_eps(fig, output_file):
    fig.set_facecolor("#FFFFFF")
    canvas = FigureCanvasAgg(fig)
    canvas.print_eps(output_file, dpi=72)

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

class FitSigmoid(_baseFunctionFit):
    def eval(self, xx=None, params=None):
        if params==None:  params=self.params #so the user can set params for this particular eval
        x0 = params[0]
        k=params[1]
        xx = np.asarray(xx)
        #yy = a+1.0/(1.0+np.exp(-k*xx))
        yy =1.0/(1.0+np.exp(-k*(xx-x0)))
        return yy

    def inverse(self, yy, params=None):
        if params==None:  params=self.params #so the user can set params for this particular eval
        x0 = params[0]
        k=params[1]
        #xx = -np.log((1/(yy-a))-1)/k
        xx = -np.log((1.0/yy)-1.0)/k+x0
        return xx

"""
file    twoway_interaction.py
author  Ernesto P. Adorio, Ph.D.
        ernesto.adorio@gmail.com
        UPDEPP at Clarkfield
desc    Performs an anova with interaction
        on replicated input data.
        Each block must have the same number
        of input values.
version 0.0.2 Sep 12, 2011

"""

def twoway_interaction(groups, first_factor_label, second_factor_label, format="html"):
    b = len(groups[0][0])
    a = len(groups)
    c = len(groups[0])
    groupsums = [0.0] * c

    #print "blocks, a, c=", b, a, c

    #print "Input groups:"
    v = 0.0   #total variation
    vs = 0.0  #subtotal variation
    vr = 0.0  #variation between rows
    GT = 0
    for i in range(a):
        vsx = 0.0
        vrx = 0.0
        for j in range(c):
            vsx = sum(groups[i][j])
            groupsums[j] += vsx
            #print "debug vsx", vsx
            vrx += vsx
            vs += vsx * vsx
            for k in range(b):
                x = groups[i][j][k]
                v += x * x
                GT += x
        vr += vrx* vrx

    #print "groupsums=", groupsums, vs

    totadjustment = GT*GT/(a * b * c)
    vs = vs/b - totadjustment
    vr = vr/(b * c)- totadjustment
    v  -= totadjustment
    vc = sum([x * x for x in groupsums])/ (a*b)-totadjustment
    vi = vs-vr -vc
    ve = v- (vr + vc + vi)
    #print "debug vs, vr, vc=", vs, vr, vc

    dfvr = (a-1)
    dfvc = (c-1.0)
    dfvi = ((a-1)*(c-1))
    dfve = (a * c* (b-1))
    dfvs = a*c - 1
    dfv  = (a * b * c -1)
    mvr = vr/(dfvr)
    mvc = vc/(dfvc)
    mvi = vi / dfvi
    mve = ve/dfve
    Fr = mvr/mve
    Fc = mvc/mve
    Fi = mvi/mve

    from scipy import stats

    pvalr = 1.0 - stats.f.cdf(Fr, dfvr, dfve)
    pvalc = 1.0 - stats.f.cdf(Fc, dfvc, dfve)
    pvali = 1.0 - stats.f.cdf(Fi, dfvi, dfve)


    if format=="html":
        output="""
    <table border="1">
    <tr><th>Variation  </th><th>Sum of Squares</th><th>  df</th><th>  Mean Sum of Squares</th><th>   F-value</th><th> p-value</th></tr>
    <td>Rows(%s) </td><td>%f</td><td>    %d</td><td>     %f</td> <td> %f</td> <td>%f</td></tr>
    <tr><td>Columns(%s)</td><td>%f</td><td>  %d</td><td>     %f</td> <td> %f</td> <td>%f</td></tr>
    <tr><td>Interaction</td><td>%f</td><td>  %d</td><td>     %f</td> <td> %f</td> <td>%f</td></tr>
    <tr><td>Subtotals </td><td> %f</td><td> %d</td></tr>
    <tr><td>Residuals(random)  </td><td>%f</td><td>  %d</td><td>%f</td></tr>
    <tr><td>Totals</td><td>%f.2 </td><td>%d </td></tr>
    </table>
    """ % (first_factor_label,vr, dfvr, mvr, mvr/mve, pvalr,
           second_factor_label,vc, dfvc, mvc, mvc/mve, pvalc,
           vi, dfvi, mvi, mvi/mve, pvali,
           vs, dfvs,
           ve, dfve, mve,
           v,  dfv)
    else:
        output=[[vr, dfvr, mvr, mvr/mve, pvalr],
                [vc, dfvc, mvc, mvc/mve, pvalc],
                [vi, dfvi, mvi, mvi/mve, pvali],
                [vs, dfvs],
                [ve, dfve, mve],
                [v,  dfv]]

    return output


def get_twod_confidence_interval(x, y):
    covariance=np.cov(x,y)
    [eigenvals, eigenvecs ] = np.linalg.eig(covariance)
    max_eigenval=np.max(eigenvals)
    max_eigenval_idx=np.where(eigenvals==max_eigenval)[0][0]
    max_eigenvec=eigenvecs[:,max_eigenval_idx]
    min_eigenval=np.min(eigenvals)
    angle = math.atan2(max_eigenvec[1], max_eigenvec[0])
    if angle < 0:
        angle = angle + 2*math.pi
    centroid_center=[np.mean(x),np.mean(y)]
    # Get the 95% confidence interval error ellipse
    chisquare_val = 2.4477
    theta_grid = np.arange(0,2*math.pi+2*math.pi/100.0,2*math.pi/100.0)
    phi = angle
    X0=centroid_center[0]
    Y0=centroid_center[1]
    a=chisquare_val*np.sqrt(max_eigenval)
    b=chisquare_val*np.sqrt(min_eigenval)
    # the ellipse in x and y coordinates
    ellipse_x_r  = a*np.cos( theta_grid )
    ellipse_y_r  = b*np.sin( theta_grid )

    #Define a rotation matrix
    R = np.array([[ math.cos(phi), math.sin(phi)],[-math.sin(phi), math.cos(phi)]])

    #let's rotate the ellipse to some angle phi
    r_ellipse = np.dot(np.transpose(np.array([ellipse_x_r,ellipse_y_r])),R)

    return r_ellipse[:,0] + X0,r_ellipse[:,1] + Y0