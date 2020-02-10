########### Stats ################
def freedman_diaconis(array):
    '''Compute the optimal number of bins for a 1-d histogram using the Freedman-Diaconis rule of thumb
       Bin width = 2IQR/cuberoot(N)
       Inputs:
           array (arr): flattened array of data
        Returns:
           bin_width (flt): width of bin in optimal binning
           n (int): number of bins
    '''
    import numpy as np
    # Get number of observations:
    N = np.shape(array)[0]
    # Get interquartile range:
    iqr = np.diff(np.quantile(array, q=[.25, .75]))
    bin_width = (2.*iqr)/(N**(1./3.))
    n = int(((np.max(array) - np.min(array)) / bin_width)+1)
    return bin_width, n

def mode(array):
    import numpy as np
    from lofti_gaia.loftiplots import freedman_diaconis
    n, bins = np.histogram(array, freedman_diaconis(array)[1])
    max_bin = np.max(n)
    bin_inner_edge = np.where(n==max_bin)[0]
    bin_outer_edge = np.where(n==max_bin)[0]+1
    # value in the middle of the highest bin:
    mode=(bins[bin_outer_edge] - bins[bin_inner_edge])/2 + bins[bin_inner_edge]
    return mode[0]

def calc_min_interval(x, alpha):
    """Internal method to determine the minimum interval of a given width
    Assumes that x is sorted numpy array.
    From: https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/hpd.py
    """
    import numpy as np
    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max

def write_stats(params,params_names,filename):
    import numpy as np
    from lofti_gaia.loftiplots import calc_min_interval
    k = open(filename, 'w')
    string = 'Parameter    Mean    Median    Std    Mode    68% Min Cred Int    95% Min Cred Int'
    k.write(string + "\n")
    k.close()
    for i in range(len(params)):
        # 68% CI:
        sorts = np.sort(params[i])
        frac=0.683
        ci68 = calc_min_interval(sorts,(1-frac))
        # 95% CI:
        frac=0.954
        ci95 = calc_min_interval(sorts,(1-frac))
        # Mode:
        m = mode(params[i])
        # Write it out:
        k = open(filename, 'a')
        string = params_names[i] + '    ' + str(np.mean(params[i])) + '    ' + str(np.median(params[i])) + '    ' +\
          str(m) + '    ' +  str(np.std(params[i])) + '    ' + str(ci68) + '    ' + str(ci95)
        k.write(string + "\n")
        k.close()

def to_si(mas,mas_yr,d):
    '''Convert from mas -> km and mas/yr -> km/s
        Input: 
         mas (array) [mas]: distance in mas
         mas_yr (array) [mas/yr]: velocity in mas/yr
         d (float) [pc]: distance to system in parsecs
        Returns:
         km (array) [km]: distance in km
         km_s (array) [km/s]: velocity in km/s
    '''
    km = ((mas*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = ((mas_yr*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = (km_s.value)*(u.km/u.yr).to(u.km/u.s)
    return km.value,km_s


################################ Plots ##################################
def plot_1d_hist(params,names,filename,bins,
                     tick_fs = 20,
                     label_fs = 25,
                     label_x_x=0.5,
                     label_x_y=-0.25,
                     figsize=(30, 5.5)
                ):
    import matplotlib.pyplot as plt
    plt.ioff()
    # Bin size fo 1d hists:
    bins=bins
    fig = plt.figure(figsize=figsize)
    for i in range(len(params)):
        ax = plt.subplot2grid((1,len(params)), (0,i))
        plt.hist(params[i],bins=bins,edgecolor='none',alpha=0.8)
        plt.tick_params(axis='both', left=True, top=True, right=True, bottom=True, \
                labelleft=False, labeltop=False, labelright=False, labelbottom=True, labelsize = tick_fs)
        plt.xticks(rotation=45)
        plt.xlabel(names[i],fontsize=label_fs)
        ax.get_xaxis().set_label_coords(label_x_x,label_x_y)

    plt.tight_layout()
    plt.savefig(filename, format='png')
    plt.close(fig)
    return fig

def plot_orbits(a1,T1,to1,e1,i1,w1,O1, filename, obsdate, plane='xy', 
                    ticksize = 10, 
                    labelsize = 12,
                    Xlabel = 'Dec (")', 
                    Ylabel = 'RA (")',
                    Zlabel = 'Z (")',
                    figsize = (7.5, 6.), 
                    axlim = 8,  
                    cmap='viridis',
                    colorlabel = 'Phase',
                    color = True,
                    colorbar = True,
               ):
    ''' Plot orbits in RA/Dec given a set of orbital elements
        Inputs:  Array of orbital elements to plot
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            filename (string): filename for written out plot
            obsdate [decimal yr]: observation date
        Optional Args:
            ticksize, labelsize (int): fontsize for tick marks and axis labels
            ylabel, xlabel (string): axis labels
            figsize (tuple): figure size (width, height)
            axlim (int) [arcsec]: axis limits
            cmap (str): colormap
            colorlabel (str): label for colorbar
            color: True plots in color, False plots orbits in bw
            colorbar: True renders colorbar, False omits colorbar.  If color is set to False, 
                colorbar must also be set to False.
        Returns: figure
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    from lofti_gaia.loftifittingtools import solve, eccentricity_anomaly
    
    fig = plt.figure(figsize=figsize)
    plt.scatter(0,0,color='orange',marker='*',s=300,zorder=10)
    plt.xlim(-axlim,axlim)
    plt.ylim(-axlim,axlim)
    plt.gca().invert_xaxis()
    plt.gca().tick_params(labelsize=ticksize)
    majorLocator   = MultipleLocator(10)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(5)
    plt.grid(ls=':')

    for a,T,to,e,i,w,O in zip(a1,T1,to1,e1,i1,w1,O1):
        times = np.linspace(obsdate,obsdate+T,5000)
        X,Y,Z = np.array([]),np.array([]),np.array([])
        E = np.array([])
        for t in times:
            n = (2*np.pi)/T
            M = n*(t-to)
            nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip([M],[e])]
            E = np.append(E,nextE)
        r1 = a*(1.-e*cos(E))
        f1 = sqrt(1.+e)*sin(E/2.)
        f2 = sqrt(1.-e)*cos(E/2.)
        f = 2.*np.arctan2(f1,f2)
        # orbit plane radius in as:
        r = (a*(1.-e**2))/(1.+(e*cos(f)))
        X1 = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
        Y1 = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
        Z1 = r * sin(w+f) * sin(i)
        X,Y,Z = np.append(X,X1),np.append(Y,Y1),np.append(Z,Z1)
        if plane == 'xy':
            if color == True:
                plt.scatter(Y,X,c=((times-obsdate)/T),cmap=cmap,s=3,lw=0)
            else:
                plt.plot(Y,X, color='black',alpha=0.4)
            plt.ylabel(Xlabel,fontsize=labelsize)
            plt.xlabel(Ylabel,fontsize=labelsize)
        if plane == 'xz':
            if color == True:
                plt.scatter(X,Z,c=((times-obsdate)/T),cmap=cmap,s=3,lw=0)
            else:
                plt.plot(X,Z, color='black',alpha=0.4)
            plt.ylabel(Zlabel,fontsize=labelsize)
            plt.xlabel(Xlabel,fontsize=labelsize)
        if plane == 'yz':
            if color == True:
                plt.scatter(Y,Z,c=((times-obsdate)/T),cmap=cmap,s=3,lw=0)
            else:
                plt.plot(Y,Z, color='black',alpha=0.4)
            plt.ylabel(Zlabel,fontsize=labelsize)
            plt.xlabel(Ylabel,fontsize=labelsize)
    if colorbar == True:
        plt.colorbar().set_label(colorlabel, fontsize=labelsize)
    plt.tight_layout()
    #plt.savefig(filename+'.pdf', format='pdf')
    plt.savefig(filename+'.png', format='png', dpi=300)
    plt.close(fig)
    return fig


def plot_orbits3d(a1,T1,to1,e1,i1,w1,O1, filename, obsdate, plane='xy',
                    num_orbits = 25,
                    ticksize = 10, 
                    labelsize = 12,
                    Xlabel = 'Dec (")', 
                    Ylabel = 'RA (")',
                    Zlabel = 'Z (")',
                    figsize = (7.5, 6.), 
                    axlim = 4,  
                    cmap='viridis',
                    colorlabel = 'Phase',
                    color = True,
                    colorbar = True,
               ):
    ''' Plot orbits in RA/Dec given a set of orbital elements
        Inputs:  Array of orbital elements to plot
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            filename (string): filename for written out plot
            obsdate [decimal yr]: observation date
        Optional Args:
            ticksize, labelsize (int): fontsize for tick marks and axis labels
            ylabel, xlabel (string): axis labels
            figsize (tuple): figure size (width, height)
            axlim (int) [arcsec]: axis limits
            cmap (str): colormap
            colorlabel (str): label for colorbar
            color: True plots in color, False plots orbits in bw
            colorbar: True renders colorbar, False omits colorbar.  If color is set to False, 
                colorbar must also be set to False.
        Returns: figure
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from mpl_toolkits.mplot3d import Axes3D
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    from matplotlib.ticker import MultipleLocator, FormatStrFormatter
    from lofti_gaia.loftifittingtools import solve, eccentricity_anomaly
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0,0,0,color='orange',marker='*',s=300,zorder=10)
    plt.xlim(axlim,-axlim)
    plt.ylim(-axlim,axlim)
    ax.set_zlim(-axlim,axlim)
    #plt.zlim(-axlim,axlim)
    ax.set_ylabel('Dec (")')
    ax.set_xlabel('RA (")')
    ax.set_zlabel('Z (")')
    plt.gca().tick_params(labelsize=ticksize)
    majorLocator   = MultipleLocator(10)
    majorFormatter = FormatStrFormatter('%d')
    minorLocator   = MultipleLocator(5)
    plt.grid(ls=':')

    for a,T,to,e,i,w,O in zip(a1,T1,to1,e1,i1,w1,O1):
        times = np.linspace(obsdate,obsdate+T,4000)
        X,Y,Z = np.array([]),np.array([]),np.array([])
        E = np.array([])
        for t in times:
            n = (2*np.pi)/T
            M = n*(t-to)
            nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip([M],[e])]
            E = np.append(E,nextE)
        r1 = a*(1.-e*cos(E))
        f1 = sqrt(1.+e)*sin(E/2.)
        f2 = sqrt(1.-e)*cos(E/2.)
        f = 2.*np.arctan2(f1,f2)
        # orbit plane radius in as:
        r = (a*(1.-e**2))/(1.+(e*cos(f)))
        X1 = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
        Y1 = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
        Z1 = r * sin(w+f) * sin(i)
        X,Y,Z = np.append(X,X1),np.append(Y,Y1),np.append(Z,Z1)
        ax.scatter(Y,X,Z,c=((times-obsdate)/T),cmap=cmap,s=3,lw=0)
    #if colorbar == True:
    #    plt.colorbar().set_label(colorlabel)
    plt.tight_layout()
    plt.savefig(filename+'.png', format='png', dpi=300)
    return fig

def plot_observables_hist(a,T,to,e,i,w,O,date,dist, filename,
                    bins = 50,
                    ticksize = 10, 
                    labelsize = 12,
                    figsize = (7.5, 6.)
                    ):
    ''' Plot histograms of posteriors of velocities and acceleration in the X, Y, and Z directions
        Inputs:
            a [as]: semi-major axis in mas
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
        Returns: 
            Histograms in:
            X, Y, Z positions in plane of the sky [mas],
            X dot, Y dot, Z dot three dimensional velocities [km/s]
            X ddot, Y ddot, Z ddot 3d accelerations in [m/s/yr]
    '''
    import matplotlib.pyplot as plt
    import numpy as np
    from lofti_gaia.loftifittingtools import calc_XYZ, calc_velocities, calc_accel
    
    ddot = calc_accel(a,T,to,e,i,w,O,date,dist)
    dot = calc_velocities(a,T,to,e,i,w,O,date,dist)
    pos = calc_XYZ(a,T,to,e,i,w,O,date)
    plt.ioff()
    # Bin size fo 1d hists:
    bins=bins

    # Plot X:
    fig = plt.figure(figsize=figsize)
    plt.hist(pos[0],bins=50)
    plt.xlabel(r'$X$ [mas]')
    plt.tight_layout()
    plt.savefig(filename+'X.png', format='png', dpi=300)
    plt.close(fig)
    # Plot Y:
    fig = plt.figure(figsize=figsize)
    plt.hist(pos[1],bins=50)
    plt.xlabel(r'$Y$ [mas]')
    plt.tight_layout()
    plt.savefig(filename+'Y.png', format='png', dpi=300)
    plt.close(fig)
    # Plot Z:
    fig = plt.figure(figsize=figsize)
    plt.hist(pos[2],bins=50)
    plt.xlabel(r'$Z$ [mas]')
    plt.tight_layout()
    plt.savefig(filename+'Z.png', format='png', dpi=300)
    plt.close(fig)
    # Plot Xdot:
    fig = plt.figure(figsize=figsize)
    plt.hist(dot[0],bins=50)
    plt.xlabel(r'$\dot{X}$ [km/s]')
    plt.tight_layout()
    plt.savefig(filename+'xdot.png', format='png', dpi=300)
    plt.close(fig)
    # Plot Ydot:
    fig = plt.figure(figsize=figsize)
    plt.hist(dot[1],bins=50)
    plt.xlabel(r'$\dot{Y}$ km/s]')
    plt.tight_layout()
    plt.savefig(filename+'ydot.png', format='png', dpi=300)
    plt.close(fig)
    # Plot Zdot:
    fig = plt.figure(figsize=figsize)
    plt.hist(dot[2],bins=50)
    plt.xlabel(r'$\dot{Z}$ [km/s]')
    plt.tight_layout()
    plt.savefig(filename+'zdot.png', format='png', dpi=300)
    plt.close(fig)
    # Plot Xddot:
    fig = plt.figure(figsize=figsize)
    plt.hist(ddot[0],bins=50)
    plt.xlabel(r'$\ddot{X}$ [m/s/yr]')
    plt.tight_layout()
    plt.savefig(filename+'xddot.png', format='png', dpi=300)
    plt.close(fig)
    # Plot Yddot:
    fig = plt.figure(figsize=figsize)
    plt.hist(ddot[1],bins=50)
    plt.xlabel(r'$\ddot{Y}$ [m/s/yr]')
    plt.tight_layout()
    plt.savefig(filename+'yddot.png', format='png', dpi=300)
    plt.close(fig)
    # Plot Zddot:
    fig = plt.figure(figsize=figsize)
    plt.hist(ddot[2],bins=50)
    plt.xlabel(r'$\ddot{Z}$ [m/s/yr]')
    plt.tight_layout()
    plt.savefig(filename+'zddot.png', format='png', dpi=300)
    plt.close(fig)
    return fig
