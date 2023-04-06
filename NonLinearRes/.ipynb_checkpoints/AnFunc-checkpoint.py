import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import numpy as np
from scipy.optimize import curve_fit
import lmfit 
import os
import scipy
from resonator_tools import circuit
import pandas as pd
import imageio




#transform dbm to watt
def dbm_to_watt(Data):
    return 10**(Data/10)

#transform amplitude of vna to db
def amp_to_db(Data):
    return 20*np.log10(Data)





def Lab_ExtractData2D(LogFile):
    """ Function for extracting the data of a 2D sweep from Labber. The function
    takes as input the logFileName and return
    
    z: A 2D array containing all the values of the measurement
    y: A 1D array of all the steps of the Channel that was swept
    x: A 1D array """
    
    z=LogFile.getData() #Two 2d array of all the values
    y= LogFile.getStepChannels()[0]["values"] #1D array of the step values (y axis)
    x=LogFile.getTraceXY()[0] #Retrieve the first trace of the measurement (we take only the x axis)
    
    return x,y,z 

def plot2D(x,y,z, labels = ["","",""], colormap = "viridis",reversed=True, fontsize = 14, dpi = 300, vmin = None, vmax = None, title = "", figsize = None):
    """ Function that makes the color map for 2D plot
    
    z: A 2D array containing all the values of the measurement
    y: A 1D array of all the steps of the Channel that was swept
    x: A 1D array 
    labels : list of the three names for the axes (x,y,z)
    colormap : string of the colormap
    fontisze : number for the size of the font
    vmin : minimum value for the colormap
    vmax : maximum value fo color map
    title: string of the graph title
    figsize : tulpe (1,1) of the size in inch by inch (1,1) is associated to 80 x 80 pixels """
    
     
    # We are using the reverse map
    if reversed:
        orig_map=plt.cm.get_cmap('viridis')
        colormap = orig_map.reversed()
    
    fig= plt.figure(figsize=figsize,dpi=dpi)
    
    ax=fig.add_axes([0.1, 0.1 ,0.9, 0.9])
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    ax.set_title(title,fontsize = fontsize+2)            
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    
    
    im=ax.pcolormesh(x,y,z,cmap=colormap,vmin=vmin,vmax=vmax,shading="auto")
    
    cbar = fig.colorbar(im)
    cbar.set_label(labels[2], fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)
    

    return fig, ax, cbar

def plot1D(x,y, labels = ["",""], title = "", grid = True, fontsize = 14, dpi = 300, label = "", figsize=None, linethickness=1,color="r"):
    """ Function that makes a 1D plot
    
    y: A 1D array 
    x: A 1D array 
    labels : list of the three names for the axes (x,y)
    fontisze : number for the size of the font
    title: string of the graph title
    figsize : tulpe (1,1) of the size in inch by inch (1,1) is associated to 80 x 80 pixels """
    
    fig=plt.figure(figsize=figsize,dpi=dpi)
    
    ax=fig.add_axes([0.1, 0.1 ,0.9, 0.9])
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    ax.set_title(title)
    ax.set_title(title,fontsize = fontsize+2)            
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.grid(grid)     

    ax.plot(x,y,label=label,color=color,linewidth=linethickness)
  
    return fig, ax


def plot1D_Scatter(x,y, marker="o", markersize="1",linestyle='None',labels = ["",""], title = "", grid = True, fontsize = 14, dpi = 300, label = "", figsize=None, linethickness=1,color="r",markerfacecolor="r",markeredgecolor="None"):
    """ Function that makes a 1D plot
    
    y: A 1D array 
    x: A 1D array 
    labels : list of the three names for the axes (x,y)
    fontisze : number for the size of the font
    title: string of the graph title
    figsize : tulpe (1,1) of the size in inch by inch (1,1) is associated to 80 x 80 pixels """
    
    fig=plt.figure(figsize=figsize,dpi=dpi)
    
    ax=fig.add_axes([0.1, 0.1 ,0.9, 0.9])
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    ax.set_title(title)
    ax.set_title(title,fontsize = fontsize+2)            
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.grid(grid)     

    ax.plot(x,y,marker=marker,markersize=markersize,linestyle=linestyle,label=label,color=color,linewidth=linethickness,markerfacecolor=markerfacecolor,markeredgecolor=markeredgecolor)
  
    return fig, ax


def plot1D_ErrorBar(x,y,yerr,log=True,marker="o", markersize=5,linestyle='None',labels = ["",""], title = "", grid = True, fontsize = 14, dpi = 300, label = "", figsize=None, linethickness=1,color="r",markerfacecolor="r",markeredgecolor="None",ecolor="red",elinewidth=1):
    """ Function that makes a 1D plot
    
    y: A 1D array 
    x: A 1D array 
    labels : list of the three names for the axes (x,y)
    fontisze : number for the size of the font
    title: string of the graph title
    figsize : tulpe (1,1) of the size in inch by inch (1,1) is associated to 80 x 80 pixels """
    
    fig=plt.figure(figsize=figsize,dpi=dpi)
    
    ax=fig.add_axes([0.1, 0.1 ,0.9, 0.9])
    if log:
        ax.loglog()
    else :
        plt.semilogx()
    ax.set_xlabel(labels[0], fontsize=fontsize)
    ax.set_ylabel(labels[1], fontsize=fontsize)
    ax.set_title(title)
    ax.set_title(title,fontsize = fontsize+2)            
    ax.tick_params(axis='x', labelsize=fontsize)
    ax.tick_params(axis='y', labelsize=fontsize)
    ax.grid(grid)
    ax.errorbar(x,y,yerr,marker=marker,linestyle=linestyle, markerfacecolor=markerfacecolor, markeredgecolor=markeredgecolor, markersize=markersize,ecolor=ecolor,elinewidth=elinewidth)
    
    

                                                   
  
    return fig, ax






def Slice1D(x,y,z,value,CutDirection):
    """ Function that returns a 1D cut of the graph. It takes into input, the x,y,z data of the graph, the value at which 
    to cut and the CutDirection specified either as "x" or "y"""
    
    if CutDirection=="y":
    
        idx=abs(y-value).argmin() # find the index corresponding to the cut value
        cut=z[idx]

    elif CutDirection=="x":
        
        idx=abs(x-value).argmin()
        cut=z[:,idx]

    return cut

def CropData_2D(x,y,z,boundaries):
    """ Function that crops a set of 2D datapoint 
    x: 1d array
    y: 1d array
    z: 2d array
    boundaries= [xmin, xmax, ymin, ymax] list of the values where to crop"""
    
    idx_xmin=abs(boundaries[0]-x).argmin()
    idx_xmax=abs(boundaries[1]-x).argmin()+1
   
    idx_ymin=abs(boundaries[2]-y).argmin()
    idx_ymax=abs(boundaries[3]-y).argmin()+1
   
    z_crop=z[idx_ymin:idx_ymax, idx_xmin:idx_xmax]
    x_crop=x[idx_xmin:idx_xmax]
    y_crop=y[idx_ymin: idx_ymax]
   
    return x_crop, y_crop, z_crop
 
def CropData_1D(x,y,boundaries):
    """ Function that crops a set of 1D datapoint along the x axis 
    x: 1d array
    y: 1d array
    boundaries= [xmin, xmax] list of the values where to crop"""
    
    idx_xmin=abs(boundaries[0]-x).argmin()
    idx_xmax=abs(boundaries[1]-x).argmin()+1

    
    x_crop=x[idx_xmin:idx_xmax]
    y_crop=y[idx_xmin: idx_xmax]
   
    return x_crop, y_crop

def Substract_mean(z,axis):
    """ Function that substracts the mean of a 2D matrix along a specific direction. 
    axis can be x or y"""
 
    if axis=="y":
       
        average=np.mean(z,axis=1)
        averaged_z=(z.transpose()-average).transpose()

    elif axis=="x":
        
        average=np.mean(z,axis=0)
        averaged_z=(z-average)
       
    return averaged_z





def Find_peaks_1D(trace,prom,dist):
    """ Function to find all the peaks in a single trace map. Mainly used to find the right parameters for the 
    2D version. Takes as an input the following parameters
    
    trace: vector that should be composed of only real values and the peaks that we want to find should
    be higher than the baseline. The peaks will be searched by going through the rows of the matrix.
    
    porm: prominence factor for peaks : defined as the distance between the peak and its lowest counter line (see matlab documentation)
    dist: distance factor for the peak search """ 
    
    column_idx=[] #empty list for the peaks index
    row_idx=[] #empty list for the frequency indexs 

    idx=find_peaks(trace,prominence=prom,distance=dist)[0] # find the peaks on this row with prominence prom and distance t
    i=0
    if len(idx)!=0: # if we find a peak 

        column_idx.extend(idx.tolist()) # we add the index of the peak to the peak_idx (idx corresponding to the column where the peak is)
        row_idx.extend([i]*len(idx.tolist())) # We also add the idx of the row in which we found the peak 

    return column_idx,row_idx 






def Find_peaks_2D(z,prom,dist):
    """ Function to find all the peaks in a 2D map. Takes
    as an input the following parameters
    
    z: matrix z that should be composed of only real values and the peaks that we want to find should
    be higher than the baseline. The peaks will be searched by going through the rows of the matrix.
    
    porm: prominence factor for peaks : defined as the distance between the peak and its lowest counter line (see matlab documentation)
    dist: distance factor for the peak search """

    i=0 #Row number counter 
    
    column_idx=[] #empty list for the peaks index
    row_idx=[] #empty list for the frequency indexs 

    for trace in z: # loop over all the rows 

        idx=find_peaks(trace,prominence=prom,distance=dist)[0] # find the peaks on this row with prominence prom and distance t

        if len(idx)!=0: # if we find a peak 
            
            column_idx.extend(idx.tolist()) # we add the index of the peak to the peak_idx (idx corresponding to the column where the peak is)
            row_idx.extend([i]*len(idx.tolist())) # We also add the idx of the row in which we found the peak 
        
        i+=1 #increment for the next row 

    return column_idx,row_idx 

def NonLin_Flux_Fitting(xdata,ydata,w_0,gamma,scaling,offset,xfit):
    """Function used to fit the flux response of the lambda/4 resonator. The model is as described in krantz thesis
    xdata and ydata are the experimental points to fit. w_0, gamma, scaling and offset are the value for the initial guess on the parameters. 
    
    w_0=resonance frequency at zero flux
    gamma : participation ratio of the squid inductance at zero flux over the resonator inductance
    scaling : is the convertion ratio from the voltage applied to the coil to the flux
    offset : offset between the theorical zero flux point and experimental 
    xfit : x value for fitting the new functions"""
    

    params=lmfit.Parameters() # object 
    params.add('w_0',value=w_0)
    params["w_0"].min=0 #sets the minimum bound to 0
    params.add('gamma',value =gamma)
    params["gamma"].min=0 #sets the minimum bound to 0
    params.add('scaling',value=scaling)
    params.add('offset',value=offset)
    
    def get_residual(params,xdata,ydata):
        "Calculate the residuals between the data and model"
        
        w_0=params['w_0'].value
        gamma=params['gamma'].value
        scaling=params['scaling'].value
        offset=params['offset'].value
        
        model = w_0/(1+gamma/np.abs(np.cos((xdata-offset)*scaling)))
        
        return ydata-model 
    
    #Minmize the residualts
    fit_params=lmfit.minimize(get_residual,params,args=(xdata,ydata))
    
    #Calculate the fitted function 
    w_0=fit_params.params['w_0'].value
    gamma=fit_params.params['gamma'].value
    scaling=fit_params.params['scaling'].value
    offset=fit_params.params['offset'].value

    fit = w_0/(1+gamma/np.abs(np.cos((xfit-offset)*scaling)))
    
    return fit_params, fit 


def save_data(datadict, meastype, name, device, cooldown_date, bias=0, filepath=r"C:\Users\HQClabo\Documents\Data\QuantumMachine"):
  
    """ Function creating a folder and a mat file to save the data contained in datadict. 
       
       datadict : dictionnary containing the variable name and its value, "I":I
       meastype : string of the measurement type for e.g time_rabi
       name : string name of the folder that will contain the measurement
       device : string of the device name
       bias : bias value in 
       cooldown_data : string of the cooldown date in the following format r"\2022_05_06"
       filepath : string of the filepath in the format """
    
    
    bias=str(bias) #Convert bias to string
    sep = "\\" #separator in file 
    
        
    # Creates the standard multidata dictionnary
    multidata = {"MeasurementType" : meastype, "CooldownDate": cooldown_date, "Device" : device, "Name":name, "bias": bias}
    
    # Append to standard dictionnary the measurement value dictionnary
    multidata.update(datadict)
    
    
  
    #Create a directory in the filepath 
    if not os.path.exists(filepath + cooldown_date + sep + device + sep + name):
        os.makedirs(filepath + cooldown_date + sep + device + sep + name)
        scipy.io.savemat(filepath + cooldown_date + sep + device + sep + name + sep + meastype + '.mat'  , multidata)
            
    
    if os.path.isfile(filepath + cooldown_date + sep + device + sep + name + sep + meastype + '.mat'):
        print("Warning: File name already exsists, please select a new file name")
    else:
         print("Data saved")
         scipy.io.savemat(filepath + cooldown_date + sep + device + sep + name + sep + meastype + '.mat'  , multidata)
        
        



def find_nearest(array, value):
    "find nearest value in an array "
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

    
def Non_Lin_Map(Filename,Linear=True,span=50) :
    """ Function extracting the data from a mat file and creating the 2d map for the triangular shape. 

    Filename : name of the file 
    Linear : True for linear and False for dbm 
    span : span of the average 

    This returns : 
    amplitudes (1, nb of amplitudes) : array of the amplitudes used in the experiment 
    Pump frequency (1, nb of pump frequency) : array of the pump frequency in the expriment 
    Analyzer_freq : #2d Matrix containing all the analyzer frequency set in the pump sweep (nb pump frequency, nb of points)
    Data : 2D Matrix where the rows are pump frequency and the columns are the analyzer frequency 
    integrate : 2D matrix of the integrated trace where the rows are the amplitudes and the cplumns are the pump frequency

    """
    
    #Extract the dictionnary 
    data_dir=scipy.io.loadmat(Filename, mdict=None, appendmat=True)
    
    
    amplitudes=data_dir["amplitudes"] # Array of all the amplitudes that were used (1, nb of amplitudes)
    Analyzer_freq=data_dir["Analyzer_freq"] # 2D matrix of every analyzer frequency set in the pump sweep  (nb pump frequency, nb of points)
    Pump_freq=data_dir["Pump_freq"] # 1 D array of the pump frequency that we spanned  (1, nb of pump frequency)
    Data=data_dir["Data"] # 2D Matrix where there the rows are pump frequency and columns are analyzer 
     
    #If linear is true, we convert the dbm to watt     
    if Linear:    
        convert_dbm_to_watt=np.vectorize(dbm_to_watt)
        Data=convert_dbm_to_watt(Data)   
        
               
    integrate=np.zeros([amplitudes.shape[1],Pump_freq.shape[1]]) # Array that will contain all the results of integration
    maximum=np.zeros([amplitudes.shape[1],Pump_freq.shape[1]]) # Array that will contain all maximum 
    average=np.zeros([amplitudes.shape[1],Pump_freq.shape[1]])

    # For each Data map that we took at a given amplitude 
    for idx in range(amplitudes.shape[1]):
        
        arr=Data[:,:,idx].mean(axis=0)#Mean value along the spectrum analyzer 
        a=find_nearest(arr, np.max(arr)) #find the maximum vna frequency indices 
        average[idx,:]=Data[:,a-span:a+span,idx].mean(axis=1) #Average
        i=0
        
        # For each trace corresponding to a pump frequency 
        for trace in Data[:,:,idx]: 
            
         
            integrate[idx,i]=np.trapz(trace,Analyzer_freq[i,:])  #Calculate the integral 
            maximum[idx,i]=max(trace)  #Calculate the maximum 
            i+=1
            
            
    return amplitudes, Pump_freq, Analyzer_freq, Data, integrate, maximum, average


def Non_Lin_Map_av(Filename,N,Linear=True,span=50):
    """ Function making the average of multiple 2D integrated maps  
    """
    amplitudes, Pump_freq, Analyzer_freq, Data, integrate_stack, maximum, average  =Non_Lin_Map(Filename+str(0),Linear=Linear,span=span)
    
    for i in range(1,N,1):
        amplitudes, Pump_freq, Analyzer_freq, Data, integrate, maximum, average  =Non_Lin_Map(Filename+str(i),Linear=True)
        integrate_stack=np.dstack((integrate_stack,integrate))
        
    
    #Average along the third dimensions 
    Averaged_integrate=np.mean(integrate_stack,axis=2)
    
    return Averaged_integrate


def average_trace(Filename, amp):
    """ Function calculating the integral and average_trace 
    
    Take as input filename : the folder name where all the data is stored 
    amp : the amplitude vector corresponding to all the amplitudes values in the loop. 
    
    return mean_data : 3d matrix of size pump freq x analyzer freq x amplitudes where each 2d matrix is the averaged for a given amplitude
    integrated_average : the integrated 2d traces 
    pump_freq : 1D array : frequency of the pump
    Analyzer_freq : 1D array : frequency of the spectrum analyzer 
    """
    
    Map=Filename+str(amp[0])+".mat"
    data_dir=scipy.io.loadmat(Map, mdict=None, appendmat=True)

    #Extract specific information from the first trace 
    Pump_freq=data_dir["Pump_freq"][0]
    Analyzer_center_freq=data_dir["Analyzer_center_freq"][0,0]
    Analyzer_span=data_dir["Analyzer_span"][0,0]
    Numpoints=data_dir["Numpoints"][0,0]
    Analyzer_freq=np.linspace((Analyzer_center_freq-Analyzer_span/2), (Analyzer_center_freq+Analyzer_span/2),Numpoints)


    Mean_Data=np.zeros((len(Pump_freq),len(Analyzer_freq),len(amp)))

    for i in range(len(amp)):
        #Extract the data for amplitude i 
        Map=Filename+str(amp[i])+".mat"
        data_dir=scipy.io.loadmat(Map, mdict=None, appendmat=True)
        Data=data_dir["Data"]

        #Calculate the mean data 
        Mean_Data[:,:,i]=np.mean(Data,axis=2)

        
        
    dbm_Mean_data=10**(Mean_Data/10)   
    integrated_average=np.zeros((len(amp),len(Pump_freq)))

    for i in range(len(amp)):
        integrate=np.trapz(dbm_Mean_data[:,:,i],Analyzer_freq)
        integrated_average[i,:]=integrate

        
        
    return Mean_Data, integrated_average, Pump_freq, Analyzer_freq



def Fit_Single(x,y,plot=True):
    """ Function to fit the resonance of a single hanged resonator"""
    
    port = circuit.notch_port()
    port.add_data(x,y)
    port.autofit()
    
    if plot:
        port.plotall()
        
    return port 

def fit_power_sweep(z,freq,power, attenuation=80, plot=False):
    """
    Function to fit resonances of a power sweep.
    Parameters
    ----------
    z: ndarray
        Complex data of dimension (n_power,n_freq).
    freq : ndarray
        List of the measured frequencies.
    power : ndarray
        List of the different powers applied in amplitude and not db.
    attenuation : number, optional
        Total estimated line attenuation. The default is 80.
    port_type : str, optional
        Type of the resonator port. Choose 'notch' or 'reflection.
        The default is 'notch'.
    plot : boolean, optional
        Enable option to plot the individual fits. The default is False.
    Returns
    -------
    fitReport : dict
        Dictionary containing the results of the fit as lists for each parameter.
    """
    
    fitReport = {
        "Qi" : [], #internal
        "Qi_err" : [], #internal error 
        "Qc" : [], #external
        "Qc_err" : [], #external error
        "Ql" : [], #total
        "Ql_err" : [], #total error 
        "Nph" : [], #nombre de photons
        "fr" : [], #frequence
        }

    for k,i in enumerate(power):
        
        port=circuit.notch_port()
        port.add_data(freq,z[k])
        port.autofit()
    
        if plot:
            port.plotall()
    
        fitReport["Qi"].append(port.fitresults["Qi_dia_corr"])
        fitReport["Qi_err"].append(port.fitresults["Qi_dia_corr_err"])
        fitReport["Qc"].append(port.fitresults["Qc_dia_corr"])
        fitReport["Qc_err"].append(port.fitresults["absQc_err"])
        fitReport["Ql"].append(port.fitresults["Ql"])
        fitReport["Ql_err"].append(port.fitresults['Ql_err'])
        fitReport["fr"].append(port.fitresults['fr'])
        fitReport["Nph"].append(port.get_photons_in_resonator(i-attenuation,unit='dBm'))
        
        #For resonator tool, Ql is the loaded (total quality factor), Qi (is the internal quality facotr), Qc is the external or coupling quality factor 
    
    return fitReport



def fit_simulation(LogFileName,center_freq, span,plot=True):
    
    """
    Function to fit the simulated spectrum of sonnet.
    
    The function take as an input a filename csv from the simulation and returns the fit results  
    """ 
                   
    data = pd.read_csv(LogFileName, delimiter= ',',header=1)
                   
    # Extract the data from the file               
    Freq=data['Frequency (GHz)'].to_numpy()
    Re_S21=data["RE[S21]"].to_numpy()
    IM_S21=data["IM[S21]"].to_numpy()
    S21=Re_S21+1j*IM_S21

    # fitting the data 
    port = circuit.notch_port()
    port.add_data(Freq,S21)
    port.cut_data(center_freq-span/2,center_freq+span/2) #cuts around the central frequency 
    port.autofit()
    
    if plot:
        port.plotall()
    
    return port.fitresults
#data["Frequency (GHz)"].to_numpy()

def blob_movie(i,I,Q,freqs,time,FileName,GiftName,bin_size,fontsize=8,movie=False,figsize=(4,12),fps=2):
    """
    Function to that creates a movie to look at the frequency evolution of the IQ blob. 
    
    i : idx of the frame to look at if movie is false
    I : I quadrature matrix
    Q : Q quadrature matirx
    freqs : frequency array
    dt : time spacing
    FileName : name of the folder that will contain all the frame
    GiftName : name of the gif that will be created
    bin_size : size of the bins for the histogram 
    """ 
   

    def create_frame(idx,fontsize,FileName,movie,bin_size):

        fig= plt.figure(figsize=figsize,dpi=300)

        ax = plt.subplot(311)
        ax.plot(I[idx,:]*1e6,Q[idx,:]*1e6,".",markersize=2)
        ax.set_title('Pump Frequency '+str(freqs[0,idx])+ "GHz"+" Frame: "+str(idx),fontsize=fontsize)
        ax.set_xlabel("I [$\mu V$]",fontsize=fontsize)
        ax.set_ylabel("Q [$\mu V$]",fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)


        ax=plt.subplot(312)
        plt.plot(time,I[idx,:], ".", markersize=2)
        plt.xlabel("Time [s]")
        plt.ylabel("Phase")
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)


        ax = plt.subplot(313)
        ax.hist2d(I[idx,:]*1e6,Q[idx,:]*1e6,bins=bin_size)
        ax.set_xlabel("I [$\mu V$]",fontsize=fontsize)
        ax.set_ylabel("Q [$\mu V$]",fontsize=fontsize)
        ax.set_ylim(-15, 15) 
        ax.set_xlim(-15, 15)
        ax.set_facecolor(color=(68/255,1/255,84/255))
    
    
        if movie:
        
            plt.savefig(f'./'+FileName+f'/img_{idx}.png', 
                    transparent = False,  
                    facecolor = 'white'
                   )    


            plt.close()
            
    if movie:
        os.makedirs(FileName)
        
        for i in range(freqs.shape[1]):
            create_frame(i,fontsize,FileName,movie,bin_size)

        frames = []

        for i in range(freqs.shape[1]):
            image = imageio.v2.imread(f'./'+FileName+f'/img_{i}.png')
            frames.append(image)

        imageio.mimsave('./'+GiftName+'.gif', # output gif
                        frames,          # array of input frames
                        fps = fps)         # optional: frames per second
        
    else:
        create_frame(i,fontsize,FileName,movie,bin_size)
        
        
        
        
def blob_amplitude_movie(i,I,Q,freqs,dt,average_trace,Pump_freq,amp_factor,FileName,GiftName,bin_size,fontsize=8,movie=False,figsize=(4,12),fps=2):
    """
    Function to that creates a movie to look at the frequency evolution of the IQ blob with the amplitude curves on top to compare. 

    i : idx of the frame to look at if movie is false
    I : I quadrature matrix
    Q : Q quadrature matirx
    freqs : pump frequency array for the blobs
    dt : time spacing
    average_trace : trace for the amplitude as a function of pump frequency 
    Pump_frequency : pump frequeency to trace the average_trace
    amp_factor : amplitude used to make the plot 
    FileName : name of the folder that will contain all the frame
    GiftName : name of the gif that will be created
    bin_size : size of the bins for the histogram 
    """ 



    def create_frame(idx,fontsize,FileName,bin_size,movie):

        fig= plt.figure(figsize=(10,8),dpi=300)

        ax = plt.subplot(211)
        ax.set_title('Amplitude '+str(amp_factor)+" Frame :"+str(idx),fontsize=fontsize)
        ax.set_xlabel("Pump Frequency [GHz]",fontsize=fontsize)
        ax.set_ylabel("Photon Number [a.u]",fontsize=fontsize)
        ax.plot((Pump_freq.transpose())/1e9,average_trace, '.-', alpha=.9, lw=.5, ms=8)
        ax.plot([freqs[0,idx],freqs[0,idx]],[0,4],'r')
        ax.plot([freqs[0,idx],freqs[0,idx]],[0,4],'r')
        ax.set_ylim(0, 0.65) 
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)


        ax = plt.subplot(223)
        ax.hist2d(I[idx,:]*1e6,Q[idx,:]*1e6,bins=bin_size);
        ax.set_xlabel("I [$\mu$V]",fontsize=fontsize)
        ax.set_ylabel("Q [$\mu$V]",fontsize=fontsize)
        ax.set_ylim(-15, 15) 
        ax.set_xlim(-15, 15)
        ax.set_facecolor(color=(68/255,1/255,84/255))
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)


        time=np.linspace(1,len(I[idx,:]),len(I[idx,:]),len(I[idx,:]))*5.0268e-5

        ax = plt.subplot(224)
        ax.plot(time,np.angle(I[idx,:]+1j*Q[idx,:]),".",markersize=2)
        ax.set_ylabel("Phase",fontsize=fontsize)
        ax.set_xlabel("Time [s]",fontsize=fontsize)
        ax.tick_params(axis='x', labelsize=fontsize)
        ax.tick_params(axis='y', labelsize=fontsize)

        if movie:

            plt.savefig(f'./'+FileName+f'/img_{idx}.png', 
                    transparent = False,  
                    facecolor = 'white'
                   )    

            plt.close()
     
   
    if movie:
        
        #Make a directory 
        os.makedirs(FileName)

        for i in range(freqs.shape[1]):
            create_frame(i,fontsize,FileName,bin_size,movie)

        frames = []

        for i in range(freqs.shape[1]):
            image = imageio.v2.imread(f'./'+FileName+f'/img_{i}.png')
            frames.append(image)

        imageio.mimsave('./'+GiftName+'.gif', # output gif
                        frames,          # array of input frames
                        fps = fps)         # optional: frames per second

    #If movie is false, we simply plot the image 
    else:
        create_frame(i,fontsize,FileName,bin_size,movie)
        
    
    
def rotate_data(I,Q, nb_angle):
    """Function that rotates the dataset to maximize the I quadrature
    
    I : Matrix (rows are the different pump frequencies) and columns are the points (or vector)
    Q: Matrix (rows are the different pump frequencies) and columns are the points (or vector)
    nb_angle : numberof points that are being used in the curve of maximum of I vs angle 
    
    returns :
    rotated I vector or matrix
    rotated Q vector or matrix 
    """
    
    #Defines all of the angles 
    rotate_angle=np.linspace(-np.pi/2,0,nb_angle)

    rot_I=np.zeros(I.shape)
    rot_Q=np.zeros(Q.shape)

    # For each row of I 
    for i in range(I.shape[0]):
        
        Data=I[i,:]+1j*Q[i,:]
        average=np.zeros((len(rotate_angle)))

        #rotate by an angle and look at the average of the absolute value 
        for idx,angle in enumerate(rotate_angle):

            rot_Data=Data*np.exp(1j*angle)
            average[idx]=np.mean(np.abs(rot_Data.real)) #take the average value of the absolute 

        angle_max=rotate_angle[np.where(average==np.amax(average))] #select the angle where the average is maximum 
        rot_Data=Data*np.exp(1j*angle_max)

        rot_I[i,:]=rot_Data.real
        rot_Q[i,:]=rot_Data.imag
        
    return rot_I, rot_Q


def noise_average(a, n=1) :
    """
    This function does a static average of a vector , in order to reduce noise in too noisy signals.
    
    a: vector to average 
    n : number of points using in the average. It should be a divided of the vector length 
    
    returns : avgResult : averaged vector 
    """    
    if len(a)%n != 0:
        a = np.append(a, a[-1]*np.ones(n-len(a)%n))
        
    avgResult = np.average(a.reshape(-1, n), axis=1)
    return avgResult

def average_data(I,Q,time,n_avg):
     """
    This function does a static average using the function noise_average over a full matrix (down sample)
    
    time : vector of the different times 
    I : Matrix (rows are the different pump frequencies) and columns are the points (or vector)
    Q: Matrix (rows are the different pump frequencies) and columns are the points (or vector)
    n_avg : number used for the average 
    """
        
    
    average_I=np.zeros(I.shape)
    average_Q=np.zeros(Q.shape)
    
    time_average=noise_average(time, n=n_avg)

    average_I=np.zeros((I.shape[0],time_average.shape[0]))
    average_Q=np.zeros((I.shape[0],time_average.shape[0]))
    
    for i in range(I.shape[0]):
        average_I[i,:]=noise_average(I[i,:], n=n_avg)
        average_Q[i,:]=noise_average(Q[i,:], n=n_avg)
        
    return average_I, average_Q, time_average