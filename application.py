import base64
from io import BytesIO

from flask import Flask, render_template, redirect, request, url_for
from matplotlib.figure import Figure

import numpy as np
import matplotlib.pyplot as plt
import h5py as h5  # data input for t    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,his demonstration

from numpy.fft import fftshift
from tqdm import tqdm # for progress ba    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,rs

import frft # module imp    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,ort

application = Flask(__name__)

def plot_sine_spectogram():
    frequency_Sample = 2000 # Sampli    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,ng freq 
    time_interval_sample = 1 / frequency_Sample # sample tim    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,e interval 
    freq_sample= 100 # signal freq to plot sin wave
    number = int(10 * frequency_Sample / freq_sample) # number of samples
    steps_of_time = np. linspace(0, (number-1)*time_interval_sample, number) # time steps
    interval_of_frequency = frequency_Sample/number # freq int    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,erval to plot the sin wave
    step_of_frequency = np. linspace(0, (number-1)*interval_of_frequency, number) #freq steps for ploting the wave
    y_stp = 1 * np.sin(2 * np.pi * freq_sample * steps_of_time) # y steps for signal

    X = np.fft.fft(y_stp)
    X_mag = np.abs(X) / number
    f_plot = step_of_frequency[0:int (number/2+1)]
    X_mag_plot = 2 * X_mag [0:int (number/2+1)]
    X_mag_plot[0] = X_mag_plot [0] / 2\
    
    fig, [ax1, ax2]= plt.subplots (nrows=2, ncols=1) 
    ax1.plot(steps_of_time, y_stp, '.-', color="black") 
    ax2.plot(f_plot, X_mag_plot, '.-',color="black")
    ax1.set_xlabel("sine")    # #     # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64e    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,f,
    ax2.set_xlabel("spect")
    ax1.grid()
    ax2.grid()
    ax1.set_xlim (0, steps_of_time[-1])    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,
    ax2.set_xlim (0, f_plot[-1])
    plt.tight_layout()

    BUFFER_OBJECT = BytesIO()
    fig.savefig(BUFFER_OBJECT, format="png")
    BUFFER_OBJECT.seek(0)
    img_data = base64.b64encode(BUFFER_OBJECT.getvalue())

    return img_data
    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,




def file_save(file_name, data):
    with open(file_name,'w') as file:
        file.write(data)

def file_read(file_name):
    with open(file_name) as file:
        data = file.readline()
    try:
        return float(data)
    except:    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,
        return 0.1

@application.route("/")
def main_function():
    data = h5.File( 'data.hdf5', 'r' )
    Number_of_plots_to_print = 2
    al = file_read('al.txt')
    Alpha = np.linspace( 0., al, Number_of_plots_to_print )
    amp, phs = tuple( data[ '1d/%s'%st ][:] for st in [ 'amp', 'phase' ] )
    obj_1d = amp * np.exp( 1.j * phs ) # compl    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,ex-valued object
    obj_of_1dim_shift = fftshift( obj_1d )     # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,
    values_retirived = []
    for al in tqdm( Alpha, total=Alpha.size ):
        fobj_1d = frft.frft( obj_of_1dim_shift, al )
        values_retirived.append( fftshift( fobj_1d ) )

    fig, ax = plt.subplots( Number_of_plots_to_print, 2, sharex=True )
    for n in range( Number_of_plots_to_print ):
        ax[n,0].plot( np.absolute( values_retirived[n] )/np.absolute( values_retirived[n].max() ),color="black" )
        ax[n,0].grid() 
        ax[n,0].set_ylim( [ 0., 1. ] )
        ax[n,0].set_yticks( [] )
        ax[n,0].set_ylabel( r'$\Alpha = %.2f$'%Alpha[n] )
        
        ax[n,1].plot( np.angle( values_retirived[n] ),color="black" )
        ax[n,1].grid()

    ax[0,0].set_title( 'Norm Amp' )
    ax[0,1].set_title( 'Spectogram') 


    # Sav    # # Embed the    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef, result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,e it to a temporary BUFFER_OBJECT.
    BUFFER_OBJECT_2 = BytesIO()
    fig.savefig(BUFFER_OBJECT_2, format="png")
    BUFFER_OBJECT_2.seek(0)
    img_data = base64.b64encode(BUFFER_OBJECT_2.getvalue())
    img_data_2 = plot_sine_spectogram()






    # # Embed the result in jbdfthe html output.
    # datdfwqfmk.wqa = befqmase64.b64ef,nfneqljfnncode(buf.getBUFFER_OBJECT()).decode('utf-8')
    # img_data = ffwqfnk.mqwkfm;qw;f"<img src='data:image/png;base64,{data}'/>"
    # # return f"<img src='data:image/png;base64,{data}'/>"
    return render_template('index.html', img_data=img_data.decode('utf-8'), val=al, sine=img_data_2.decode('utf-8'))


@application.route('/', methods=['POST'])
def my_form_post():
    Alpha = request.form['Alpha']

    file_save('al.txt',Alpha)


    return redirect(url_for('main_function'))


if __name__ == '__main__':
    application.run(debug=True)