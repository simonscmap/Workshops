import os, glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt

# %matplotlib inline


def normalize(vec, MinMax=True):    
    """
    Scales the input data. 
    By default the data is transformed so that the max and min values are transformed to a range between 1 to 0.
    If MinMax=False, a standard scaler is applied (mean of zero and standard deviation of 1).

    Paratmeters:
    ================
    :param numpy.array or pandas.series vec: an array of float values.
    :param bool MinMax: if True (default), the input array is normalized by its min and max values. Otherwise the input array is standardized. 
    """

    scaler = preprocessing.MinMaxScaler() if MinMax else preprocessing.StandardScaler()
    return scaler.fit_transform(vec)


def plot_double_axis(x, obs, mod, xLabel, obsLabel, modLabel, title):
    """
    Make a double-axis plot comparing observation and model.

    Paratmeters:
    ================
    :param array x: x-axis values.
    :param array obs: observations.
    :param array mod: model values.
    :param array xLabel: x-axis label.
    :param array obsLabel: y-axis label for observations.
    :param array modLabel: y-axis label for model values.
    :param str title: figure title.
    """

    plt.clf()
    fig, ax1 = plt.subplots()
    alpha = 1
    if len(obs) > 200: alpha = .8 
    if len(obs) > 500: alpha = .7 
    if len(obs) > 1000: alpha = .6 
    if len(obs) > 2000: alpha = .5 
    if len(obs) > 5000: alpha = .4 
    if len(obs) > 10000: alpha = .1         
    obsColor = (0.0, 0.749, 1.0, alpha) 
    obsColor1 = (0.0, 0.749, 1.0, 1)
    obsMarker = "."
    modColor = (0.698, 0.133, 0.133, alpha)  
    modColor1 = (0.698, 0.133, 0.133, 1)    
    modMarker = "."
    ## observation trace
    ax1.set_xlabel(xLabel)
    ax1.set_ylabel(obsLabel, color=obsColor1)
    plt1 = ax1.plot(x, obs, lw=1, marker=obsMarker, markeredgewidth=0, color=obsColor, label="Observation")
    ax1.tick_params(axis="y", labelcolor=obsColor1)
    ## model trace
    ax2 = ax1.twinx()    
    ax2.set_ylabel(modLabel, color=modColor1)  
    plt2 = ax2.plot(x, mod, lw=1, marker=modMarker, markeredgewidth=0, color=modColor, label="Model")
    ax2.tick_params(axis="y", labelcolor=modColor1)    
    ## legend & title
    plts = plt1 + plt2
    labs = [l.get_label() for l in plts]
    leg = ax1.legend(plts, labs)
    leg.legendHandles[0].set_color(obsColor1)
    leg.legendHandles[1].set_color(modColor1)
    plt.title(title)
    ## export
    fig.tight_layout()
    plt.savefig("%splot_%s.png" % (figDir, title), dpi=300)
    # plt.show()
    plt.close()
    return


def hist(df, title):
    """
    Make a histogram plot comparing the density distributions of model and observation data.
    The histograms are superimposed by Kernel Density Estimate (KDE) plots computed by Gaussian kernels.

    Paratmeters:
    ================
    :param array x: x-axis values.
    :param array obs: observations.
    :param array mod: model values.
    :param array xLabel: x-axis label.
    :param array obsLabel: y-axis label for observations.
    :param array modLabel: y-axis label for model values.
    :param str title: figure title.
    """

    plt.clf()
    _, ax = plt.subplots()
    df.plot.kde(ax=ax, lw=1, legend=False, color=["darkorchid", "orange"], title=title)
    df.plot.hist(density=True, ax=ax, color=["deepskyblue", "firebrick"], alpha=1)
    ax.set_ylabel("Probability")
    plt.savefig("%shist_%s.png" % (figDir, title), dpi=300)
    plt.close()
    return




def plot_metrics(vars, corr, mse, evs):
    """
    Make a histogram plot comparing the density distributions of model and observation data.
    The histograms are superimposed by Kernel Density Estimate (KDE) plots computed by Gaussian kernels.

    Paratmeters:
    ================
    :param array x: x-axis values.
    """

    plt.clf()
    plt.plot(vars, corr, '-o', label="Spearman Correlation Coefficient")
    plt.plot(vars, mse, '-o', label="Mean Squared Error")
    # plt.plot(vars, evs, '-.', label="Explained Variance Score")
    plt.title("Comparison Metrics")
    plt.ylabel("Scores")
    plt.xticks(rotation=90, fontsize=5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("%smetrics.png" % figDir, dpi=300)
    plt.close()
    return








    
species = "diatom"
files = glob.glob("./data/%s/*.csv" % species)
obsCol = 4   # observation values are at the fifth column (index 4)
modCol = 6   # model values are at the seventh column 
obsVars, corr, mse, evs = [], [], [], []


figDir = "./fig/"
if not os.path.exists(figDir): os.makedirs(figDir)
for fIndex, fName in enumerate(files):
    print(fName)
    df = pd.read_csv(fName)    
    varName = os.path.splitext(os.path.basename(fName))[0]

    ## remove rows with missing values
    df.dropna(how='any', subset=[df.columns[obsCol], df.columns[modCol]], inplace=True)
    if len(df) < 1: continue


    ## normalize
    obs = df.iloc[:, [obsCol]].values.astype(float)    
    mod = df.iloc[:, [modCol]].values.astype(float)    
    df.iloc[:, [obsCol]] = normalize(obs, MinMax=True) 
    df.iloc[:, [modCol]] = normalize(mod, MinMax=True) 


    obsVars.append(varName)
    ## data-model correlation
    corr.append(df.iloc[:, [obsCol, modCol]].corr(method="spearman").iloc[0, 1])
    ## mse
    mse.append(mean_squared_error(df.iloc[:, [obsCol]], df.iloc[:, [modCol]]))
    ## explained_variance_score
    evs.append(explained_variance_score(df.iloc[:, [obsCol]], df.iloc[:, [modCol]]))

    if fIndex == len(files)-1:
        plot_metrics(obsVars, corr, mse, evs)     

    title = varName
    hist(df.iloc[:, [obsCol, modCol]], title)
    xColLabel = "time"


    # if fName.find("tblSeaFlow") != -1: continue
    plot_double_axis(
                    x=df[xColLabel], 
                    obs=df.iloc[:, [obsCol]], 
                    mod=df.iloc[:, [modCol]], 
                    xLabel=xColLabel, 
                    obsLabel=df.columns[obsCol] + df.iloc[0, 5], 
                    modLabel=df.columns[modCol] + df.iloc[0, 8], 
                    title=title
                    )                             

