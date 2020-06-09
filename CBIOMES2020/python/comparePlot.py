import os, glob
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error, explained_variance_score
import matplotlib.pyplot as plt



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


def normalize_obs_model(df, obsCol, modCol):    
    """
    Normalize the observation and model values.

    Paratmeters:
    ================
    :param dataframe df: the dataframe containing the model and observation values. 
    :param int obsCol: the index of column at the data file holding observations.
    :param int modCol: the index of column at the data file holding model estimates.
    """

    obs = df.iloc[:, [obsCol]].values.astype(float)    
    mod = df.iloc[:, [modCol]].values.astype(float)    
    df.iloc[:, [obsCol]] = normalize(obs, MinMax=True) 
    df.iloc[:, [modCol]] = normalize(mod, MinMax=True) 
    return df



def plot_double_axis(x, obs, mod, xLabel, obsLabel, modLabel, title, figDir):
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


def hist(df, title, figDir):
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




def plot_metrics(vars, corr, mse, evs, figDir):
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
    plt.legend(loc=3, prop={'size': 6})
    plt.tight_layout()
    plt.savefig("%smetrics.png" % figDir, dpi=300)
    plt.close()
    return



def stackFiles(obsCol, modCol, files):
    """
    Take a list of files containing the observation and model-estimated values
    corresponding to the specified species and stack them vertically.

    Paratmeters:
    ================
    :param int obsCol: the index of column at the data file holding observations.
    :param list x: list of file names where the observation and model data are stored.
    """

    stacked = pd.DataFrame({})
    for fname in files:
        df = pd.read_csv(fname)
        df.rename(columns={df.columns[obsCol]: "obs"}, inplace=True)
        df.dropna(how='any', subset=[df.columns[obsCol], df.columns[modCol]], inplace=True)
        if len(df) < 1: continue
        df = normalize_obs_model(df, obsCol, modCol)
        if len(stacked) < 1: 
            stacked = df
            continue
        stacked = pd.concat([stacked, df],ignore_index=True)
    return stacked    



def plot_aggregated_err_by_depth(df, figDir):
    depth = [0, 5, 10, 20, 40, 100, 200, 500, 1000, 5000]
    xs, errs, samples = [], [], []
    for i in range(1, len(depth)):
        sub = df.query("depth>=%f and depth<%f" % (depth[i-1], depth[i]))
        if len(sub) < 1: continue
        errs.append(sub.err.mean())
        xs.append("%d_%d" % (depth[i-1], depth[i]))
        samples.append(len(sub))
    plt.clf()
    _, ax1 = plt.subplots()
    ax1.plot(xs, errs, "bo", zorder=0)
    ax1.tick_params(axis="y", labelcolor='b') 
    ax1.set_xlabel("depth [m]")
    ax1.set_ylabel("obs - model", color='b')
    plt.xticks(rotation=20)
    
    ax2 = ax1.twinx() 
    ax2.bar(xs, samples, color="r", alpha=0.1)
    ax2.set_ylabel('sample size', color='r')  
    ax2.tick_params(axis="y", labelcolor='r')  
    ax2.set_yscale('log')  
    
    plt.tight_layout()
    plt.savefig("%sstack_err_depth.png" % figDir, dpi=300)
    plt.close()    
    return



def plot_aggregated_err_by_lat(df, figDir):
    lat = np.arange(-90, 90.1, 10)
    xs, errs, samples = [], [], []
    for i in range(1, len(lat)):
        sub = df.query("lat>=%f and lat<%f" % (lat[i-1], lat[i]))
        if len(sub) < 1: continue
        errs.append(sub.err.mean())
        xs.append("%d_%d" % (lat[i-1], lat[i]))
        samples.append(len(sub))
    plt.clf()
    _, ax1 = plt.subplots()
    ax1.plot(xs, errs, "bo", zorder=0)
    ax1.tick_params(axis="y", labelcolor='b') 
    ax1.set_xlabel("latitude [deg]")
    ax1.set_ylabel("obs - model", color='b')
    plt.xticks(rotation=40)
    
    ax2 = ax1.twinx() 
    ax2.bar(xs, samples, color="r", alpha=0.1)
    ax2.set_ylabel('sample size', color='r')  
    ax2.tick_params(axis="y", labelcolor='r')  
    ax2.set_yscale('log')  
    
    plt.tight_layout()
    plt.savefig("%sstack_err_lat.png" % figDir, dpi=300)
    plt.close()    
    return


def plot_aggregated_err_by_month(df, figDir):
    df["month"] = pd.to_datetime(df["time"]).dt.month
    df = df.query("lat>=0")
    df = df.groupby("month").agg({"lat": "size", "err": "mean"}).rename(columns={"lat": "count", "err": "err"}).reset_index()

    plt.clf()
    _, ax1 = plt.subplots()
    ax1.plot(df.month, df.err, "bo", zorder=0)
    ax1.tick_params(axis="y", labelcolor='b') 
    ax1.set_xlabel("month")
    ax1.set_ylabel("obs - model", color='b')
    ax2 = ax1.twinx() 
    ax2.bar(df.month, np.array(df["count"]), color="r", alpha=0.1)
    ax2.set_ylabel('sample size', color='r')  
    ax2.tick_params(axis="y", labelcolor='r')      
    plt.tight_layout()
    plt.savefig("%sstack_err_month.png" % figDir, dpi=300)
    plt.close()    
    return



def stacked_figures(obsCol, modCol, files, figDir):
    df = stackFiles(obsCol, modCol, files)
    df["err"] = (df[df.columns[obsCol]] - df[df.columns[modCol]]) #/ df[df.columns[obsCol]]
    # df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]

    hist(df.iloc[:, [obsCol, modCol]], 'stacked', figDir)
    plot_aggregated_err_by_depth(df, figDir)
    plot_aggregated_err_by_lat(df, figDir)
    plot_aggregated_err_by_month(df, figDir)
    return



def make_dataset_figures(obsCol, modCol, files, figDir):
    """
    Take a list of files containing the observation and model-estimated values
    corresponding to the specified species and create plots comparing the 
    observations with model estimates.

    Paratmeters:
    ================
    :param string species: the name of species.
    :param int obsCol: the index of column at the data file holding observations.
    :param int modCol: the index of column at the data file holding model estimates.
    :param list x: list of file names where the observation and model data are stored.
    """

    obsVars, corr, mse, evs = [], [], [], []
    for fIndex, fName in enumerate(files):
        print(fName)
        df = pd.read_csv(fName)    
        varName = os.path.splitext(os.path.basename(fName))[0]

        ## remove rows with missing values
        df.dropna(how='any', subset=[df.columns[obsCol], df.columns[modCol]], inplace=True)
        if len(df) < 1: continue

        ## normalize
        df = normalize_obs_model(df, obsCol, modCol)

        obsVars.append(varName)
        ## data-model correlation
        corr.append(df.iloc[:, [obsCol, modCol]].corr(method="spearman").iloc[0, 1])
        ## mse
        mse.append(mean_squared_error(df.iloc[:, [obsCol]], df.iloc[:, [modCol]]))
        ## explained_variance_score
        evs.append(explained_variance_score(df.iloc[:, [obsCol]], df.iloc[:, [modCol]]))

        if fIndex == len(files)-1:
            plot_metrics(obsVars, corr, mse, evs, figDir)     

        title = varName
        hist(df.iloc[:, [obsCol, modCol]], title, figDir)
        xColLabel = "time"

        if fName.find("tblSeaFlow") != -1: continue  # takes long time to plot seaflow data
        plot_double_axis(
                        x=df[xColLabel], 
                        obs=df.iloc[:, [obsCol]], 
                        mod=df.iloc[:, [modCol]], 
                        xLabel=xColLabel, 
                        obsLabel=df.columns[obsCol] + df.iloc[0, 5], 
                        modLabel=df.columns[modCol] + df.iloc[0, 8], 
                        title=title,
                        figDir=figDir
                        )                             











###########################################
#                                         # 
#                  main                   # 
#                                         # 
###########################################




if __name__ == "__main__":
    species = "proch"
    files = glob.glob("./data/%s/*.csv" % species)
    obsCol = 4   # observation values are at the fifth column (index 4)
    modCol = 6   # model values are at the seventh column 
    figDir = "./fig/"
    if not os.path.exists(figDir): os.makedirs(figDir)

    # stack all data and then compare with Darwin
    stacked_figures(obsCol, modCol, files, figDir)

    # compare the observation datasets with Darwin individually 
    make_dataset_figures(obsCol, modCol, files, figDir)