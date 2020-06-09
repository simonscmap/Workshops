import datetime
import pycmap
import pandas as pd
import numpy as np
from dateutil.parser import parse



def localizer(api, source, targetTable, targetVariable, timeTolerance, latTolerance, lonTolerance, depthTolerance):
    """
    Match the observation values with traget datastet within the tolerance parameters. 

    Paratmeters:
    ================
    :param obj api: an instance of CMAP API.
    :param dataframe source: the source (observation) dataset.
    :param string targetTable: the name of the target table to be matched with source.
    :param string targetVariable: the name of the target variable to be matched with source.
    :param int timeTolerance: temporal tolerance [day].
    :param float latTolerance: spatial tolerance in meridional direction [deg].
    :param float lonTolerance: spatial tolerance in zonal direction [deg].
    :param float depthTolerance: spatial tolerance in vertical direction [m].
    """
    
    def shift_dt(dt, delta):
        delta = float(delta)
        dt = parse(dt)
        dt += datetime.timedelta(days=delta)
        # TODO: Handel monthly climatology data sets
        return dt.strftime("%Y-%m-%d %H:%M:%S")


    def in_time_window(sourceDT, targetMinDT, targetMaxDT):
        targetMinDT = targetMinDT.split(".000Z")[0]
        targetMaxDT = targetMaxDT.split(".000Z")[0]
        return not (
                    parse(sourceDT) < parse(targetMinDT) or 
                    parse(sourceDT) > parse(targetMaxDT)
                    )


    targetCoverage = api.get_var_coverage(targetTable, targetVariable)
    targetUnit = api.get_unit(targetTable, targetVariable)
    targetVariableSTD = targetVariable + "_std"
    source[targetVariable], source[targetVariableSTD], source["target_unit"] = None, None, None
    for i in range(len(source)):
        df = pd.DataFrame({})
        if in_time_window(source.iloc[i]["time"], targetCoverage["Time_Min"][0], targetCoverage["Time_Max"][0]):            
            df = api.space_time(
                                table=targetTable,
                                variable=targetVariable,
                                dt1=shift_dt(source.iloc[i]["time"], -timeTolerance),
                                dt2=shift_dt(source.iloc[i]["time"], timeTolerance),
                                lat1=source.iloc[i]["lat"] - latTolerance,
                                lat2=source.iloc[i]["lat"] + latTolerance,
                                lon1=source.iloc[i]["lon"] - lonTolerance,
                                lon2=source.iloc[i]["lon"] + lonTolerance,
                                depth1=source.iloc[i]["depth"] - depthTolerance,
                                depth2=source.iloc[i]["depth"] + depthTolerance,
                                )

        targetMean, targetSTD = None, None
        if len(df) > 0: targetMean, targetSTD = df[targetVariable].mean(), df[targetVariable].std()
        source.at[i, targetVariable] = targetMean
        source.at[i, targetVariableSTD] = targetSTD
        print("%d / %d" % (i+1, len(source))) 
    source["target_unit"] = targetUnit           
    return source