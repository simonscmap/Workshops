import os, sys, time, pycmap
import pandas as pd
from simpleLocalizer import localizer





def filter_vars(api, species, secondaryKeyword):    
    """ 
    Search the catalog and return variables matching the keywords.

    Paratmeters:
    ================
    :param list vars: list of detected variables matching the keywords.
    :param str species: partial or full name of an species (example: "proch"). 
    :param str secondaryKeyword: any other keywords (separated by blank space).
    """

    vars = api.search_catalog("%s %s" % (species, secondaryKeyword))
    if len(vars) < 1: 
        print("No matching entry for: %s %s" % (species, secondaryKeyword))
        sys.exit()
    print("\n\n\n******** The following %d variables identified ********\n" % len(vars))
    print(vars[["Table_Name", "Variable", "Unit"]].to_string(index=False))
    print("\n******************************************************\n\n\n")
    return vars




def fetch_var(api, tableName, varName):
    """ 
    Retrieves all records of a variable (varName) within a dataset (tableName).
    Refuse to retrieve if the dataset has more than a max threshold number of records (150k).
    This is to make sure that a massive dataset such as darwin is not retrieved.

    Paratmeters:
    ================
    :param obj api: an instance of CMAP API.
    :param str tableName: the table name where the variable is hosted. 
    :param str varName: the name of the variable to be retrieved.
    """
    
    stat = api.get_var_stat(tableName, varName)
    if int(stat["Variable_Count"][0]) < 150000:
        sql = "SELECT [time], lat, lon, depth, %s FROM %s WHERE %s IS NOT NULL" % (varName, tableName, varName)
        if not api.has_field(tableName, "depth"): sql = sql.replace(", depth", ", 0 depth")
        return api.query(sql)
    else:
        print("\n\nDataset too large: \n (Table:%s,  Variable: %s)\n\n" % (tableName, varName))        
        return None




def compiler(species, secondaryKeyword, targetTable, targetVariable):
    """ 
    Retrieves all records of a variable (varName) within a dataset (tableName).
    Refuse to retrieve if the dataset has more than a max threshold number of records (150k).
    This is to make sure that a massive dataset such as darwin is not retrieved.

    Paratmeters:
    ================
    :param str species: partial or full name of an species (examples: "proch", "syn"). 
    :param str secondaryKeyword: any other keywords (separated by blank space).
    :param string targetTable: the name of the table to be matched with the compiled observations.
    :param string targetVariable: the name of the variable to be matched with the compiled observations.
    """
    
    ### you need to pass your API key here: 
    ### api = pycmap.API("your api key") 
    api = pycmap.API()    
    vars = filter_vars(api, species, secondaryKeyword)
    dataDir = "./data/"
    if not os.path.exists(dataDir): os.makedirs(dataDir)
    for i in range(len(vars)):
        if vars.iloc[i].Variable.find('_quality') != -1 or vars.iloc[i].Variable.find('_stdev') != -1: continue

        print("Downloading: (Table:%s,  Variable: %s)" % (vars.iloc[i].Table_Name, vars.iloc[i].Variable))
        tic = time.perf_counter()
        data = fetch_var(api, vars.iloc[i].Table_Name, vars.iloc[i].Variable)
        if data is not None:
            print("\n%s downloaded after %1.1f seconds." % (vars.iloc[i].Variable, time.perf_counter()-tic))
            print("______________________________\n")            
            data["Unit"] = "[" + vars.iloc[i]["Unit"] + "]" if isinstance(vars.iloc[i]["Unit"], str) else "" 

            # data = localizer(
            #                 source=data, 
            #                 targetTable=targetTable, 
            #                 targetVariable=targetVariable, 
            #                 timeTolerance=2, 
            #                 latTolerance=0.5, 
            #                 lonTolerance=0.5, 
            #                 depthTolerance=5
            #                 )

            data.to_csv("%s%s_%s.csv" % (dataDir, vars.iloc[i]["Table_Name"], vars.iloc[i]["Variable"]), index=False)





if __name__ == "__main__":
    compiler(
            species="proch",  
            secondaryKeyword="abun",
            targetTable="tblDarwin_Phytoplankton", 
            targetVariable="picoprokaryote"
            )
