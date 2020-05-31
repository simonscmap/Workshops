

compiler('proch', 'abun', 'tblDarwin_Phytoplankton', 'picoprokaryote');


function vars = filter_vars(species, secondaryKeyword)
     
    % Search the catalog and return variables matching the keywords.

    % Paratmeters:
    % ================
    % :param array vars: list of detected variables matching the keywords.
    % :param str species: partial or full name of an species (example: "proch"). 
    % :param str secondaryKeyword: any other keywords (separated by blank space).
    
    vars = CMAP.search_catalog(sprintf('%s %s', species, secondaryKeyword));
    if height(vars) < 1
    	error('No matching entry for: %s %s', species, secondaryKeyword)
    end    
    fprintf('\n\n\n******** The following %d variables identified ********\n', height(vars))
    disp(vars(:, 1:4))
    fprintf('\n******************************************************\n\n\n')
end    
    
    

function data = fetch_var(tableName, varName)
    % Retrieves all records of a variable (varName) within a dataset (tableName).
    % Refuse to retrieve if the dataset has more than a max threshold number of records (150k).
    % This is to make sure that a massive dataset such as darwin is not retrieved.

    % Paratmeters:
    % ================
    % :param obj api: an instance of CMAP API.
    % :param str tableName: the table name where the variable is hosted. 
    % :param str varName: the name of the variable to be retrieved.
    
    
    stat = CMAP.get_var_stat(tableName, varName);
    if (stat.Variable_Count < 150000)
       sql = sprintf('SELECT [time], lat, lon, depth, %s FROM %s WHERE %s IS NOT NULL', varName, tableName, varName);
       if ~CMAP.has_field(tableName, 'depth') 
           sql = strrep(sql , ', depth', ', 0 depth');
       end
       data = CMAP.query(sql);       
    else
       fprintf('\n\nDataset too large: \n (Table:%s,  Variable: %s)\n\n\n', tableName, varName)
       data = NaN;
    end   
            
end



function compiler(species, secondaryKeyword, targetTable, targetVariable)
    
    % Retrieves all records of a variable (varName) within a dataset (tableName).
    % Refuse to retrieve if the dataset has more than a max threshold number of records (150k).
    % This is to make sure that a massive dataset such as darwin is not retrieved.

    % Paratmeters:
    % ================
    % :param str species: partial or full name of an species (examples: "proch", "syn"). 
    % :param str secondaryKeyword: any other keywords (separated by blank space).
    % :param string targetTable: the name of the table to be matched with the compiled observations.
    % :param string targetVariable: the name of the variable to be matched with the compiled observations.
    
    
    vars = filter_vars(species, secondaryKeyword);
    dataDir = "./data/";
    if ~exist(dataDir, 'dir')
       mkdir(dataDir)
    end
    
    
     for i = 1:height(vars)
         varName = string(vars(i, :).Variable);
         tableName = string(vars(i, :).Table_Name);
         unit = string(vars(i, :).Unit);
          if contains(varName, '_quality') || contains(varName, '_stdev')
              continue
          end
          fprintf('Downloading: (Table:%s,  Variable: %s)\n', tableName, varName)
          tStart = cputime;
          data = fetch_var(tableName, varName);
          if ~isempty(data)
             fprintf('\n%s downloaded after %1.1f seconds.\n', varName, cputime-tStart)
             fprintf('______________________________\n')  
              data.Unit(:) = cellstr(strcat('[', unit, ']'));
              % data = localizer(data, targetTable, targetVariable, 2, 0.5, 0.5, 5);  % a simple colocalizer
              write(data, sprintf('%s%s_%s.csv', dataDir, tableName, varName))
          end    
     end    
end
