




function source = localizer(source, targetTable, targetVariable, timeTolerance, latTolerance, lonTolerance, depthTolerance)
    % Match the observation values with traget datastet within the tolerance parameters. 

    % Paratmeters:
    % ================
    % :param dataframe source: the source (observation) dataset.
    % :param string targetTable: the name of the target table to be matched with source.
    % :param string targetVariable: the name of the target variable to be matched with source.
    % :param int timeTolerance: temporal tolerance [day].
    % :param float latTolerance: spatial tolerance in meridional direction [deg].
    % :param float lonTolerance: spatial tolerance in zonal direction [deg].
    % :param float depthTolerance: spatial tolerance in vertical direction [m].
    

    function shifted = shift_dt(dt, delta)
        delta = double(delta);
        dtFormat = 'yyyy-MM-dd';
        if contains(dt, 'T')
            dt = strrep(dt, 'T', ' ');
            if contains(dt, '.000Z')
                dt = strrep(dt, '.000Z', '');
            end                            
            dtFormat = 'yyyy-MM-dd HH:mm:SS';
        end
        dt = datetime(dt, 'InputFormat', dtFormat);
        dt = dt + days(delta);
        % TODO: Handle monthly climatology data sets
        shifted = datestr(dt, 'yyyy-mm-dd HH:MM:SS');
    end
        

    function isBetween = in_time_window(sourceDT, targetMinDT, targetMaxDT)
        sourceDT = strrep(sourceDT , '.000Z', '');
        sourceDT = strrep(sourceDT, 'T', ' ');
        targetMinDT = strrep(targetMinDT , '.000Z', '');
        targetMinDT = strrep(targetMinDT, 'T', ' ');
        targetMaxDT = strrep(targetMaxDT , '.000Z', '');
        targetMaxDT = strrep(targetMaxDT, 'T', ' ');
        dtFormat = 'yyyy-MM-dd HH:mm:SS';
        isBetween = ~ (...
                       datetime(sourceDT, 'InputFormat', dtFormat) < datetime(targetMinDT, 'InputFormat', dtFormat) ||... 
                       datetime(sourceDT, 'InputFormat', dtFormat) > datetime(targetMaxDT, 'InputFormat', dtFormat)...
                      );
    end              
 
    targetCoverage = CMAP.get_var_coverage(targetTable, targetVariable);
    targetUnit = CMAP.get_unit(targetTable, targetVariable);
    targetVariableSTD = strcat(targetVariable, '_std');
    for i = 1:height(source)
        df = NaN;
        if in_time_window(source(i, :).time, targetCoverage(1, :).Time_Min, targetCoverage(1, :).Time_Max)
            df = CMAP.space_time(...
                                targetTable,...
                                targetVariable,...
                                shift_dt(source(i, :).time, -timeTolerance),...
                                shift_dt(source(i, :).time, timeTolerance),...
                                source(i, :).lat - latTolerance,...
                                source(i, :).lat + latTolerance,...
                                source(i, :).lon - lonTolerance,...
                                source(i, :).lon + lonTolerance,...
                                source(i, :).depth - depthTolerance,...
                                source(i, :).depth + depthTolerance...
                                );
        end                    
        targetMean = NaN; 
        targetSTD = NaN;
         if istable(df) && height(df) > 0 
             targetMean = nanmean(table2array(df(:, {targetVariable})));
             targetSTD = nanstd(table2array(df(:, {targetVariable})));
         end    
         source(i, {targetVariable}) = num2cell(targetMean);
         source(i, {targetVariableSTD}) = num2cell(targetSTD);        
        fprintf('%d / %d\n', i, height(source)) 
    end            
    source.target_unit(:) = cellstr(strcat('[', targetUnit, ']'));
end