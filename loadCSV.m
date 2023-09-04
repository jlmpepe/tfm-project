function output = loadCSV(filename,delimiter)
% Reads CSV from a filename
train = readtable(filename, 'Delimiter', delimiter, 'VariableNamingRule','preserve');
num_rows = height(train);

for i = 2:length(train.Properties.VariableNames)
    temp = cell(num_rows, 1);
    for j = 1:num_rows
        temp(j) = {str2num(char(table2cell(train(j, i))))};
    end
    train(:, i) = temp;
end

output = train;
end

