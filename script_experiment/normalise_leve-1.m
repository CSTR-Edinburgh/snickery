% Old script -- note that this is actually broken -- this line 
%
% y = y ./ max(abs(y)); 
%
% Undoes the work of svtool and scales to -1, 1 again!!!
% See script_experiment/normalise_level.py for a working script
%


sv_tool = '/public/homepages/zwu2/web/listening/tasl_trajectory/sv56demo';

old_dir = '/afs/inf.ed.ac.uk/group/project/dnn_tts/samples/icassp16_lstm/NPH';
new_dir = '/public/homepages/zwu2/web/listening/icassp16/NPH';

mkdir(new_dir);
    dir_data = dir(fullfile(old_dir, '*.wav'));
    file_list = {dir_data.name}';
    
    for j = 1:length(file_list)
        old_wav_file = sprintf('%s/%s', old_dir, file_list{j});
         [x, fs] = wavread(old_wav_file);
         old_raw_file = sprintf('%s/temp.raw', old_dir);
         y = (x/max(abs(x))) * 32707;
         fid = fopen(old_raw_file, 'w');
         fwrite(fid, y, 'int16');
         fclose(fid);
        new_raw_file = sprintf('%s/temp1.raw', new_dir);
        cmd = sprintf('%s -sf 48000 %s %s', sv_tool, old_raw_file, new_raw_file);
        system(cmd);
        fid = fopen(new_raw_file);
        y = fread(fid, inf, 'int16');
        fclose(fid);
        y = y ./ max(abs(y));
        new_wav_file = sprintf('%s/%s', new_dir, file_list{j});
        wavwrite(y, 48000, 16, new_wav_file);
    end

