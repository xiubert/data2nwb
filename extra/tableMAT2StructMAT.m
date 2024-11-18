%%
% helper script to convert saved table in .mat to python friendly struct
% loops over directories matching experiment folder naming scheme: [A-Z]{2}\d{4}
% applies specifically to pulsePupilUVlegend2P.mat

clearvars
close all
filePath = "/media/DATA/backups/sutter2P_backup/D_drive/ZnT3_pupil/BIN1";

aDir = dir(filePath);
aDir = aDir(cell2mat({aDir.isdir}));
aDir = aDir(cellfun(@any,regexp({aDir.name}','[A-Z]{2}\d{4}')));
%%
for aNum = 1:length(aDir)
    disp(['Re-saving pulsePupilUVlegend2P.mat as struct for: ' aDir(aNum).name]);
    if isfolder(fullfile(aDir(aNum).folder,aDir(aNum).name))
        try
            pData = load(fullfile(aDir(aNum).folder,aDir(aNum).name,...
                [aDir(aNum).name ...
                '_pulsePupilUVlegend2P.mat']),'pulsePupilLegend2P');
            disp('loaded!')
            pulsePupilLegend2P = table2struct(pData.pulsePupilLegend2P);
            save(fullfile(aDir(aNum).folder,aDir(aNum).name,...
                [aDir(aNum).name ...
                '_pulsePupilUVlegend2P_s.mat']),'pulsePupilLegend2P')
            clear pData pulsePupilLegend2P
        catch
            disp('no pulsePupilUVlegend2P file found')
        end
    end
end