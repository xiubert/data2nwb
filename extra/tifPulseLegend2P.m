function pulseLegend2P = tifPulseLegend2P(tif2Pdir, saveFile)
% tifPulseLegend2P  Build a pulse legend struct array from *_Pulses.mat files.
%
%   pulseLegend2P = tifPulseLegend2P(tif2Pdir)
%   pulseLegend2P = tifPulseLegend2P(tif2Pdir, saveFile)
%
%   Scans tif2Pdir for *_Pulses.mat files and extracts pulse metadata.
%   Fields added per entry:
%     tif        - corresponding .tif filename
%     pulseName  - pulse name (trimmed of trailing _N index for single pulses)
%     pulseSet   - pulse set name
%     stimDelay  - stimulus delay (s) from params
%     ISI        - inter-stimulus interval (s) from params
%     xsg        - associated .xsg file(s): string for single pulse,
%                  cell array of strings for map (multi-pulse) files
%
%   saveFile (optional): true/false whether to save pulseLegend2P as
%                        pulseLegend2P.mat in tif2Pdir (default: false).

if nargin < 2
    saveFile = false;
end

if ~isfolder(tif2Pdir)
    warning('tifPulseLegend2P: directory does not exist — select manually.')
    tif2Pdir = uigetdir();
end

dList = dir(tif2Pdir);
pulses = dList(contains({dList.name}, '_Pulses.mat'));

for f = 1:length(pulses)
    pulses(f).tif = strrep(pulses(f).name, '_Pulses.mat', '.tif');

    load(fullfile(pulses(f).folder, pulses(f).name), 'pulse', 'params')

    % --- params fields ---
    pulses(f).stimDelay = params.stimDelay;
    pulses(f).ISI       = params.ISI;

    % --- pulse fields ---
    if size(pulse, 2) > 1
        % map / multi-pulse file — store all pulse names, sets, and xsg files as cell arrays
        pulses(f).pulseName = {pulse.pulsename};
        pulses(f).pulseSet  = {pulse.pulseset};
        pulses(f).xsg       = {pulse.curXSG};
    elseif endsWith(pulse.pulsename, {'_1','_2','_3','_4','_5','_6','_7','_8','_9'})
        % single pulse with trailing index — strip it
        pulses(f).pulseName = pulse.pulsename(1:end-2);
        pulses(f).pulseSet  = pulse.pulseset;
        pulses(f).xsg       = pulse.curXSG;
    else
        pulses(f).pulseName = pulse.pulsename;
        pulses(f).pulseSet  = pulse.pulseset;
        pulses(f).xsg       = pulse.curXSG;
    end

    clear pulse params
end

pulseLegend2P = pulses;

if saveFile
    saveName = fullfile(tif2Pdir, 'pulseLegend2P.mat');
    save(saveName, 'pulseLegend2P', '-v7.3')
    disp(['Saved: ' saveName])
end
