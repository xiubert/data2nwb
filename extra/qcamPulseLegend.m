function pulseLegendQcam = qcamPulseLegend(qcamDir, saveFile)
% qcamPulseLegend  Build a pulse legend struct array from *_Pulses.mat files
%                  for a widefield qcamraw experiment.
%
%   pulseLegendQcam = qcamPulseLegend(qcamDir)
%   pulseLegendQcam = qcamPulseLegend(qcamDir, saveFile)
%
%   Scans qcamDir for *_Pulses.mat files and extracts pulse metadata.
%   Fields added per entry:
%     file       - .qcamraw basename (imaging file key)
%     pulseName  - pulse name (trimmed of trailing _N index for single pulses)
%     pulseSet   - pulse set name
%     stimDelay  - stimulus delay (s) from params
%     ISI        - inter-stimulus interval (s) from params
%     xsg        - associated .xsg file(s): string for single pulse,
%                  cell array of strings for map (multi-pulse) files
%
%   saveFile (optional): true/false whether to save pulseLegendQcam as
%                        pulseLegendQcam.mat in qcamDir (default: false).
%
%   If *_Pulses.mat files are absent, use matchXSG.py --pattern "*.qcamraw"
%   to generate pulseLegendQcam.csv from .xsg timestamps instead.

if nargin < 2
    saveFile = false;
end

if ~isfolder(qcamDir)
    warning('qcamPulseLegend: directory does not exist — select manually.')
    qcamDir = uigetdir();
end

dList = dir(qcamDir);
pulses = dList(contains({dList.name}, '_Pulses.mat'));

for f = 1:length(pulses)
    pulses(f).file = strrep(pulses(f).name, '_Pulses.mat', '.qcamraw');

    load(fullfile(pulses(f).folder, pulses(f).name), 'pulse', 'params')

    % --- params fields ---
    pulses(f).stimDelay  = params.stimDelay;
    pulses(f).ISI        = params.ISI;
    pulses(f).treatment  = '';

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

pulseLegendQcam = pulses;

if saveFile
    saveName = fullfile(qcamDir, 'pulseLegendQcam.mat');
    save(saveName, 'pulseLegendQcam', '-v7.3')
    disp(['Saved: ' saveName])
end
