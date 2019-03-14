function [KDE,t,Error]=kde_wrapper(SpikeTimesIn,t,Response_samprate,Weight)
% SpikeTimes are the spike arrival times in s
% t is the time points at which the density function should be calulated in
% seconds
% Response_samprate is the frequency of sampling of the density function in Hz.
% It is 1/Step with Step = duration between 2 time points in
% t in s.

% KDE is given in spike/s or Hz if all the inputs are in s and Hz, KDE is
% in spike/ms if all the input are given in ms and kHz

% Weight is a vector of the same size as t that indicates the number of
% events ate each time point over which the KDE was calculated. If the same
% number of events were used at each and all time points then Weight can be
% a scalar. By Default, Weight =1.

if nargin<4
    Weight=1;
end
if length(Weight)==1
    Weight = Weight*ones(1,length(t));
elseif length(Weight)~=length(t)
    error('Weight should either be a scalar or a vector of the same size as t\n')
end

% If there is no input spike our best estimate of the spike rate density
% during the whole time span of t is 1/2 spike so within a bin, it is
% 1/(2*number of time bins)
if isempty(SpikeTimesIn)
    y=ones(1,length(t))./(2*length(t));
    KDE =  y * Response_samprate ./ Weight;
    Error = nan(2,length(t)); % For now the code does not calculate an error when there is no input spike
else
    % If there is only one spike our best estimate of the spike rate density
    % during the whole time span of t is 1 spike so within a bin, it is 1/number of time bins
    if length(SpikeTimesIn)==1
        y=ones(1,length(t))./length(t);
        KDE =  y .* length(SpikeTimesIn) .* Response_samprate ./Weight;
        Error = nan(2,length(t)); % For now the code does not calculate an error when there is no input spike
    else
        % calculate the density estimate
        [y,t,~,~,~,bconf95]=sskernel(SpikeTimesIn,t);
        % y is a density function that sums to 1
        % multiplying by the total number of spikes gives the number of expecting spike per time bin (here 10 ms)
        % multiplying by the response sampling rate in kHz gives the expected spike rate to one stimulus presentation in spike/ms
        % Make sure y sum to 1
        y_scale = y/sum(y);
        KDE =  y_scale .* length(SpikeTimesIn) .* Response_samprate ./Weight;
        Error = abs(flipud(bconf95) ./ sum(y) .* length(SpikeTimesIn) .* Response_samprate ./repmat(Weight,2,1) - repmat(KDE,2,1));
    end
    
end
end