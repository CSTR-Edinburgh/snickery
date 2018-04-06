function [mix] = addsigs(s,n,targetsnr,interval)
% Add signals in style of Hurricane challenge (scale speech signal and add it to noise at a specified SNR calculated in a specified interval)
% Cassia VB
% 19/10/17
%
% Usage [mix,nout] = addsigs(s,n,targetsnr)
% Input
%    s            vector containing speech
%    n            vector containing masker
%    targetsnr    signal-to-noise ratio required
%    interval     a vector of two elements containing t1 and t2 (see eq. 1 in [1]) in samples
%
% Output
%    mix          mixture signal
%
% [1] Martin Cooke, Catherine Mayo, Cassia Valentini-Botinhao, Yannis Stylianou, Bastian Sauert, Yan Tang, 'Evaluating the intelligibility benefit of speech modifications in known noise conditions'

s=s(:)'; n=n(:)';

% If no interval is provided consider the whole interval of the signal
if nargin < 4
    interval = 1:length(s);
else
    interval = interval(1):interval(2);
end

% Find scale factor
k    = sqrt( (sum(n(interval).^2)/ sum(s(interval).^2)) * power(10,targetsnr/10) ) ;

% Apply scale factor
sout = k*s;

% Sum signals
mix  = sout + n;

% Check for clipping
if max(abs(mix)) > 1
    disp('Clipping occured, noise signal has to be attenuated before adding it to speech');
end