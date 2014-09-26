function cubehelix_view(start,rotations,hue,gamma,rng)
% Create an interactive figure for Cubehelix colormap parameter selection.
%
% (c) 2013 Stephen Cobeldick
%
% Syntax:
%  cubehelix_view
%  cubehelix_view(start,rotations,hue,gamma)
%  cubehelix_view(start,rotations,hue,gamma,rng)
%
% View any of Dave Green's Cubehelix colorschemes: shows the chosen scheme
% as both color and grayscale images/colorbars, and a 2D line plot with
% both the RGB values together with the equivalent grayscale values.
%
% Six sliders allow real-time interactive adjustment of the Cubehelix
% parameter values, which are also displayed. The parameters can also be
% set/reset by calling the function with these values as input arguments.
%
% The scheme is defined here: http://astron-soc.in/bulletin/11June/289392011.pdf
% For more information and examples: http://www.mrao.cam.ac.uk/~dag/CUBEHELIX/
%
% See also CUBEHELIX CUBEHELIX_FIND BREWERMAP RGBPLOT3 RGBPLOT COLORMAP COLORMAPEDITOR COLORBAR UICONTROL ADDLISTENER
%
% ### Input Arguments ###
%
% Inputs (*=default):
%  start = NumericScalar, *0.5, the start color, with R=1, G=2, B=3 etc. (modulus 3).
%  rots  = NumericScalar, *-1.5, the number of R->G->B rotations over the scheme length.
%  hue   = NumericScalar, *1, controls how saturated the colors are.
%  gamma = NumericScalar, *1, can be used to emphasize low or high intensity values.
%  rng   = NumericVector, *[0,1], brightness levels of the colormap's endnodes. Size 1by2.
%
% cubehelix_view(start,rotations,hue,gamma,rng)

switch nargin
    case 0
        chv_updt(0.5,-1.5,1,1,[0,1])
    case 4
        chv_updt(start,rotations,hue,gamma,[0,1])
    case 5
        chv_updt(start,rotations,hue,gamma,rng)
    otherwise
        error('Wrong number of inputs. Parameters may be entered in a vector or individually.')
end
%
end
%----------------------------------------------------------------------END:cubehelix_view
function chv_updt(varargin)
% Draw a new figure or update an existing figure. Also slider callback handling.
%
persistent pltA pltL imgA imgI uicS txtH
%
% LHS and RHS slider bounds/limits and step sizes:
lb = [0,-3, 0, 0, 0, 0];
ub = [3, 3, 3, 3, 1, 1];
sp = [1, 1, 1, 1, 1, 1;...
      5, 5, 5, 5, 2, 2].'./10;
%
if nargin==2 % Slider callback
    % Get slider values:
    blah = arrayfun(@(h)get(h,'Value'),uicS, 'UniformOutput',false);
    varargin = [blah(1:4),[blah{5:6}]];
elseif nargin==5 % Function call
    % Create a new figure:
    if isempty(pltL) || ~all(ishghandle(pltL))
        [pltA,pltL,imgA,imgI,uicS,txtH] = chv_plot(lb,ub,sp);
    end
    % Update slider positions:
    blah = [varargin(1:4),varargin{5}(1),varargin{5}(2)];
    arrayfun(@(h,v,n,x)set(h,'Value',max(n,min(x,v))),uicS,[blah{:}],lb,ub);
else
    error('This should not happen... ')
end
%
% Update text:
for m = 1:6
    set(txtH(m),'String',num2str(blah{m}));
end
% Update XData and axes limits:
N = 128+round(pow2(log2(1+abs(blah{2})),6));
set(pltA, 'XLim',[1,N]);
arrayfun(@(h)set(h, 'XData',1:N), pltL);
arrayfun(@(h)set(h, 'YLim', [0,N]+0.5), imgA);
% Update images/colorbars and line data values:
map = cubehelix(N,varargin{:});
mag = sum(map*[0.298936;0.587043;0.114021],2);
set(imgI(1), 'CData',reshape(map,N,[],3))
set(imgI(2), 'CData',repmat(mag,[1,1,3]))
set(pltL(1), 'YData',map(:,1))
set(pltL(2), 'YData',map(:,2))
set(pltL(3), 'YData',map(:,3))
set(pltL(4), 'YData',mag)
%
end
%----------------------------------------------------------------------END:chv_updt
function [pltA,pltL,imgA,imgI,uicS,txtH] = chv_plot(lb,ub,sp)
% Draw a new figure with RGBplot axes, ColorBar axes, and uicontrol sliders.
%
% Parameter names:
names = {'start';'rotations';'hue';'gamma';'range_L';'range_R'};
gap = 0.01;
hgt = 0.70;
lft = 0.20;
rgt = 0.24;
wdt = 1-lft-rgt-2*gap;
brh = (1-gap-hgt)/numel(names)-gap;
%
% RGBplot gray color:
gry = 0.7*[1,1,1];
% RGBplot X-values:
X = (1:2).'*[1,1,1,1];
%
figH = figure('HandleVisibility','callback', 'NumberTitle','off',...
    'Name','Cubehelix Interactive Parameter Selector');
pltA = axes('Parent',figH, 'Position',[gap,1-hgt+gap,lft+wdt,hgt-2*gap],...
    'ColorOrder',[1,0,0;0,1,0;0,0,1;gry], 'XTick',[], 'YTick',[], 'YLim',[0,1]);
pltL = line(X,X, 'Parent',pltA);
%
blah = [0,0,0,0,0,0];
txtH = [0,0,0,0,0,0];
uicS = [0,0,0,0,0,0];
for n = 1:6
    Y = gap+(6-n)*(brh+gap);
    uicS(n) = uicontrol(figH,'Style','slider', 'Units','normalized',...
        'Position',[lft+gap,Y,wdt,brh], 'Min',lb(n), 'Max',ub(n),...
        'SliderStep',sp(n,:)/(ub(n)-lb(n)));
    addlistener(uicS(n), 'Value', 'PostSet', @chv_updt);
    %
    blah(n) = uicontrol(figH,'Style','text', 'Units','normalized',...
        'Position',[gap,Y,lft/2,brh],'String',names{n});
    txtH(n) = uicontrol(figH,'Style','text', 'Units','normalized',...
        'Position',[gap+lft/2,Y,lft/2,brh],'String','X');
end
%
C = reshape([1,1,1],1,[],3);
imgA(1) = axes('Parent',figH, 'Visible','off', 'Units','normalized',...
    'Position',[1-rgt/1,gap,rgt/2-gap,1-2*gap], 'YLim',[0.5,1.5]);
imgA(2) = axes('Parent',figH, 'Visible','off', 'Units','normalized',...
    'Position',[1-rgt/2,gap,rgt/2-gap,1-2*gap], 'YLim',[0.5,1.5]);
imgI(1) = image('Parent',imgA(1), 'CData',C);
imgI(2) = image('Parent',imgA(2), 'CData',C);
%
end
%----------------------------------------------------------------------END:chv_plot