function cubehelix_demo
% Interesting Cubehelix colorschemes are shown in a <cubehelix_view> slideshow.
%
% (c) 2013 Stephen Cobeldick
%
% See also CUBEHELIX CUBEHELIX_VIEW CUBEHELIX_FIND BREWERMAP RGBPLOT3 RGBPLOT COLORMAP

V = {0.5, 0.0,1,1,  1, 0:0.1:2.99;... % Vary start ( 0 rotations).
     0.5,-1.0,1,1,  1, 0:0.1:2.99;... % Vary start (-1 rotations).
     0.5,-1.0,1,1,  2, -5:0.2:5;...   % Vary rotations.
     0.5,-1.5,1,1,  3, -1:0.1:3;...   % Vary hue.
     0.5,-1.0,1,1,  4, 0.1:0.1:2.5};  % Vary gamma.
%
% Show selection of Cubehelix colorschemes:
for i = 1:size(V,1)
    for j = V{i,6}
        V{i,V{i,5}} = j;
        cubehelix_view(V{i,1:4});
        pause(0.1)
    end
end
% Return to default Cubehelix colorscheme:
cubehelix_view
%
end
%----------------------------------------------------------------------END:cubehelix_demo