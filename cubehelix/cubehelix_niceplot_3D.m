function cubehelix_niceplot_3D(VARX, VARY, VARZ, VARcol, x)
%% This function does a nice scatter plot of your data

% Getting VARZ ready for plotting.
% Let's say you want to plot VARX VARY VARZ. If VARZ is always a non-null
% integer then you can use it directly in cubehelix by specifying x=1
% otherwise, the function first multiplies VARY by 10^x to make sure its
% variability is maximized after ceiling it to an integer
if nargin<5
    x=3;% here I'm doing *1000 as an example
end

% Make sure you don't have negative values as input for cubehelix or set a
% slide value
Slide = 0;
if sum(VARcol<=0)>0
    Slide = -min(VARcol)+ 10^(-x);
end

% Set up the diameter of the balls
%VAR_diam=0.01;
%Set up the number of micro surfaces of your sphere
%r=30;


VARcol_cube = ceil((VARcol+Slide)*10^x);
%GRAD=cubehelix(max(VARZ_cube));%Note that you can choose the values of...
...your gradient by doing cubehelix_view and then specifying the values in...
    ...the function, for instance: GRAD=cubehelix(max(VARZ_cube),0.5, -0.8, 1.5, 1.7, [1,0]);
GRAD=cubehelix(max(VARcol_cube),0.5, -1.1, 1.5, 0.5, [1,0]);
%figure()
NU = length(VARX); % this is the number of points you have in your plot
for jj=1:NU
    %[xs,ys,zs]=sphere(r);
    %sph=surf(xs*VAR_diam+VARX(jj), ys*VAR_diam+VARY(jj), zs*VAR_diam+VARZ(jj),'EdgeColor','none','LineStyle','none','FaceColor',GRAD(VARcol_cube(jj),:), 'FaceLighting', 'phong','AmbientStrength', 0.5);
    plot3(VARX(jj),VARY(jj),VARZ(jj),'ko','MarkerFaceColor',GRAD(VARcol_cube(jj),:))
    hold on
end
%light('Position', [-1 0 0.5], 'Style', 'infinite')
%% set the limits of the axis to see the spheres well in 3D version
% ZLIM=get(gca, 'ZLim');
% Zrange=ZLIM(2) - ZLIM(1);
% XLIM=get(gca, 'XLim');
% Xrange=XLIM(2) - XLIM(1);
% YLIM=get(gca, 'YLim');
% Yrange=YLIM(2) - YLIM(1);
% Maxrange=max([Zrange, Xrange, Yrange]);
% set(gca,'XLim', [XLIM(1)-(Maxrange-Xrange)/2 XLIM(2)+(Maxrange-Xrange)/2])
% set(gca,'ZLim', [ZLIM(1)-(Maxrange-Zrange)/2 ZLIM(2)+(Maxrange-Zrange)/2])
% set(gca,'YLim', [YLIM(1)-(Maxrange-Yrange)/2 YLIM(2)+(Maxrange-Yrange)/2])
% %set(gca,'Color',[0.2 0.2 0.2]);
% set(gca,'XLim', [0.5 1])
% set(gca,'ZLim', [1.5 2.5])
% set(gca,'YLim', [0 1])


%% Legend on axes
xlabel('VARX')
ylabel('VARY')
zlabel('VARZ')
title(sprintf('My graph with Cubehelix'));
% cc=colorbar();
% colormap(GRAD)
% fprintf('Cubehelix_niceplot had to transform you z data so they can be color ploted\nby multiplying by 10^%d and substracting %f\nNow it is going to change the z colormap to reflect the actual values\n', x,Slide); 
% ChangeZ=input('If you prefer not type "0" otherwise "1"\n');
% if ChangeZ==1
%     YTL = get(cc, 'YTickLabel');
%     YTL_new = str2num(YTL)/10^x - Slide;
%     YTL_new = num2str(round(YTL_new.*10^(x-1))./10^(x-1));
%     set(cc, 'YTickLabel',YTL_new)%here you correct the value of the z axis that you artificially multiplied by 10^x
% end
hold off