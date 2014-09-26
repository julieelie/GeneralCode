function cubehelix_niceplot(VARX, VARY, VARZ, x)
%% This function does a nice scatter plot of your data

% Getting VARZ ready for plotting.
% Let's say you want to plot VARX VARY VARZ. If VARZ is always a non-null
% integer then you can use it directly in cubehelix by specifying x=1
% otherwise, the function first multiplies VARY by 10^x to make sure its
% variability is maximized after ceiling it to an integer
if nargin<4
    x=3;% here I'm doing *1000 as an example
end

% Make sure you don't have negative values as input for cubehelix or set a
% slide value
Slide = 0;
if sum(VARZ<=0)>0
    Slide = -min(VARZ)+ 10^(-x);
end


VARZ_cube = ceil((VARZ+Slide)*10^x);
%GRAD=cubehelix(max(VARZ_cube));%Note that you can choose the values of...
...your gradient by doing cubehelix_view and then specifying the values in...
    ...the function, for instance: GRAD=cubehelix(max(VARZ_cube),0.5, -0.8, 1.5, 1.7, [1,0]);
GRAD=cubehelix(max(VARZ_cube),0.5, -1.1, 1.5, 0.5, [1,0]);
%figure()
NU = length(VARX); % this is the number of points you have in your plot
for jj=1:NU
    plot(VARX(jj), VARY(jj),'ko', 'MarkerFaceColor',GRAD(VARZ_cube(jj),:)); % for each dot you call the line in the gradient of color that correspond to the value in Z
    hold on
end
xlabel('VARX')
ylabel('VARY')
title(sprintf('My graph with Cubehelix'));
cc=colorbar();
colormap(GRAD)
fprintf('Cubehelix_niceplot had to transform you z data so they can be color ploted\nby multiplying by 10^%d and substracting %f\nNow it is going to change the z colormap to reflect the actual values\n', x,Slide); 
ChangeZ=input('If you prefer not type "0" otherwise "1"\n');
if ChangeZ==1
    YTL = get(cc, 'YTickLabel');
    YTL_new = str2num(YTL)/10^x - Slide;
    YTL_new = num2str(round(YTL_new.*10^(x-1))./10^(x-1));
    set(cc, 'YTickLabel',YTL_new)%here you correct the value of the z axis that you artificially multiplied by 10^x
end
hold off