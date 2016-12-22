function [legend_h,object_h,plot_h,text_strings] = columnlegend(numcolumns, str, varargin)
%
%   columnlegend creates a legend with a specified number of columns.
%   
%   columnlegend(numcolumns, str, varargin)
%       numcolumns - number of columns in the legend
%       str - cell array of strings for the legend
%       
%   columnlegend(..., 'Location', loc)
%       loc - location variable for legend, default is 'NorthEast'
%                  possible values: 'NorthWest', 'NorthEast', 'SouthEast', 'SouthWest', 
%                                   'NorthOutside', 'SouthOutside',
%                                   'NortheastOutside', 'SoutheastOutside'
%
%   columnlegend(..., 'boxon')
%   columnlegend(..., 'boxoff')
%        set legend bounding box on/off
%
%   example:
%      plot(bsxfun(@times, [0:9]',[1:10])); 
%      columnlegend(3, cellstr(num2str([1:10]')), 'location','northwest');
%
%
%   Author: Simon Henin <shenin@gc.cuny.edu>
%   
%   4/09/2013 - Fixed bug with 3 entries / 3 columns
%   4/09/2013 - Added bounding box option as per @Durga Lal Shrestha (fileexchage)
%   11 May 2010 - 1.2 Add instructions for printing figure with columns
%   08 Feb 2011 - 1.4 Added functionality when using markers.
%   31 Oct 2015 - Updates for compatibility with 2015a, Adds minor improvements as per user suggestions
%   07 Nov 2016 - Bug fixes, added functionality for bar plots, added all valid legend locations 


location = 'NorthEast';
boxon = false; legend_h = false;
for i=1:2:length(varargin),
    switch lower(varargin{i})
        case 'location'
            location = varargin{i+1};
            i=i+2;
        case 'boxon'
            boxon = true;
        case 'boxoff'
            boxon = false;
        case 'legend'
            legend_h = varargin{i+1};
            i=i+2;
        case 'object'
            object_h = varargin{i+1};
            i=i+2;
    end
end

if legend_h == false,
    %create the legend
    [legend_h,object_h,plot_h,text_strings] = legend(str);
end

%some variables
numlines = length(str);
numpercolumn = ceil(numlines/numcolumns);

%get old width, new width and scale factor
set(legend_h, 'units', 'normalized');
set(gca, 'units', 'normalized');

pos = get(legend_h, 'position');
width = numcolumns*pos(3);
newheight = (pos(4)/numlines)*numpercolumn;
rescale = pos(3)/width;


%get some old values so we can scale everything later
type = get(object_h(numlines+1), 'type');
switch type,
    case {'line'}
        xdata = get(object_h(numlines+1), 'xdata');
        ydata1 = get(object_h(numlines+1), 'ydata');
        ydata2 = get(object_h(numlines+3), 'ydata');
        
        %we'll use these later to align things appropriately
        sheight = ydata1(1)-ydata2(1);                  % height between data lines
        height = ydata1(1);                             % height of the box. Used to top margin offset
        line_width = (xdata(2)-xdata(1))*rescale;       % rescaled linewidth to match original
        spacer = xdata(1)*rescale;                      % rescaled spacer used for margins
    case {'hggroup'}
        text_pos    = get(object_h(1), 'position');
        child       = get(object_h(numlines+1), 'children');
        vertices_1  = get(child, 'vertices');
        child       = get(object_h(numlines+2), 'children');
        vertices_2  = get(child, 'vertices');
        sheight     = vertices_1(2,2)-vertices_1(1,2);
        height      = vertices_1(2,2);
        line_width  = (vertices_1(3,1)-vertices_1(1,1))*rescale;       % rescaled linewidth to match original
        spacer      = vertices_1(1,2)-vertices_2(2,2);                      % rescaled spacer used for margins
        text_space  = (text_pos(1)-vertices_1(4,1))./numcolumns;      
end


%put the legend on the upper left corner to make initial adjustments easier
% set(gca, 'units', 'pixels');
loci = get(gca, 'position');
set(legend_h, 'position', [loci(1) pos(2) width pos(4)]);

col = -1;
for i=1:numlines,
    if (mod(i,numpercolumn)==1 || (numpercolumn == 1)),
        col = col+1;
    end
    
    if i==1
        linenum = i+numlines;
    else
        if strcmp(type, 'line'),
            linenum = linenum+2;
        else
            linenum = linenum+1;
        end
    end
    labelnum = i;
    
    position = mod(i,numpercolumn);
    if position == 0,
         position = numpercolumn;
    end
    
    switch type,
        case {'line'}
            %realign the labels
            set(object_h(linenum), 'ydata', [(height-(position-1)*sheight) (height-(position-1)*sheight)]);
            set(object_h(linenum), 'xdata', [col/numcolumns+spacer col/numcolumns+spacer+line_width]);
            
            set(object_h(linenum+1), 'ydata', [height-(position-1)*sheight height-(position-1)*sheight]);
            set(object_h(linenum+1), 'xdata', [col/numcolumns+spacer*3.5 col/numcolumns+spacer*3.5]);
            
            set(object_h(labelnum), 'position', [col/numcolumns+spacer*2+line_width height-(position-1)*sheight]);
        case {'hggroup'},
            child = get(object_h(linenum), 'children');
            v = get(child, 'vertices');
            %x-positions
            v([1:2 5],1) = col/numcolumns+spacer;
            v(3:4,1) = col/numcolumns+spacer+line_width;
            % y-positions
            v([1 4 5],2) = (height-(position-1)*sheight-(position-1)*spacer);
            v([2 3], 2) = v(1,2)+sheight;
            set(child, 'vertices', v);
            set(object_h(labelnum), 'position', [v(3,1)+text_space v(1,2)+(v(2,2)-v(1,2))/2 v(3,1)-v(1,1)]);
    end
   
end

%unfortunately, it is not possible to force the box to be smaller than the
%original height, therefore, turn it off and set background color to none
%so that it no longer appears
set(legend_h, 'Color', 'None', 'Box', 'off');

%let's put it where you want it
fig_pos = get(gca, 'position');
pos = get(legend_h, 'position');
padding = 0.01; % padding, in normalized units
% if location is some variation on south, then we need to take into account
% the new height
if strfind(location, 'south'),
    h_diff = pos(4)-newheight;
    pos(4) = newheight;
end
switch lower(location),
    case {'northeast'}
        set(legend_h, 'position', [pos(1)+fig_pos(3)-pos(3)-padding pos(2) pos(3) pos(4)]);
    case {'northwest'}
        set(legend_h, 'position', [pos(1)+padding pos(2) pos(3) pos(4)]);        
    case {'southeast'}
        pos(4) = newheight;
        set(legend_h, 'position', [pos(1)+fig_pos(3)-pos(3)-padding fig_pos(2)-pos(4)/2+pos(4)/4 pos(3) pos(4)]);
    case {'southwest'}
        set(legend_h, 'position', [fig_pos(1)+padding fig_pos(2)-pos(4)/2+pos(4)/4 pos(3) pos(4)]);
    case {'northeastoutside'}
        % need to resize axes to allow legend to fit in figure window
        set(gca, 'position', [fig_pos]-[0 0 pos(3) 0]);
        set(legend_h, 'position', [pos(1)+fig_pos(3)-pos(3) pos(2) pos(3) pos(4)]);
    case {'northwestoutside'}
        % need to resize axes to allow legend to fit in figure window
        set(gca, 'position', [fig_pos]+[pos(3) 0 -pos(3) 0]);
        set(legend_h, 'position', [fig_pos(1)-fig_pos(3)*.1 pos(2) pos(3) pos(4)]); % -10% figurewidth to account for axis labels
    case {'north'}
        % need to resize axes to allow legend to fit in figure window
        set(legend_h, 'position', [fig_pos(1)+fig_pos(3)/2-pos(3)/2 fig_pos(2)+(fig_pos(4)-pos(4))-padding pos(3) pos(4)]);    
    case {'northoutside'}
        % need to resize axes to allow legend to fit in figure window
        set(gca, 'position', [fig_pos]-[0 0 0 pos(4)]);
        set(legend_h, 'position', [fig_pos(1)+fig_pos(3)/2-pos(3)/2 fig_pos(2)+(fig_pos(4)-pos(4)) pos(3) pos(4)]);
    case {'south'}
        y_pos = fig_pos(2)-h_diff+pos(4);
        set(legend_h, 'position', [fig_pos(1)+fig_pos(3)/2-pos(3)/2  y_pos pos(3) pos(4)]);
    case {'southoutside'}
        % need to resize axes to allow legend to fit in figure window
        set(gca, 'position', [fig_pos]-[0 -pos(4) 0 pos(4)]);
        set(legend_h, 'position', [fig_pos(1)+fig_pos(3)/2-pos(3)/2 fig_pos(2)-pos(4)-pos(3)*0.1 pos(3) pos(4)]);
    case {'eastoutside'}
        % need to resize axes to allow legend to fit in figure window
        set(gca, 'position', [fig_pos]-[0 0 pos(3) 0]);
        set(legend_h, 'position', [pos(1)+fig_pos(3)-pos(3) fig_pos(2)+fig_pos(4)/2-pos(4)/2 pos(3) pos(4)]);
    case {'southeastoutside'}
        % need to resize axes to allow legend to fit in figure window
        set(gca, 'position', [fig_pos]-[0 0 pos(3) 0]);
        set(legend_h, 'position', [pos(1)+fig_pos(3)-pos(3) fig_pos(2)-pos(4)/4 pos(3) pos(4)]);
    case {'westoutside'}
        % need to resize axes to allow legend to fit in figure window
        set(gca, 'position', [fig_pos]+[pos(3) 0 -pos(3) 0]);
        set(legend_h, 'position', [fig_pos(1)-fig_pos(3)*.1 fig_pos(2)+fig_pos(4)/2-pos(4)/2 pos(3) pos(4)]); % -10% figurewidth to account for axis labels    
    case {'southwestoutside'}
        % need to resize axes to allow legend to fit in figure window
        set(gca, 'position', [fig_pos]+[pos(3) 0 -pos(3) 0]);
        set(legend_h, 'position', [fig_pos(1)-fig_pos(3)*.1 fig_pos(2)-pos(4)/4 pos(3) pos(4)]); % -10% figurewidth to account for axis labels        
end

% display box around legend
if boxon,
    drawnow; % make sure everyhting is drawn in place first.
%     set(legend_h, 'units', 'normalized');
    pos = get(legend_h, 'position');
    orgHeight = pos(4);
    pos(4) = (orgHeight/numlines)*numpercolumn;
    pos(2)=pos(2) + orgHeight-pos(4) - pos(4)*0.05;
    pos(1) = pos(1)+pos(1)*0.01;
    annotation('rectangle',pos, 'linewidth', 1)
end

% re-set to normalized so that things scale properly
set(legend_h, 'units', 'normalized');
set(gca, 'units', 'normalized');