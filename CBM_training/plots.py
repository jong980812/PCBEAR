import re
import numpy as np
from matplotlib import pyplot as pl

import colors
import matplotlib.pyplot as pl
import numpy as np
import colorsys

# def complementary_color(rgb):
#     # Convert RGB to HLS
#     h, l, s = colorsys.rgb_to_hls(rgb[0], rgb[1], rgb[2])
#     # Complementary color is 180 degrees (0.5 in HLS scale) around the color wheel
#     h = (h + 0.5) % 1.0
#     # Convert back to RGB
#     return colorsys.hls_to_rgb(h, l, s)

def set_color(feature_inds, values, feature_names):
    color = []
    for j in feature_inds:
        feature_value = values[j]
        if feature_value < 0:
            color.append('y')
        else:
            first_letter = feature_names[j][0]
            if first_letter == 'N':
                first_letter = feature_names[j][4]
            if first_letter == 'O':
                color.append('#e35f62')
            elif first_letter == 'T':
                color.append('dodgerblue')
            elif first_letter == 'S':
                color.append('C2')
            else:
                color.append((230 / 255.0, 230 / 255.0, 250 / 255.0))
    return color

def bar(contributions, feature_names, max_display=5, show=True, title=None, fontsize=13, pos_color=(1, 0, 0)):  # default pos_color is red (1,0,0)
    values = contributions

    # build our auto xlabel based on the transform history of the Explanation object
    xlabel = "Concept contributions"

    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([f'Feature {i}' for i in range(len(values[0]))])

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(feature_names)
    num_features = min(max_display, len(values))
    max_display = min(max_display, num_features)

    orig_inds = [i for i in range(len(values))]

    feature_order = np.argsort(np.abs(values), 0)[::-1]
    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display] # top-k
    y_pos = np.arange(len(feature_inds), 0, -1)

    # build our y-tick labels
    # yticklabels = []
    # for i in feature_inds:
    #     yticklabels.append(feature_names[i])

    # if num_features < len(values):
    #     num_cut = np.sum([1 for i in range(num_features - 1, len(values))])
    #     values[feature_order[num_features - 1]] = np.sum([values[feature_order[i]] for i in range(num_features - 1, len(values))], 0)

    # if num_features < len(values):
    #     yticklabels[-1] = "Sum of %d other features" % num_cut

    # compute our figure size based on how many features we are showing
    row_height = 0.55
    pl.gcf().set_size_inches(8, num_features * row_height + 1.5)  # * np.sqrt(len(values))

    # if negative values are present then we draw a vertical line to mark 0, otherwise the axis does this for us...
    negative_values_present = np.sum(values[feature_order[:num_features]] < 0) > 0
    if negative_values_present:
        pl.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    # draw the bars
    # comp_color = complementary_color(pos_color)
    comp_color = 'y'
    total_width = 0.7
    bar_width = total_width

    pl.barh(
        y_pos, values[feature_inds],
        bar_width, align='center',
        color = set_color(feature_inds, values, feature_names),
    #     color=[
    #     # comp_color if values[j] < 0 else '#e35f62' if feature_names[j][0] == 'O' else 'dodgerblue' if feature_names[j][0] == 'T' else 'C2' if feature_names[j][0] == 'S' else pos_color for j in feature_inds
    # ],
        # color=[comp_color if values[feature_inds[j]] <= 0 else pos_color for j in range(len(y_pos))],
        edgecolor=(1, 1, 1, 0.8), label=None
    )
    yticklabels = []
    for i in feature_inds:
        ft_new = feature_names[i].replace('O:', '').replace('T:', '').replace('S:', '')
        yticklabels.append(ft_new)

    # draw the yticks (the 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks)
    pl.yticks(list(y_pos) + list(y_pos + 1e-8), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=fontsize)

    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen / width

    for j in range(len(y_pos)):
        ind = feature_order[j]
        if values[ind] < 0:
            pl.text(
                values[ind] - (5 / 72) * bbox_to_xscale, y_pos[j], f'{values[ind]:+0.02f}',
                horizontalalignment='right', verticalalignment='center', color="black",  # set text color to black
                fontsize=fontsize
            )
        else:
            pl.text(
                values[ind] + (5 / 72) * bbox_to_xscale, y_pos[j], f'{values[ind]:+0.02f}',
                horizontalalignment='left', verticalalignment='center', color="black",  # set text color to black
                fontsize=fontsize
            )

    # put horizontal lines for each feature row
    for i in range(num_features):
        pl.axhline(i + 1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)

    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    if negative_values_present:
        pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params('x', labelsize=fontsize)

    xmin, xmax = pl.gca().get_xlim()

    if negative_values_present:
        pl.gca().set_xlim(xmin - (xmax - xmin) * 0.1, xmax + (xmax - xmin) * 0.1)
    else:
        pl.gca().set_xlim(xmin, xmax + (xmax - xmin) * 0.1)

    pl.xlabel(xlabel, fontsize=fontsize)
    if title:
        pl.title(title, fontsize=fontsize)

    if show:
        pl.show()


def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s


def bar_percentage(contributions, feature_names, bias, conf, max_display=10, show=True, title=None, fontsize=13):

    values = contributions

    # build our auto xlabel based on the transform history of the Explanation object
    xlabel = "Concept contributions"

    # ensure we at least have default feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(values[0]))])

    # determine how many top features we will plot
    if max_display is None:
        max_display = len(values)
    max_display = min(max_display, len(values))

    
    orig_inds = [i for i in range(len(values))]
    orig_values = values.copy()
    
    feature_order = np.argsort(np.abs(values), 0)[::-1]
    # here we build our feature names, accounting for the fact that some features might be merged together
    feature_inds = feature_order[:max_display]
    y_pos = np.arange(len(feature_inds), 0, -1)
    feature_names_new = []
    for pos,inds in enumerate(orig_inds):
        feature_names_new.append(feature_names[inds])
    feature_names = feature_names_new
    
    # build our y-tick labels
    yticklabels = []
    for i in feature_inds:
        yticklabels.append(feature_names[i])
    
    if max_display < len(values):
        values[feature_order[max_display-1]] = np.sum([values[feature_order[i]] for i in range(max_display-1, len(values))], 0)+bias
    
    if max_display < len(values):
        yticklabels[-1] = "Sum of other concepts"

    # compute our figure size based on how many features we are showing
    row_height = 0.55
    # pl.gcf().set_size_inches(8, max_display * row_height  + 1.5)#* np.sqrt(len(values))

    # if negative values are present then we draw a vertical line to mark 0, otherwise the axis does this for us...
    negative_values_present = np.sum(values[feature_order[:max_display]] < 0) > 0
    if negative_values_present:
        pl.axvline(0, 0, 1, color="#000000", linestyle="-", linewidth=1, zorder=1)

    # draw the bars
    patterns = (None, '\\\\', '++', 'xx', '////', '*', 'o', 'O', '.', '-')
    total_width = 0.7
    bar_width = total_width# / len(values)
    
    
    pl.barh(
        y_pos, values[feature_inds]*conf/(np.sum(orig_values)+bias),
        bar_width, align='center',
        color=[colors.blue_rgb if values[feature_inds[j]] <= 0 else colors.red_rgb for j in range(len(y_pos))],
        hatch=patterns[0], edgecolor=(1,1,1,0.8), label=None
    )

    # draw the yticks (the 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks)
    pl.yticks(list(y_pos) + list(y_pos + 1e-8), yticklabels + [l.split('=')[-1] for l in yticklabels], fontsize=fontsize)

    xlen = pl.xlim()[1] - pl.xlim()[0]
    fig = pl.gcf()
    ax = pl.gca()
    #xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width

    labels = ["{}{:.2f}%".format("+" if value>=0 else "-", value) for value in values*conf/(np.sum(orig_values)+bias)]
    
    for j in range(len(y_pos)):
        ind = feature_order[j]
        if values[ind] < 0:
            pl.text(
                values[ind]*conf/(np.sum(orig_values)+bias) - (5/72)*bbox_to_xscale, y_pos[j], labels[ind],#format_value(values[ind], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=colors.blue_rgb,
                fontsize=fontsize
            )
        else:
            pl.text(
                values[ind]*conf/(np.sum(orig_values)+bias) + (5/72)*bbox_to_xscale, y_pos[j], labels[ind],#format_value(values[ind], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=colors.red_rgb,
                fontsize=fontsize
            )

    # put horizontal lines for each feature row
    for i in range(max_display):
        pl.axhline(i+1, color="#888888", lw=0.5, dashes=(1, 5), zorder=-1)
    
    pl.gca().xaxis.set_ticks_position('bottom')
    pl.gca().yaxis.set_ticks_position('none')
    pl.gca().spines['right'].set_visible(False)
    pl.gca().spines['top'].set_visible(False)
    if negative_values_present:
        pl.gca().spines['left'].set_visible(False)
    pl.gca().tick_params('x', labelsize=fontsize)

    xmin,xmax = pl.gca().get_xlim()
    
    if negative_values_present:
        pl.gca().set_xlim(xmin - (xmax-xmin)*0.1, xmax + (xmax-xmin)*0.1)
    else:
        pl.gca().set_xlim(xmin, xmax + (xmax-xmin)*0.15)
    
    pl.xlabel(xlabel, fontsize=fontsize)
    if title:
        pl.title(title, fontsize=fontsize)
    
    if show:
        pl.show()