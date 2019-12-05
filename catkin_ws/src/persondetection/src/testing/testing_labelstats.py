#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import scipy.stats
from mpl_toolkits.axes_grid1 import make_axes_locatable

from utils.boxes import *
from utils.transformbox import get_boxes_zedframe
from connected_components.connectedcomponents import *

SLOPE = {}

folder = "src/persondetection/src/testing/"
cam = "zed"

SMALL_SIZE = 8
MEDIUM_SIZE = 10
BIGGER_SIZE = 12


def get_random_dataframe():
    df = pd.DataFrame(np.random.randint(1,1000,size=(2000, 4))/1000, columns=['y','x','width','height'])
    return df


def get_dataframe(label_folder=cam):
    rows_list = []
    #<object-class> <x> <y> <width> <height>
    for txtfile in glob.glob(folder + "test_annotations/" + label_folder + "/*.txt"):

        with open(txtfile, "r") as txt:
            for line in txt:
                box = line.strip("\n").split(" ")
                if box[0] == '0': #person caltegory
                    x = float(box[1])
                    y = float(box[2])
                    height = float(box[4])
                    width = float(box[3])
                    if height < 0.01: # incorrect annotation
                        print(height,txtfile)
                        continue
                    dict_row = {'y':y, 'x':x, 'width':width, 'height':height}
                    rows_list.append(dict_row)

    if len(rows_list):
        df = pd.DataFrame(rows_list)
        #print(df)
        return df


def run_regression_fit(df=get_dataframe(),xy='y_height',plot=False):
    global SLOPE, INTERCEPT
    if xy == 'y_height':
        x = 'y'
        y = 'height'
    elif xy == 'aspect_ratio':
        y = 'width'
        x = 'height'
    if plot:
        ax = sns.regplot(x=df[x],y=df[y],data=df)
        ax.set_ylim(0,max(df['height']))
        ax.set_xlim(0,max(df['y']))
        ax.set_ylabel('Bounding box ' + y)
        ax.set_xlabel('Bounding box ' + x)
        plt.xlabel('Bounding box ' + x, fontsize=16)
        plt.ylabel('Bounding box ' + y, fontsize=16)
        plt.savefig(folder + "results/regression/" + xy + "_regression_" + cam + ".png", bbox_inches='tight')
        plt.show()

        ax = sns.residplot(x=x, y=y, data=df,
                  scatter_kws={"s": 80});
        plt.xlabel('Bounding box ' + x, fontsize=16)
        plt.ylabel('Residuals ' + y, fontsize=16)

        plt.savefig(folder + "results/regression/"+xy+"_regression_"+cam+"_resid.png", bbox_inches='tight')
        plt.show()
    slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(df[x], y=df[y])
    print(scipy.stats.linregress(df[x], y=df[y]))
    SLOPE[xy] = slope
    INTERCEPT[xy] = intercept


def test_confidence_score_colorbar(df=get_dataframe(),xy='y_height'):
    if xy == 'y_height':
        x = 'y'
        y = 'height'
    elif xy == 'aspect_ratio':
        y = 'width'
        x = 'height'
    df['score'] = df.apply(lambda row: calculate_confidence_score(row[x], row[y], SLOPE[xy], INTERCEPT[xy]), axis=1)
    #Create a matplotlib colormap from the sns seagreen color palette
    cmap    = sns.light_palette("seagreen", reverse=False, as_cmap=True )
    # Normalize to the range of possible values from df["c"]
    norm = matplotlib.colors.Normalize(vmin=0, vmax=df["score"].max())
    # create a color dictionary (value in c : color from colormap)
    colors = {}
    for cval in df["score"]:
        colors.update({cval : cmap(norm(cval))})

    #create a figure
    fig = plt.figure(figsize=(5,5))
    #plot the swarmplot with the colors dictionary as palette
    ax = sns.scatterplot(x=df[x],y=df[y],hue='score',data=df,palette=colors)
    plt.gca().legend_.remove()
    ax.set_ylim(0,1)
    ax.set_xlim(0,1)
    ax.set_ylabel('Bounding box ' + y)
    ax.set_xlabel('Bounding box ' + x)
    ## create colorbar ##
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)
    fig.add_axes(ax_cb)
    cb1 = matplotlib.colorbar.ColorbarBase(ax_cb, cmap=cmap,
                                    norm=norm,
                                    orientation='vertical')
    cb1.set_label('Confidence score')
    plt.savefig(folder + "results/regression/" + xy + "_scorerandom_" + cam + ".png", bbox_inches='tight')
    plt.show()


run_regression_fit(get_dataframe(),xy='aspect_ratio',plot=True)
test_confidence_score_colorbar(get_random_dataframe(), xy='aspect_ratio')

run_regression_fit(get_dataframe(),xy='y_height',plot=True)
test_confidence_score_colorbar(get_random_dataframe(), xy='y_height')
