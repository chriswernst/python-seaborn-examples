#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 09:11:48 2018

@author: ChrisErnst

Tutorial on Seaborn
"""
%matplotlib
# %matplotlib inline # to print plots in the prompt

import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# https://seaborn.pydata.org/tutorial/aesthetics.html

np.random.seed(sum(map(ord, "aesthetics")))

def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)
        
sinplot()  # plots as white, full box around, tickmarks

sns.set() # Sets seaborn defaults
sinplot() 

sns.set_style("white") # changes background to white no tickmarks
sinplot() 

sns.set_style("whitegrid") # changes background to white with gridlines
sinplot() 

sns.set_style("dark") # changes background to dark, no ticks
sinplot() 

sns.set_style("darkgrid") # sns.setchanges background to dark with gridlines
sinplot() 

sns.set_style("ticks") # changes back to initial setup from import - background to white with tickmarks
sinplot() 


# My preferred aesthetic is:
sns.set_style("white") # changes background to white no tickmarks
sinplot()
sns.despine() # Removes spines from upper and right

# Despine select areas
sns.despine() # Removes top and right spines
sns.despine(left=True, bottom=True) # Removes left and bottom spines


# Gives temporary style
with sns.axes_style("darkgrid"):
    plt.subplot(211)
    sinplot()
with sns.axes_style("white"):    
    plt.subplot(212)
    sinplot(-1)
    sns.despine()

# Set individual settings of the chart
sns.axes_style() # Shows all of the current settings
sns.set_style({"axes.facecolor": "#00ffff"}) # Sets the backgroudn to HTML hex color Aqua # https://www.w3schools.com/colors/colors_picker.asp
sinplot()

sns.set() # reset to defaults
sinplot()

# Set the font size, line thickness for different presentation contexts
sns.set_context("paper") # Options are 'paper', 'notebook', 'talk', 'poster'
sinplot()

sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5}) # Set specifics


# many(if not all) of the matplotlib pyplot function should work with charts. For instance:
plt.xlabel('') # remove x and y labels
plt.ylabel('') # remove x and y labels

plt.savefig('test.png')


# Boxplot
sns.set_style("whitegrid")
data = np.random.normal(size=(20, 6)) + np.arange(6) / 2
sns.boxplot(data=data)

# Violin Plot
f, ax = plt.subplots()
sns.violinplot(data=data)
sns.despine(offset=10, trim=True)




# https://seaborn.pydata.org/tutorial/color_palettes.html#palette-tutorial

# Color Palletes
current_palette = sns.color_palette()
sns.palplot(current_palette) #  the six themes are deep, muted, pastel, bright, dark, and colorblind.

sns.palplot(sns.color_palette("hls", 8)) # 8 color palette
sns.palplot(sns.hls_palette(8, l=0.85, s=0.9)) # setting the saturation and lightness

sns.palplot(sns.color_palette("husl", 8)) # a more uniform palette

sns.palplot(sns.color_palette("Paired"))

sns.palplot(sns.color_palette("Set2", 10))

# Set our own list of colors:
flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
sns.palplot(sns.color_palette(flatui))

# Set individual line colors to xkcd_rbg color lists
sns.set()
plt.plot([0, 1], [0, 1], sns.xkcd_rgb["pale red"], lw=3)
plt.plot([0, 1], [0, 2], sns.xkcd_rgb["medium green"], lw=3)
plt.plot([0, 1], [0, 3], sns.xkcd_rgb["denim blue"], lw=3)

# Sequences of colors
sns.palplot(sns.color_palette("Blues"))
sns.palplot(sns.color_palette("BuGn_r"))
sns.palplot(sns.color_palette("GnBu_d"))

# Sequential “cubehelix” palettes
sns.palplot(sns.color_palette("cubehelix", 8))
sns.palplot(sns.cubehelix_palette(8))
sns.palplot(sns.cubehelix_palette(8, start=.1, rot=-.75)) # 8 colors, start is between 0-3, rot is between -1 and 1
sns.palplot(sns.cubehelix_palette(8, start=2, rot=0, dark=0, light=.95, reverse=True)) # Reverse the ordering


# Create topical maps
x, y = np.random.multivariate_normal([0, 0], [[1, -.5], [-.5, 1]], size=300).T
cmap = sns.cubehelix_palette(light=1, as_cmap=True)
sns.kdeplot(x, y, cmap=cmap, shade=True)

# Outlined topical maps
pal = sns.dark_palette("palegreen", as_cmap=True)
sns.kdeplot(x, y, cmap=pal)

# Light or Dark Sequential Colors
sns.palplot(sns.light_palette("green"))
sns.palplot(sns.dark_palette("red"))
sns.palplot(sns.light_palette("blue"))

sns.palplot(sns.light_palette((210, 90, 60), input="husl"))
sns.palplot(sns.dark_palette("muted purple", input="xkcd"))

# Heatmap colors
sns.palplot(sns.color_palette("BrBG", 7))
sns.palplot(sns.color_palette("RdBu_r", 7))
sns.palplot(sns.color_palette("coolwarm", 7))

sns.palplot(sns.diverging_palette(220, 20, n=7))
sns.palplot(sns.diverging_palette(145, 280, s=85, l=25, n=7))
sns.palplot(sns.diverging_palette(10, 220, sep=80, n=7))
sns.palplot(sns.diverging_palette(255, 133, l=60, n=7, center="dark"))

# Setting the default color palette
def sinplot(flip=1):
    x = np.linspace(0, 14, 100)
    for i in range(1, 7):
        plt.plot(x, np.sin(x + i * .5) * (7 - i) * flip)

sns.set_palette("husl")
sinplot()

# Temporarily change the palette
with sns.color_palette("PuBuGn_d"):
    sinplot()

##### END STYLE
    
##### BEGIN PLOTTING
    # https://seaborn.pydata.org/tutorial/distributions.html
    
import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib.pyplot as plt    
import seaborn as sns
sns.set(color_codes=True)

# Plotting a normal plot / univariate plot
np.random.seed(sum(map(ord, "distributions")))
x = np.random.normal(size=100)
sns.distplot(x)

# Histogram
sns.distplot(x, kde=False, rug=True)

# Changing default bins
sns.distplot(x, bins=20, kde=False, rug=True)

# Kernel Density - show shape of a distribution
sns.distplot(x, hist=False, rug=True)
sns.kdeplot(x, shade=True)

# Multiple Kernel Densities
sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend()

# Cuts the edges off
sns.kdeplot(x, shade=True, cut=0)
sns.rugplot(x)

# Plotting Parametric
x = np.random.gamma(6, size=200)
sns.distplot(x, kde=False, fit=stats.gamma)

# Scatterplots
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
sns.jointplot(x="x", y="y", data=df)

# Hexbin Plots
x, y = np.random.multivariate_normal(mean, cov, 1000).T
with sns.axes_style("white"):
    sns.jointplot(x=x, y=y, kind="hex", color="k")

# Contour Plots
sns.jointplot(x="x", y="y", data=df, kind="kde")

# Outlines
f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df.x, df.y, ax=ax)
sns.rugplot(df.x, color="g", ax=ax)
sns.rugplot(df.y, vertical=True, ax=ax)

# Blurry One
f, ax = plt.subplots(figsize=(6, 6))
cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
sns.kdeplot(df.x, df.y, cmap=cmap, n_levels=60, shade=True)

# Grid with Markers
g = sns.jointplot(x="x", y="y", data=df, kind="kde", color="m")
g.plot_joint(plt.scatter, c="w", s=30, linewidth=1, marker="+")
g.ax_joint.collections[0].set_alpha(0)
g.set_axis_labels("$X$", "$Y$")


# Visualizing Multiple pairs of data sets
iris = sns.load_dataset("iris")
sns.pairplot(iris)

g = sns.PairGrid(iris)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)

# PLOTTING CATEGORICAL DATA
# https://seaborn.pydata.org/tutorial/categorical.html

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

sns.stripplot(x="day", y="total_bill", data=tips)

# These wont overlap so much due to 'jitter'
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)

# Swarm overlap
sns.swarmplot(x="day", y="total_bill", data=tips)

# Add a nested category
sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)

# Spread out
sns.swarmplot(x="size", y="total_bill", data=tips)

# Flip axes
sns.swarmplot(x="total_bill", y="day", hue="time", data=tips)

# Boxplots
sns.boxplot(x="day", y="total_bill", hue="time", data=tips)

# Barplots
sns.barplot(x="sex", y="survived", hue="class", data=titanic)

sns.countplot(x="deck", data=titanic, palette="Greens")
sns.countplot(x="deck", data=titanic, palette="Greens_r")
sns.countplot(x="deck", data=titanic, palette="Greens_d")

# Horizontal Bar
sns.countplot(y="deck", hue="class", data=titanic, palette="Greens_d")

# Line chart graph
sns.pointplot(x="sex", y="survived", hue="class", data=titanic)


sns.pointplot(x="class", y="survived", hue="sex", data=titanic,
              palette={"male": "g", "female": "m"},
              markers=["^", "o"], linestyles=["-", "--"])
              
sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips)

sns.factorplot(x="day", y="total_bill", hue="smoker", data=tips, kind="bar")

# Frequency plots with Swarm
sns.factorplot(x="day", y="total_bill", hue="smoker",
               col="time", data=tips, kind="swarm")
               
##### Visualizing Linear Relationships - Regression
# https://seaborn.pydata.org/tutorial/regression.html

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

# Regression lines
np.random.seed(sum(map(ord, "regression")))

tips = sns.load_dataset("tips")

sns.regplot(x="total_bill", y="tip", data=tips) # regplot
sns.lmplot(x="total_bill", y="tip", data=tips) # lmplot

sns.lmplot(x="size", y="tip", data=tips, x_jitter=.2)

# With Mean bars
sns.lmplot(x="size", y="tip", data=tips, x_estimator=np.mean)

# Estimating 2nd order polynomials
anscombe = sns.load_dataset("anscombe")
sns.lmplot(x="x", y="y", data=anscombe.query("dataset == 'II'"),
           order=2, ci=None, scatter_kws={"s": 80})
           
# Linear Separation
tips["big_tip"] = (tips.tip / tips.total_bill) > .15
sns.lmplot(x="total_bill", y="big_tip", data=tips,
           y_jitter=.03)
           
# Logistic Separation
sns.lmplot(x="total_bill", y="big_tip", data=tips,
           logistic=True, y_jitter=.03)

# Lowess
sns.lmplot(x="total_bill", y="tip", data=tips,
           lowess=True)

# Multiple datasets Regression
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)

# Special Markers
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips,
           markers=["o", "x"], palette="Set1")

# Multiple Plots
sns.lmplot(x="total_bill", y="tip", hue="smoker",
           col="time", row="sex", data=tips)

# Controlling size and shape of the plot (Regplot)
f, ax = plt.subplots(figsize=(5, 6))
sns.regplot(x="total_bill", y="tip", data=tips, ax=ax)

# Controlling size and shape of the plot (lmplot)
sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           col_wrap=2, size=3)

sns.lmplot(x="total_bill", y="tip", col="day", data=tips,
           aspect=.5)

# Regression with normal distributions
sns.jointplot(x="total_bill", y="tip", data=tips, kind="reg")

sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             size=5, aspect=.8, kind="reg")

# Multiple plot
sns.pairplot(tips, x_vars=["total_bill", "size"], y_vars=["tip"],
             hue="smoker", size=5, aspect=.8, kind="reg")

##### SMALL MULTIPLES
# https://seaborn.pydata.org/tutorial/axis_grids.html

import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
import matplotlib as mpl
import matplotlib.pyplot as plt

sns.set(style="ticks")
np.random.seed(sum(map(ord, "axis_grids")))

g = sns.FacetGrid(tips, col="time")
g.map(plt.hist, "tip")

g = sns.FacetGrid(tips, col="sex", hue="smoker")
g.map(plt.scatter, "total_bill", "tip", alpha=.7)
g.add_legend()

# 2x2
g = sns.FacetGrid(tips, row="smoker", col="time", margin_titles=True)
g.map(sns.regplot, "size", "total_bill", color=".3", fit_reg=False, x_jitter=.1)

# multiple
ordered_days = tips.day.value_counts().index
g = sns.FacetGrid(tips, row="day", row_order=ordered_days,
                  size=1.7, aspect=4,)
g.map(sns.distplot, "total_bill", hist=False, rug=True)

pal = dict(Lunch="seagreen", Dinner="gray")
g = sns.FacetGrid(tips, hue="time", palette=pal, size=5)
g.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, linewidth=.5, edgecolor="white")
g.add_legend()

# Add markers
g = sns.FacetGrid(tips, hue="sex", palette="Set1", size=5, hue_kws={"marker": ["^", "v"]})
g.map(plt.scatter, "total_bill", "tip", s=100, linewidth=.5, edgecolor="white")
g.add_legend()

# 12 chart 
attend = sns.load_dataset("attention").query("subject <= 12")
g = sns.FacetGrid(attend, col="subject", col_wrap=4, size=2, ylim=(0, 10))
g.map(sns.pointplot, "solutions", "score", color=".3", ci=None)

with sns.axes_style("white"):
    g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, size=2.5)
g.map(plt.scatter, "total_bill", "tip", color="#334488", edgecolor="white", lw=.5)
g.set_axis_labels("Total bill (US Dollars)", "Tip")
g.set(xticks=[10, 30, 50], yticks=[2, 6, 10])
g.fig.subplots_adjust(wspace=.02, hspace=.02)

g = sns.FacetGrid(tips, col="smoker", margin_titles=True, size=4)
g.map(plt.scatter, "total_bill", "tip", color="#338844", edgecolor="white", s=50, lw=1)
for ax in g.axes.flat:
    ax.plot((0, 50), (0, .2 * 50), c=".2", ls="--")
g.set(xlim=(0, 60), ylim=(0, 14))

# Hexbin
def hexbin(x, y, color, **kwargs):
    cmap = sns.light_palette(color, as_cmap=True)
    plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

with sns.axes_style("dark"):
    g = sns.FacetGrid(tips, hue="time", col="time", size=4)
g.map(hexbin, "total_bill", "tip", extent=[0, 50, 0, 10])

# Multicolor multiple
g = sns.PairGrid(tips, hue="size", palette="GnBu_d")
g.map(plt.scatter, s=50, edgecolor="white")
g.add_legend()

# Multi Multiple graph
g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", size=2.5)





##### Additional examples from the gallery
# Heatmap
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# Load the example flights dataset and conver to long-form
flights_long = sns.load_dataset("flights")
flights = flights_long.pivot("month", "year", "passengers")

# Draw a heatmap with the numeric values in each cell
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(flights, annot=True, fmt="d", linewidths=.5, ax=ax)





# 2 contours
sns.set(style="darkgrid")
iris = sns.load_dataset("iris")

# Subset the iris dataset by species
setosa = iris.query("species == 'setosa'")
virginica = iris.query("species == 'virginica'")

# Set up the figure
f, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect("equal")

# Draw the two density plots
ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
                 cmap="Reds", shade=True, shade_lowest=False)
ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
                 cmap="Blues", shade=True, shade_lowest=False)

# Add labels to the plot
red = sns.color_palette("Reds")[-2]
blue = sns.color_palette("Blues")[-2]
ax.text(2.5, 8.2, "virginica", size=16, color=blue)
ax.text(3.8, 4.5, "setosa", size=16, color=red)







# Multiple Distributions
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

# Create the data
rs = np.random.RandomState(1979)
x = rs.randn(500)
g = np.tile(list("ABCDEFGHIJ"), 50)
df = pd.DataFrame(dict(x=x, g=g))
m = df.g.map(ord)
df["x"] += m

# Initialize the FacetGrid object
pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
g = sns.FacetGrid(df, row="g", hue="g", aspect=15, size=.5, palette=pal)

# Draw the densities in a few steps
g.map(sns.kdeplot, "x", clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=2, bw=.2)
g.map(plt.axhline, y=0, lw=2, clip_on=False)

# Define and use a simple function to label the plot in axes coordinates
def label(x, color, label):
    ax = plt.gca()
    ax.text(0, .2, label, fontweight="bold", color=color, 
            ha="left", va="center", transform=ax.transAxes)

g.map(label, "x")

# Set the subplots to overlap
g.fig.subplots_adjust(hspace=-.25)

# Remove axes details that don't play will with overlap
g.set_titles("")
g.set(yticks=[])
g.despine(bottom=True, left=True)








# Vertical Bars
sns.set(style="whitegrid")

networks = sns.load_dataset("brain_networks", index_col=0, header=[0, 1, 2])
networks = networks.T.groupby(level="network").mean().T
order = networks.std().sort_values().index

sns.lvplot(data=networks, order=order, scale="linear", palette="mako")







# Cool Hexbin
import numpy as np
from scipy.stats import kendalltau
import seaborn as sns
sns.set(style="ticks")

rs = np.random.RandomState(11)
x = rs.gamma(2, size=1000)
y = -.5 * x + rs.normal(size=1000)

sns.jointplot(x, y, kind="hex", stat_func=kendalltau, color="#4CB391")
              
              





# Lines with shadows              
sns.set(style="darkgrid")

# Load the long-form example gammas dataset
gammas = sns.load_dataset("gammas")

# Plot the response with standard error
sns.tsplot(data=gammas, time="timepoint", unit="subject",
           condition="ROI", value="BOLD signal")





# Circular PLots

sns.set()

# Generate an example radial datast
r = np.linspace(0, 10, num=100)
df = pd.DataFrame({'r': r, 'slow': r, 'medium': 2 * r, 'fast': 4 * r})

# Convert the dataframe to long-form or "tidy" format
df = pd.melt(df, id_vars=['r'], var_name='speed', value_name='theta')

# Set up a grid of axes with a polar projection
g = sns.FacetGrid(df, col="speed", hue="speed",
                  subplot_kws=dict(projection='polar'), size=4.5,
                  sharex=False, sharey=False, despine=False)

# Draw a scatterplot onto each axes in the grid
g.map(plt.scatter, "theta", "r")






# Multiple Contours
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="dark")
rs = np.random.RandomState(50)

# Set up the matplotlib figure
f, axes = plt.subplots(3, 3, figsize=(9, 9), sharex=True, sharey=True)

# Rotate the starting point around the cubehelix hue circle
for ax, s in zip(axes.flat, np.linspace(0, 3, 10)):

    # Create a cubehelix colormap to use with kdeplot
    cmap = sns.cubehelix_palette(start=s, light=1, as_cmap=True)

    # Generate and plot a random bivariate dataset
    x, y = rs.randn(2, 50)
    sns.kdeplot(x, y, cmap=cmap, shade=True, cut=5, ax=ax)
    ax.set(xlim=(-3, 3), ylim=(-3, 3))

f.tight_layout()





# Diagonal correlation matrix
from string import ascii_letters
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="white")

# Generate a large random dataset
rs = np.random.RandomState(33)
d = pd.DataFrame(data=rs.normal(size=(100, 26)),
                 columns=list(ascii_letters[26:]))

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})






# Multiple barcharts
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style="white", context="talk")
rs = np.random.RandomState(7)


# Set up the matplotlib figure
f, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 6), sharex=True)

# Generate some sequential data
x = np.array(list("ABCDEFGHI"))
y1 = np.arange(1, 10)
sns.barplot(x, y1, palette="BuGn_d", ax=ax1)
ax1.set_ylabel("Sequential")

# Center the data to make it diverging
y2 = y1 - 5
sns.barplot(x, y2, palette="RdBu_r", ax=ax2)
ax2.set_ylabel("Diverging")

# Randomly reorder the data to make it qualitative
y3 = rs.choice(y1, 9, replace=False)
sns.barplot(x, y3, palette="Set3", ax=ax3)
ax3.set_ylabel("Qualitative")

# Finalize the plot
sns.despine(bottom=True)
plt.setp(f.axes, yticks=[])
plt.tight_layout(h_pad=3)




# Cmap colors
Colormap BuGn_d is not recognized. Possible values are: Accent, Accent_r, Blues, Blues_r, BrBG, BrBG_r, BuGn, BuGn_r, BuPu, BuPu_r, CMRmap, CMRmap_r, Dark2, Dark2_r, GnBu, GnBu_r, Greens, Greens_r, Greys, Greys_r, OrRd, OrRd_r, Oranges, Oranges_r, PRGn, PRGn_r, Paired, Paired_r, Pastel1, Pastel1_r, Pastel2, Pastel2_r, PiYG, PiYG_r, PuBu, PuBuGn, PuBuGn_r, PuBu_r, PuOr, PuOr_r, PuRd, PuRd_r, Purples, Purples_r, RdBu, RdBu_r, RdGy, RdGy_r, RdPu, RdPu_r, RdYlBu, RdYlBu_r, RdYlGn, RdYlGn_r, Reds, Reds_r, Set1, Set1_r, Set2, Set2_r, Set3, Set3_r, Spectral, Spectral_r, Vega10, Vega10_r, Vega20, Vega20_r, Vega20b, Vega20b_r, Vega20c, Vega20c_r, Wistia, Wistia_r, YlGn, YlGnBu, YlGnBu_r, YlGn_r, YlOrBr, YlOrBr_r, YlOrRd, YlOrRd_r, afmhot, afmhot_r, autumn, autumn_r, binary, binary_r, bone, bone_r, brg, brg_r, bwr, bwr_r, cool, cool_r, coolwarm, coolwarm_r, copper, copper_r, cubehelix, cubehelix_r, flag, flag_r, gist_earth, gist_earth_r, gist_gray, gist_gray_r, gist_heat, gist_heat_r, gist_ncar, gist_ncar_r, gist_rainbow, gist_rainbow_r, gist_stern, gist_stern_r, gist_yarg, gist_yarg_r, gnuplot, gnuplot2, gnuplot2_r, gnuplot_r, gray, gray_r, hot, hot_r, hsv, hsv_r, icefire, icefire_r, inferno, inferno_r, jet, jet_r, magma, magma_r, mako, mako_r, nipy_spectral, nipy_spectral_r, ocean, ocean_r, pink, pink_r, plasma, plasma_r, prism, prism_r, rainbow, rainbow_r, rocket, rocket_r, seismic, seismic_r, spectral, spectral_r, spring, spring_r, summer, summer_r, tab10, tab10_r, tab20, tab20_r, tab20b, tab20b_r, tab20c, tab20c_r, terrain, terrain_r, viridis, viridis_r, vlag, vlag_r, winter, winter_r

# Change font size
with sns.plotting_context("notebook",font_scale=0.6):
    sns.barplot(x="grandtotal", y="state" ,data=salesByState, palette="Blues_r", orient="h", ci=None, label='small')
