import PlotterClass
import numpy as np

plotter = PlotterClass.PlotterClass("outfiles_no_filter_2/diff_allcolumns.csv", "y", "outfiles_no_filter_2")

# x_span = np.array([-6, -4, -2, 0, 2, 4, 6])
# y_span = np.array([-6, -4, -2, 0, 2, 4])

# for x in x_span:
#     for y in y_span:
#         plotter.yx_pos_distribution(x,y)

plotter.YX_heatmap(is_single_plot=False, x_value=0, y_value=0)
