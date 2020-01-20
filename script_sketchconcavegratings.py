#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 18 22:29:19 2019

@author: samuel
"""

import numpy as np
import matplotlib.pyplot as plt
from general import wavelength_to_rgb


# load data from text table copied from richardson website, 2019-01-18
# http://gratinglab.com/Products/Product_Tables/T6.aspx
grating_data = {}
with open('dat_concavegratings.txt', 'r', encoding = "ISO-8859-1") as f:
    lines = [line.replace("Ê", "").rstrip('\n') for line in f]
    for i, line in enumerate(lines):
        if i == 0:
            grating_data_labels = line.split('\t')
        else:
            grating_row = line.split('\t')
            for j, field_value in enumerate(grating_row):
                if j == 0:
                    grating_name = field_value
                    grating_data[grating_name] = {}
                elif j > 0:
                    try:
                        field_name = grating_data_labels[j]
                        grating_data[grating_name][field_name] = float(field_value)
                    except ValueError:
                        #print(field_value)
                        try:
                            grating_data[grating_name][field_name] = float(field_value.split("x", 1)[0].strip())
                        except:
                            grating_data[grating_name][field_name] = field_value
    


# calculate the xy coords of the input slit and image location, assuming grating center at (0,0)
for catalog_name, grating in grating_data.items():
    grating['xy_input'] = (grating["r (mm)"] * np.cos(grating["alpha (deg)"] * np.pi / 180),
                           grating["r (mm)"] * np.sin(grating["alpha (deg)"] * np.pi / 180))
    grating['xy_output1'] = (grating["r'1 (mm)"] * np.cos(grating["beta1 (deg)"] * np.pi / 180),
                             grating["r'1 (mm)"] * np.sin(grating["beta1 (deg)"] * np.pi / 180))
    grating['xy_output2'] = (grating["r'2 (mm)"] * np.cos(grating["beta2 (deg)"] * np.pi / 180),
                             grating["r'2 (mm)"] * np.sin(grating["beta2 (deg)"] * np.pi / 180))
    grating['L_detector'] = np.sqrt( (grating['xy_output2'][1] - grating['xy_output1'][1]) ** 2 + 
                                     (grating['xy_output2'][0] - grating['xy_output1'][0]) ** 2 )





# make layout diagrams
for catalog_name, grating in grating_data.items():
    plt.figure()
    
    plt.plot(grating['xy_input'][0], grating['xy_input'][1], "ok")
    
    plt.plot(0, grating['ruled']/2, "ok")
    plt.plot(0, -grating['ruled']/2, "ok")
    plt.plot([0, 0], [grating['ruled']/2, -grating['ruled']/2], "-k")

    plt.plot(grating['xy_output1'][0], grating['xy_output1'][1], "ok")
    plt.plot(grating['xy_output2'][0], grating['xy_output2'][1], "ok")
    plt.plot([grating['xy_output1'][0], grating['xy_output2'][0]], 
             [grating['xy_output1'][1], grating['xy_output2'][1]], ":k")
    
    plt.plot([grating['xy_input'][0], 0, grating['xy_output1'][0]], 
             [grating['xy_input'][1], grating['ruled']/2, grating['xy_output1'][1]], "-", 
             color=wavelength_to_rgb(grating['m lambda1'], gamma=1.0, floor_wave=440, ceiling_wave=660))
    plt.plot([grating['xy_input'][0], 0, grating['xy_output1'][0]], 
             [grating['xy_input'][1], -grating['ruled']/2, grating['xy_output1'][1]], "-", 
             color=wavelength_to_rgb(grating['m lambda1'], gamma=1.0, floor_wave=440, ceiling_wave=660))
    plt.plot([grating['xy_input'][0], 0, grating['xy_output2'][0]], 
             [grating['xy_input'][1], grating['ruled']/2, grating['xy_output2'][1]], "-", 
             color=wavelength_to_rgb(grating['m lambda2'], gamma=1.0, floor_wave=440, ceiling_wave=660))
    plt.plot([grating['xy_input'][0], 0, grating['xy_output2'][0]], 
             [grating['xy_input'][1], -grating['ruled']/2, grating['xy_output2'][1]], "-", 
             color=wavelength_to_rgb(grating['m lambda2'], gamma=1.0, floor_wave=440, ceiling_wave=660))
    
    title_string = catalog_name + ",  " + str(grating['m lambda1']) + " nm - " + \
                   str(grating['m lambda2']) + " nm,  L = " + f"{grating['L_detector']:.2f} mm, " + \
                   f" f/# {grating['f/#']}"
    plt.title(title_string)
    plt.axis("equal")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")

    plt.savefig("grating diagrams/" + catalog_name + ".pdf", format="pdf")
    plt.close()

# 1,3,6,7,9,12,13,18









