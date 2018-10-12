"""
Filename: JensenWECTest1.py
Author: Spencer McOmber
Description: This is an attempt at applying Jared Thomas' WEC model into Jensen's 1983 Cosine Wake Model. I have
copied and pasted the code from "Jensen3DCosineComparison.py" and will adjust the Jensen model slightly so I can
attempt to implement WEC into the model.
"""

from plantenergy.jensen import jensen_wrapper, add_jensen_params_IndepVarComps
from plantenergy.OptimizationGroups import AEPGroup
import numpy as np
from openmdao.api import Group, Problem
import matplotlib.pyplot as plt

"""THIS IS THE RUN SCRIPT FOR JENSEN3D"""

# Create an array of the x/ro ratios used in Jensen's 1983 paper. Note that I'm only testing the ratio of x/ro = 16,
# because the previous results seemed to indicate that Jensen3D matched pretty well with all models at this ratio. No
# need to use the other ratios since I'm testing WEC and not the models' accuracy.
x_over_ro = 16.0
# x_over_ro = np.array([16.0, 10.0, 6.0])

# Instead of looping through different x/ro ratios, this program will cycle through different values for the
# relaxation factor used in WEC. Note that the stop value is 1.0 - 0.25, or the desired stop value minus the step
# size. This is to ensure that 1.0 is included in the array.
relaxationFactor = np.arange(3.0, 0.75, -0.25)

# define turbine locations in global reference frame
# turbineX = np.array([1164.7, 947.2,  1682.4, 1464.9, 1982.6, 2200.1])
# turbineY = np.array([1024.7, 1335.3, 1387.2, 1697.8, 2060.3, 1749.7])

# Define the start, stop, and step values for a thetaVector. Units in degrees.
thetaMax = 30.0
dTheta = 1.0
thetaVector = np.arange(-thetaMax, thetaMax, dTheta)

# initialize input variable arrays. Turbine coordinate arrays need to have the same sizes.
# For this run script, we only want two turbines for each run - one causing the wake, one receiving
# the wake.
nTurbines = 2

# Have the number of elements in the relaxationFactor vector as the number of rows in each turbine's position vector.
# This is so that we can create a new plot of v/u vs. crosswind position for each value of the relaxation factor we
# try. Each position vector also has thetaVector.size columns so that we can plot
turbineX = np.zeros((relaxationFactor.size, thetaVector.size))
turbineY = np.zeros((relaxationFactor.size, thetaVector.size))
turbineYNormalized = np.zeros((relaxationFactor.size, thetaVector.size))
rotorDiameter = np.zeros(nTurbines)
axialInduction = np.zeros(nTurbines)
Ct = np.zeros(nTurbines)
Cp = np.zeros(nTurbines)
generatorEfficiency = np.zeros(nTurbines)
yaw = np.zeros(nTurbines)

# define initial values
for turbI in range(0, nTurbines):
    rotorDiameter[turbI] = 126.4            # m
    axialInduction[turbI] = 1.0/3.0
    Ct[turbI] = 4.0*axialInduction[turbI]*(1.0-axialInduction[turbI])
    Cp[turbI] = 0.7737/0.944 * 4.0 * 1.0/3.0 * np.power((1 - 1.0/3.0), 2)
    generatorEfficiency[turbI] = 0.944
    yaw[turbI] = 0.     # deg.

# Calculate the x separation distance between turbines.
rotorRadius = rotorDiameter[0] / 2.0
turbineXInitialPosition = 0.0
turbineYInitialPosition = 0.0
# Calculate the x separation distance between turbines. Calculate the y-turbine positions based on the angle theta.
for i in range(relaxationFactor.size):
    for j in range(thetaVector.size):

        # Note that I only need one other x-coordinate for this program to work; however, the OpenMDAO functions need
        # the turbine position vectors to have the same size, which is why I'm stuffing the x position vector with
        # all the same numbers.
        turbineX[i, j] = x_over_ro * rotorRadius

        # Formula for getting y-coordinates from x position and theta.
        turbineY[i, j] = turbineYInitialPosition + turbineX[i, j] * np.arctan(np.radians(thetaVector[j]))

        # Calculate the normalized y-positions (crosswind positions).
        turbineYNormalized[i, j] = turbineY[i, j] / rotorDiameter[0]

# Plot the y-coordinates of the turbine to see how realistic they are.
# plt.subplot(3, 1, 1)
# plt.plot(thetaVector, turbineYNormalized[0, :])
# plt.ylabel('Crosswind Position (m)')
#
# plt.subplot(3, 1, 2)
# plt.plot(thetaVector, turbineYNormalized[1, :])
# plt.ylabel('Crosswind Position (m)')
#
# plt.subplot(3, 1, 3)
# plt.plot(thetaVector, turbineYNormalized[2, :])
# plt.ylabel('Crosswind Position (m)')
# plt.xlabel('ThetaVector (degrees)')
#
# plt.show()

# Define flow properties
nDirections = 1
wind_speed = 8.1                                # m/s
air_density = 1.1716                            # kg/m^3
# wind_direction = 270.-0.523599*180./np.pi       # deg (N = 0 deg., using direction FROM, as in met-mast data)
wind_direction = 270.0       # deg (N = 0 deg., using direction FROM, as in met-mast data)

wind_frequency = 1.                             # probability of wind in this direction at this speed

# set up problem

wake_model_options = {'variant': 'Cosine'}
prob = Problem(root=AEPGroup(nTurbines, nDirections, wake_model=jensen_wrapper, wake_model_options=wake_model_options,
                             params_IdepVar_func=add_jensen_params_IndepVarComps,
                             params_IndepVar_args={'use_angle': False}))

# initialize problem
prob.setup(check=True)

# assign values to turbine states
# prob['turbineX'] = turbineX
# prob['turbineY'] = turbineY
prob['yaw0'] = yaw

# assign values to constant inputs (not design variables)
prob['rotorDiameter'] = rotorDiameter
prob['axialInduction'] = axialInduction
prob['generatorEfficiency'] = generatorEfficiency
prob['windSpeeds'] = np.array([wind_speed])
prob['air_density'] = air_density
prob['windDirections'] = np.array([wind_direction])
prob['windFrequencies'] = np.array([wind_frequency])
prob['Ct_in'] = Ct
prob['Cp_in'] = Cp
# prob['model_params:spread_angle'] = 20.0
# prob['model_params:alpha'] = 0.1

# Need to set the relaxation factor for each iteration that we run. Relaxation factor needs to be a double scalar in
# the OpenMDAO code, so I need to pass it a new value for each iteration.
# prob['relaxationFactor'] = relaxationFactor

# run the problem
# prob.run()

# Create a text file that I can save data into.
velocityFile = open('Data Files/JensenWECTestWindspeed.txt', 'w+')

# Solve the wind velocities for each x/ro ratio (outer loop) while looping through each y-position of the turbine in
# the wake (inner loop).
for i in range(x_over_ro.size):

    # print 'x/ro ratio = %i' % x_over_ro[i]

    for j in range(thetaVector.size):

        # print 'Turbine X coordinates', turbineX[i, j]
        # print 'Turbine Y coordinates', turbineY[i, j]
        prob['turbineX'] = np.array([turbineXInitialPosition, turbineX[i, j]])
        prob['turbineY'] = np.array([turbineYInitialPosition, turbineY[i, j]])
        prob.run_once()
        # prob.run()

        # Print the second element of each array that's passed back. This second element represents the reduced
        # velocity at the turbine within the wake of the upstream turbine.
        # print prob['wtVelocity0'][1]

        # Save the wind turbine velocities and put them into a file I can load from. I actually need to save the v/u
        # ratio since that's what I'm interested in; thus, I'll divide the wind velocity I calculate by the given
        # wind_speed.
        velocityFile.write('%f\n' % (prob['wtVelocity0'][1] / wind_speed))

    # Write a blank line in between each group of x/ro. THIS WAS CAUSING ERRORS WHEN READING FILE.
    # velocityFile.write('\n')

# Close the file for writing.
velocityFile.close()

# Reopen the velocity file I just closed so I can read it.
velocityFile = open('Data Files/JensenWECTestWindspeed.txt', 'r')



# Show the graph.
# plt.show()

# # Import data from Spencer M.'s Jensen model to see how Jensen3D compares to it.
# MyJensenCosineVelocity16 = np.loadtxt('Data Files/MyJensenCosineModelVelocity16.txt')
# MyJensenCosineVelocity10 = np.loadtxt('Data Files/MyJensenCosineModelVelocity10.txt')
# MyJensenCosineVelocity6 = np.loadtxt('Data Files/MyJensenCosineModelVelocity6.txt')
# MyThetaVector = np.loadtxt('Data Files/MyJensenModelThetaVector.txt')
#
# # Import points from the txt file I got from WebPlotDigitizer. Results obtained by taking screenshot of the plots from
# # the document and using WebPlotDigitizer to obtain specific coordinates. Because the screenshot was of a paper that
# # was scanned into the computer, and the paper in the scan appears to be a little scrunched or askew, the results
# # may not be completely accurate.
# cosinePoints16 = np.loadtxt('Data Files/JensenCosineGraph16.txt', delimiter=',')
# cosinePoints10 = np.loadtxt('Data Files/JensenCosineGraph10.txt', delimiter=',')
# cosinePoints6 = np.loadtxt('Data Files/JensenCosineGraph6.txt', delimiter=',')
# scatterPoints16 = np.loadtxt('Data Files/JensenScatterGraph16.txt', delimiter=',')
# scatterPoints10 = np.loadtxt('Data Files/JensenScatterGraph10.txt', delimiter=',')
# scatterPoints6 = np.loadtxt('Data Files/JensenScatterGraph6.txt', delimiter=',')
#
# # Slice the cosine points into x- and y-coordinates.
# JensenCosineGraph16_X_coordinates = cosinePoints16[:, 0]
# JensenCosineGraph16_Y_coordinates = cosinePoints16[:, 1]
# JensenCosineGraph10_X_coordinates = cosinePoints10[:, 0]
# JensenCosineGraph10_Y_coordinates = cosinePoints10[:, 1]
# JensenCosineGraph6_X_coordinates = cosinePoints6[:, 0]
# JensenCosineGraph6_Y_coordinates = cosinePoints6[:, 1]
#
# # Slice the scatter points into x- and y-coordinates.
# JensenScatterGraph16_X_coordinates = scatterPoints16[:, 0]
# JensenScatterGraph16_Y_coordinates = scatterPoints16[:, 1]
# JensenScatterGraph10_X_coordinates = scatterPoints10[:, 0]
# JensenScatterGraph10_Y_coordinates = scatterPoints10[:, 1]
# JensenScatterGraph6_X_coordinates = scatterPoints6[:, 0]
# JensenScatterGraph6_Y_coordinates = scatterPoints6[:, 1]
#
# # Setup variables for subplots. This includes the variable I will use to store all the values from reading the file.
# nRows = 3
# nCols = 1
# v_over_u = np.array([np.zeros(thetaVector.size),
#                      np.zeros(thetaVector.size),
#                      np.zeros(thetaVector.size)])
# annotationStrings = [r'$x/r_o=16$', r'$x/r_o=10$', r'$x/r_o=6$']
# plt.subplots(nRows, nCols, figsize=(9, 9))  # Setting overall figure's size and axes.
#
# # Reopen the velocity file as a reading file.
# velocityFile = open('Data Files/Jensen3DVelocityOverWindspeed.txt', 'r')
#
# # Loop through the velocityFile to save the values from there into a 2D array so that I can plot the results.
# for i in range(x_over_ro.size):
#
#     for j in range(thetaVector.size):
#
#         # This command should read the next line after each iteration (e.g., starts at 1st line, goes to 2nd line at
#         # next iteration, etc.). NOTE THAT IN ORDER TO COMPARE WITH JENSEN'S 1983 DATA, we have to normalize this
#         # wake velocity we calculated by the wind_speed.
#         v_over_u[i, j] = float(velocityFile.readline())
#
#     # Plot the results as I'm going. Have to use i+1 because 'subplot' starts counting from 1, not zero.
#     plt.subplot(nRows, nCols, i+1)
#     plt.plot(thetaVector, v_over_u[i, :], label='Jensen3D Cosine Model')
#     plt.ylabel(r'$V/U$')
#     plt.annotate(annotationStrings[i], xy=(20.0, 1.0), xytext=(22, 0.96), xycoords='data')
#
#     # Plot the data obtained from Jensen's 1983 paper.
#     # Use 'if' statements to load the right data into the right plot.
#     if(i == 0):
#         plt.plot(MyThetaVector, MyJensenCosineVelocity16, label='My Jensen Model')
#         plt.plot(JensenCosineGraph16_X_coordinates, JensenCosineGraph16_Y_coordinates, label='Jensen\'s Cosine Model')
#         plt.plot(JensenScatterGraph16_X_coordinates, JensenScatterGraph16_Y_coordinates, 'r*', label='Datapoints')
#     elif(i == 1):
#         plt.plot(MyThetaVector, MyJensenCosineVelocity10, label='My Jensen Model')
#         plt.plot(JensenCosineGraph10_X_coordinates, JensenCosineGraph10_Y_coordinates, label='Jensen\'s Cosine Model')
#         plt.plot(JensenScatterGraph10_X_coordinates, JensenScatterGraph10_Y_coordinates, 'r*', label='Datapoints')
#     elif(i == 2):
#         plt.plot(MyThetaVector, MyJensenCosineVelocity6, label='My Jensen Model')
#         plt.plot(JensenCosineGraph6_X_coordinates, JensenCosineGraph6_Y_coordinates, label='Jensen\'s Cosine Model')
#         plt.plot(JensenScatterGraph6_X_coordinates, JensenScatterGraph6_Y_coordinates, 'r*', label='Datapoints')
#
#     # If the index i is zero, we're on the first plot. Add a legend and title to the plot. If we're on the final plot,
#     # add the x-label.
#     if(i == 0):
#         plt.title('Jensen3D Cosine Wake Model Comparison')
#         plt.legend()
#     elif(i == (x_over_ro.size - 1)):
#         plt.xlabel(r'$\Delta\theta$ (deg)')
