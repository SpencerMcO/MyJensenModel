"""
Filename: JensenWakeModelTopHatTest1.py
Author: Spencer McOmber
Created: September 10, 2018
Description: This file is a homework assigment for my ME EN 497R with Dr. Ning and his graduate student, jared Thomas. I'm going to use Jensen's article where he detailed the Jensen Model to try to recreate his model. This is in an attempt to help me to learn more about the lab and how these wake models work.
THIS FIRST FILE IS FOR THE TOP-HAT JENSEN MODEL.
WHAT ARE THE INPUTS AND OUTPUTS? DO I NEED TO FIND SOME DATA TO FEED INTO THIS PYTHON FILE?
"""

# Import modules.
import numpy as np
import matplotlib.pyplot as plt

# Is the top-hat formula at the top of p. 6?
# Looks like cosine formula is just the formula at the top of p. 6 with the cosine equation (eq. 3) on p. 8 applied to it.

# INITIALIZE VARIABLES.

# Alpha constant obtained from paper. Might be closer to 0.07. See p. 6. Unitless.
alpha = 0.1

# Calculate the angle theta of the wake. Units in degrees.
theta_wake = np.arctan(alpha)*180.0/np.pi

# Set the max theta value that will be plotted. Units in degrees.
theta_max = 30.0

# Create a theta_vector that will help plot v/u vs. theta. Units are in degrees.
d_theta = 0.01  # Step size for theta_vector
theta_vector = np.linspace(-theta_max, theta_max, (1.0/d_theta))

# Initialize the rotor_radius variable. An example of 20 meters is given in Jensen's paper, so I'll use that. Units are in meters.
rotor_radius = 20.0

# Create list of ratios of x/r0 that we want to achieve. These come from Jensen's paper, p. 7.
ratios = [16.0, 10.0, 6.0]

# Initialize the x_distance array based on the rotor_radius array to achieve the x/ro ratios in Jensen's paper. This represents the distance downstream from the turbine.
x_distance = np.zeros(len(ratios))

# Fill the x_distance array with values calculated from the rotor_radius and the ratios. Units are in meters.
for i in range(0,len(x_distance),1):
    x_distance[i] = ratios[i] * rotor_radius

# Initialize the velocity_percentage array. Should be the same length as the x_distance array or the ratios array.
velocity_percentage = np.zeros(len(x_distance))

# Initialize v_over_u array.
v_over_u = [np.zeros(len(theta_vector)), np.zeros(len(theta_vector)), np.zeros(len(theta_vector))]

# DONE INITIALIZING VARIABLES.

# Prompt the user to enter the necessary variables. Might be changed to reading from a file.
#rotor_radius = float(raw_input('Enter the radius of the wind turbine rotor (meters): '))

# Calculate the velocity percentage v/u for each x-distance above.
for i in range(0,len(x_distance),1):
    velocity_percentage[i] = 1.0 - ((2.0/3.0)*((rotor_radius/(rotor_radius + (alpha * x_distance[i]))) ** 2))

# Inform the user what x-distances and reduced velocities will be used.
print "The x-distances used in this program will be %f meters, %f meters, and %f meters." % (x_distance[0], x_distance[1], x_distance[2])

print "The velocity percentages at the x-distances are: %f, %f, and %f, respectively." % (velocity_percentage[0], velocity_percentage[1], velocity_percentage[2])

# Using similar triangles and trigonometry, I find that theta = arctan(alpha). I'll calculate this value below for theta_wake so I can have an exact value, but it looks like it should be around 5.711 degrees. Units are in degrees.
print "The angle theta defining the wake boundaries is %f" % theta_wake
print "Length of theta_vector is %d" % len(theta_vector)

# Create nested 'for' loops to help determine the v_over_u values for the v/u vs. theta plot. For top-hat Jensen model, assume no velocity deficit at all except for in the wake. At all angles within the wake, and at the x-distance in the wake, the v/u value should be constant. This constant value is determined by the velocity_percentage function above.
for i in range(0,len(x_distance),1):
    for j in range(0,len(theta_vector),1):
        if((theta_vector[j] >= -theta_wake) and (theta_vector[j] <= theta_wake)):
            v_over_u[i][j] = velocity_percentage[i]
        else:
            v_over_u[i][j] = 1.0

# Import points from the txt file I got from WebPlotDigitizer.
points16 = np.loadtxt('JensenTopHatGraph16.txt', delimiter=',')
points10 = np.loadtxt('JensenTopHatGraph10.txt', delimiter=',')
points6 = np.loadtxt('JensenTopHatGraph6.txt', delimiter=',')

# Slice these points into x- and y-coordinates.
JensenTopHatGraph16_X_coordinates = points16[:,0]
JensenTopHatGraph16_Y_coordinates = points16[:,1]
JensenTopHatGraph10_X_coordinates = points10[:,0]
JensenTopHatGraph10_Y_coordinates = points10[:,1]
JensenTopHatGraph6_X_coordinates = points6[:,0]
JensenTopHatGraph6_Y_coordinates = points6[:,1]

# Now plot the results I just got. Use a subplot command to create a single plot in figure 1 with 3 rows of plots, 1 column of plots, with the first plot going into the first index (1st row and 1st column).
plt.subplots(3,1, sharex = True) # Not sure about this line.
plt.subplot(3,1,1)
plt.plot(theta_vector, v_over_u[0], label="My Model: x/ro=16")
plt.axis([-35, 35, 0.8, 1.1])
plt.title('Wind Velocity vs. Angle')
plt.ylabel(r'$v/u$')

# Plot the results from web-plot digitizer for the x/ro = 16 graph (top graph on page 7) on top of my graph to determine how well my model matches Jensen's model. This is for the top-hat experiment. Only 4 points were taken that defined points where the line changed direction - in all other areas, the line was straight.
plt.plot(JensenTopHatGraph16_X_coordinates, JensenTopHatGraph16_Y_coordinates, label="Jensen Model")
plt.legend()

# Create plot in 2nd index of subplot. This is for the x/ro = 10 plot in Jensen's paper.
plt.subplot(3,1,2)
plt.plot(theta_vector, v_over_u[1], label = "My Model: x/ro=10")

# Plot the data from WebPlotDigitizer.
plt.plot(JensenTopHatGraph10_X_coordinates, JensenTopHatGraph10_Y_coordinates, label="Jensen Model")

# Add some labels and features to the plot.
plt.ylabel(r'$v/u$')
plt.legend()

# Create plot in 3rd index of subplot. This is for the x/ro = 6 plot in Jensen's paper.
plt.subplot(3,1,3)
plt.plot(theta_vector, v_over_u[2], label = "My Model: x/ro = 6")

# Plot the data from WebPlotDigitizer.
plt.plot(JensenTopHatGraph6_X_coordinates, JensenTopHatGraph6_Y_coordinates, label="Jensen Model")

# Add some labels and features to the plot.
plt.xlabel(r'$\Delta\theta$ (deg)')
plt.ylabel(r'$v/u$')
plt.legend()
plt.show()
