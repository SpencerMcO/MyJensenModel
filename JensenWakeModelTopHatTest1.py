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

# Initialize variables.
alpha = 0.1 # See p. 6

# Prompt the user to enter the necessary variables. Might be changed to reading from a file.
rotor_radius = float(raw_input('Enter the radius of the wind turbine rotor (meters): '))

# Initialize the x_distance array. Units should be in meters. Length of three because there are three plots we want to create.
x_distance = np.zeros(3)

# Create list of ratios of x/r0 that we want to achieve. These come from Jensen's paper, p. 7.
ratios = [16.0, 10.0, 6.0]

# Compute x for the given rotor radius r0. Calculate values of x in such a way that we get the ratios of x/r0 = 16, 10, and 6. This way, we can compare my results with the data in Jensen's paper.
for i in range(0,len(x_distance),1):
    x_distance[i] = rotor_radius * ratios[i]

print "The x-distances used in this program will be %f meters, %f meters, and %f meters." % (x_distance[0], x_distance[1], x_distance[2])

# Create a test vector x_distance so I can calculate reduced velocity v for any x. Units are in meters.
# x_distance = np.linspace(0.0,1000.0,1000)

# Intialize velocity percentage v.
# velocity_percentage = np.zeros(1000)

# Test how things are working so far by printing out some numbers.
#print 'Length of x_distance is %d' % len(x_distance)
#print 'Length of velocity_percentage is %d' % len(velocity_percentage)

# Calculate v for each x in the requested range. v is the percentage of the ambient wind velocity at a distance x behind the wind turbine.
#for i in np.arange(0,x_distance.size):
#    velocity_percentage[i] = 1.0 - ((2.0/3.0)*((rotor_radius/(rotor_radius + (alpha * x_distance[i]))) ** 2))

# Plot the results of velocity_percetange (v/u) vs. x.
# plt.figure(1)
# plt.plot(x_distance, velocity_percentage)
# plt.title('Wind Velocity vs. Distance')
# plt.ylabel('v/u')
# plt.xlabel('Distance from Turbine (m)')
# plt.grid(True)
# plt.show()

# Need user to tell us what x we're at in order to plot v/u vs. theta.
# x_actual = int(raw_input('Please enter the x-distance from the turbine to the closest meter: '))

# Find the index where this actual x-distance is stored in x_distance.
# x_actual_index = x_distance.index(x_actual)
# print "Index of the given x-distance in our x_distance vector is %d" % x_actual_index

# Using similar triangles and trigonometry, I find that theta = arctan(alpha). I'll calculate this value below for theta_wake so I can have an exact value, but it looks like it should be around 5.711 degrees. Units are in degrees.
theta_wake = np.arctan(alpha)*180.0/np.pi
print "The angle theta defining the wake boundaries is %f" % theta_wake
theta_max = 30.0
d_theta = 0.01  # Step size for theta_vector
theta_vector = np.linspace(-theta_max, theta_max, (1.0/d_theta))
print "Length of theta_vector is %d" % len(theta_vector)

# Initialize v_over_u array.
v_over_u = np.zeros(len(theta_vector))

# For top-hat Jensen model, assume no velocity deficit at all except for in the wake. At all angles within the wake, and at the x-distance in the wake, the v/u value should be constant. This constant value is determined by the velocity_percentage function above.
for i in range(0,len(theta_vector),1):
    if((theta_vector[i] >= -theta_wake) and (theta_vector[i] <= theta_wake)):
        v_over_u[i] = velocity_percentage[x_actual_index]
    else:
        v_over_u[i] = 1.0

# Now plot the results I just got.
plt.figure(2)
plt.plot(theta_vector, v_over_u, label="My Jensen Model")
plt.axis([-35, 35, 0.8, 1.1])
plt.title('Wind Velocity vs. Angle')
plt.ylabel(r'$v/u$')
plt.xlabel(r'$\Delta\theta$ (deg)')

points = np.loadtxt('JensenTopHatGraph16.txt', delimiter=',')
print points

JensenTopHatGraph16_X_coordinates = points[:,0]
JensenTopHatGraph16_Y_coordinates = points[:,1]

# Save the coordinates from WebPlotDigitizer.
#JensenTopGraphTopHat_XCoordinates = [-30.00060085321157, -6.644234813435084, -6.578541528971144, 6.396683290272179, 6.457970317851348, 18.29197460393759]
#JensenTopGraphTopHat_YCoordinates = [1.0007751006429129, 0.9977167577960704, 0.8939213683430471, 0.8930661539385928, 0.9962326503635162, 0.9985058783472531]

# Plot the results from web-plot digitizer for the x/ro = 16 graph (top graph on page 7) on top of my graph to determine how well my model matches Jensen's model. This is for the top-hat experiment. Only 4 points were taken that defined points where the line changed direction - in all other areas, the line was straight.
plt.figure(2)
plt.plot(JensenTopHatGraph16_X_coordinates, JensenTopHatGraph16_Y_coordinates, label="Jensen Model")
plt.legend()
plt.show()
