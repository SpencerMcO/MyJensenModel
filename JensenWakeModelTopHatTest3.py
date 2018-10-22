"""
Filename: JensenWakeModelTopHatTest3.py
Author: Spencer McOmber
Created: September 10, 2018
Description: This file is a homework assigment for my ME EN 497R with Dr. Ning and his graduate student, Jared Thomas. I'm going to use Jensen's article where he detailed the Jensen Model to try to recreate his model. This is in an attempt to help me to learn more about the lab and how these wake models work.
THIS FIRST FILE IS FOR THE TOP-HAT JENSEN MODEL.
WHAT ARE THE INPUTS AND OUTPUTS? DO I NEED TO FIND SOME DATA TO FEED INTO THIS PYTHON FILE?
"""

# Import modules.
# from ../../Programs/Jensen3D/src/jensen3d/JensenOpenMDAO.py import
import numpy as np
import matplotlib.pyplot as plt

# Is the top-hat formula at the top of p. 6?
# Looks like cosine formula is just the formula at the top of p. 6 with the cosine equation (eq. 3) on p. 8 applied to it.

# DEFINE FUNCTIONS.

# Create a function for the cosine modification for the cosine Jensen model.
def getJensenCosineAdjustment(angle):

    return (1.0 / 2.0) * (1.0 + np.cos(9 * angle))

def getPartialVelocityDeficit(alpha, rotor_radius, x_distance, JensenCosineAdjustment=1.0):

    return ((2.0/3.0)*((JensenCosineAdjustment*(rotor_radius/(rotor_radius + (alpha * x_distance)))) ** 2))

def getFullVelocityDeficit(jensen_model_type, alpha, rotor_radius, x_distance, theta):

    # Enter 'c' for jensen_model_type if you want to use the cosine Jensen model, enter 't' to use the top-hat Jensen model.
    if(jensen_model_type == 't'):
        return 1.0 - getPartialVelocityDeficit(alpha, rotor_radius, x_distance)
    else:

        # METHOD 1: JENSEN COSINE ADJUSTMENT OUTSIDE OF THE SQUARED TERMS. ASSUMING CORRECT.
        return 1.0 - (getJensenCosineAdjustment(np.radians(theta)) * getPartialVelocityDeficit(alpha, rotor_radius, x_distance))

        # METHOD 2: JENSEN COSINE ADJUSTMENT INSIDE THE SQUARED TERMS.
        # return 1.0 - getPartialVelocityDeficit(alpha, rotor_radius, x_distance, getJensenCosineAdjustment(
        #     np.radians(theta)))

# DONE DEFINING FUNCTIONS.

# START THE MAIN PROGRAM.
def main():

    # INITIALIZE VARIABLES.

    # Alpha constant obtained from paper. Might be closer to 0.07. See p. 6. Unitless.
    alpha = 0.1

    # Calculate the angle theta of the wake. See written work in OneNote on my laptop on Sept. 11, 2018 notes. Units in degrees.
    theta_wake = np.arctan(alpha)*180.0/np.pi

    print 'theta_wake is %f' % theta_wake

    # quit()

    # Create an angular limit for the cosine model. When theta's magnitude exceeds 20 degrees, the cosine model should return a v/u value of 1.0. Units in degrees.
    theta_cosine_limit = 20.0

    # Set the max theta value that will be plotted. Units in degrees.
    theta_max = 30.0

    # Create a theta_vector that will help plot v/u vs. theta. Units are in degrees.
    d_theta = 0.001 # Step size for theta_vector
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

    # Initialize v_over_u_tophat array for both the top-hat Jensen model and the cosine Jensen model. Note that
    # these are both 3-row by length-of-theta_vector-column arrays.
    v_over_u_tophat = np.array([np.zeros(len(theta_vector)), np.zeros(len(theta_vector)), np.zeros(len(theta_vector))])
    v_over_u_cosine = np.array([np.zeros(len(theta_vector)), np.zeros(len(theta_vector)), np.zeros(len(theta_vector))])

    # DONE INITIALIZING VARIABLES. BEGIN PRINTING INFO.

    # Inform the user what x-distances and reduced velocities will be used.
    print "The x-distances used in this program will be %f meters, %f meters, and %f meters." % (x_distance[0], x_distance[1], x_distance[2])

    # Using similar triangles and trigonometry, I find that theta = arctan(alpha). I'll calculate this value below for theta_wake so I can have an exact value, but it looks like it should be around 5.711 degrees. Units are in degrees.
    print "The angle theta defining the wake boundaries is %f degrees." % theta_wake
    print "Length of theta_vector is %d" % len(theta_vector)

    # DONE PRINTING INFO. BEGIN MAIN CALCULATIONS.

    # Create nested 'for' loops to help determine the v_over_u_tophat values for the v/u vs. theta plot. For top-hat Jensen model, assume no velocity deficit at all except for in the wake. At all angles within the wake, and at the x-distance in the wake, the v/u value should be constant. This constant value is determined by the velocity_percentage_tophat function above.
    for i in range(0, len(x_distance), 1):
        for j in range(0, len(theta_vector), 1):
            if((theta_vector[j] >= -theta_wake) and (theta_vector[j] <= theta_wake)):
                v_over_u_tophat[i, j] = getFullVelocityDeficit('t', alpha, rotor_radius, x_distance[i], theta_vector[j])
            else:
                v_over_u_tophat[i, j] = 1.0

    # Calculate v_over_u_cosine depending on the value of theta being used and on the x/ro ratio being used. The outer loop will focus on the ratio, and the inner loop will cycle through each value of theta in theta_vector.
    for i in range(0,len(x_distance),1):
        for j in range(0,len(theta_vector),1):
            if((theta_vector[j] >= -theta_cosine_limit) and (theta_vector[j] <= theta_cosine_limit)):
                v_over_u_cosine[i, j] = getFullVelocityDeficit('c', alpha, rotor_radius, x_distance[i], theta_vector[j])
            else:
                v_over_u_cosine[i, j] = 1.0

    # DONE WITH MAIN CALCULATIONS. BEGIN PLOTTING.

    # Import points from the txt file I got from WebPlotDigitizer.
    topHatPoints16 = np.loadtxt('../DataFiles/JensenTopHatGraph16.txt', delimiter=',')
    topHatPoints10 = np.loadtxt('../DataFiles/JensenTopHatGraph10.txt', delimiter=',')
    topHatPoints6 = np.loadtxt('../DataFiles/JensenTopHatGraph6.txt', delimiter=',')
    cosinePoints16 = np.loadtxt('../DataFiles/JensenCosineGraph16.txt', delimiter=',')
    cosinePoints10 = np.loadtxt('../DataFiles/JensenCosineGraph10.txt', delimiter=',')
    cosinePoints6 = np.loadtxt('../DataFiles/JensenCosineGraph6.txt', delimiter=',')
    scatterPoints16 = np.loadtxt('../DataFiles/JensenScatterGraph16.txt', delimiter=',')
    scatterPoints10 = np.loadtxt('../DataFiles/JensenScatterGraph10.txt', delimiter=',')
    scatterPoints6 = np.loadtxt('../DataFiles/JensenScatterGraph6.txt', delimiter=',')

    # Slice the top-hat points into x- and y-coordinates.
    JensenTopHatGraph16_X_coordinates = topHatPoints16[:,0]
    JensenTopHatGraph16_Y_coordinates = topHatPoints16[:,1]
    JensenTopHatGraph10_X_coordinates = topHatPoints10[:,0]
    JensenTopHatGraph10_Y_coordinates = topHatPoints10[:,1]
    JensenTopHatGraph6_X_coordinates = topHatPoints6[:,0]
    JensenTopHatGraph6_Y_coordinates = topHatPoints6[:,1]

    # Slice the cosine points into x- and y-coordinates.
    JensenCosineGraph16_X_coordinates = cosinePoints16[:,0]
    JensenCosineGraph16_Y_coordinates = cosinePoints16[:,1]
    JensenCosineGraph10_X_coordinates = cosinePoints10[:,0]
    JensenCosineGraph10_Y_coordinates = cosinePoints10[:,1]
    JensenCosineGraph6_X_coordinates = cosinePoints6[:,0]
    JensenCosineGraph6_Y_coordinates = cosinePoints6[:,1]

    # Slice the scatter points into x- and y-coordinates.
    JensenScatterGraph16_X_coordinates = scatterPoints16[:,0]
    JensenScatterGraph16_Y_coordinates = scatterPoints16[:,1]
    JensenScatterGraph10_X_coordinates = scatterPoints10[:,0]
    JensenScatterGraph10_Y_coordinates = scatterPoints10[:,1]
    JensenScatterGraph6_X_coordinates = scatterPoints6[:,0]
    JensenScatterGraph6_Y_coordinates = scatterPoints6[:,1]

    # Now plot the results I just got. Use a subplot command to create a single plot in figure 1 with 3 rows of plots, 1 column of plots, with the first plot going into the first index (1st row and 1st column).
    plt.subplots(3, 1, figsize = (9,9), sharex = True) # Not sure about this line.
    # plt.figure(num = 1, figsize=(1,1))
    plt.subplot(3,1,1)
    plt.plot(theta_vector, v_over_u_tophat[0], label="My Top-Hat Model")
    plt.axis([-35, 35, 0.8, 1.1])
    plt.title('Wind Velocity vs. Angle')
    plt.ylabel(r'$v/u$')

    # Also plot the cosine Jensen model on there.
    plt.plot(theta_vector, v_over_u_cosine[0], label = "My Cosine Model")

    # Plot the results from web-plot digitizer for the x/ro = 16 graph (top graph on page 7) on top of my graph to determine how well my model matches Jensen's model. This is for the top-hat experiment. Only 4 points were taken that defined points where the line changed direction - in all other areas, the line was straight.
    plt.plot(JensenTopHatGraph16_X_coordinates, JensenTopHatGraph16_Y_coordinates, label="Jensen Top-Hat Model")
    plt.plot(JensenCosineGraph16_X_coordinates, JensenCosineGraph16_Y_coordinates, label = "Jensen Cosine Model")
    plt.plot(JensenScatterGraph16_X_coordinates, JensenScatterGraph16_Y_coordinates, 'r*', label = "Datapoints")

    # Add a legend to the first subplot. Only one legend is needed since all the plots will follow the same coloring convention.
    plt.legend(ncol = 3)
    plt.annotate(r'$x/r_o=16$', xy = (20.0, 1.0), xytext = (22, 0.95), xycoords = 'data')

    # Create plot in 2nd index of subplot. This is for the x/ro = 10 plot in Jensen's paper.
    plt.subplot(3,1,2)
    plt.plot(theta_vector, v_over_u_tophat[1], label = "My Top-Hat Model: x/ro=10")

    # Also plot the cosine Jensen model on there.
    plt.plot(theta_vector, v_over_u_cosine[1], label = "My Cosine Model")

    # Plot the data from WebPlotDigitizer.
    plt.plot(JensenTopHatGraph10_X_coordinates, JensenTopHatGraph10_Y_coordinates, label="Jensen Model")
    plt.plot(JensenCosineGraph10_X_coordinates, JensenCosineGraph10_Y_coordinates, label = "Jensen Cosine Model")
    plt.plot(JensenScatterGraph10_X_coordinates, JensenScatterGraph10_Y_coordinates, 'r*', label = "Datapoints")

    # Add some labels and features to the plot.
    plt.ylabel(r'$v/u$')
    plt.annotate(r'$x/r_o=10$', xy = (20.0, 1.0), xytext = (22, 0.95), xycoords = 'data')

    # Create plot in 3rd index of subplot. This is for the x/ro = 6 plot in Jensen's paper.
    plt.subplot(3,1,3)
    plt.plot(theta_vector, v_over_u_tophat[2], label = "My Top-Hat Model: x/ro = 6")

    # Also plot the cosine Jensen model on there.
    plt.plot(theta_vector, v_over_u_cosine[2], label = "My Cosine Model")

    # Plot the data from WebPlotDigitizer.
    plt.plot(JensenTopHatGraph6_X_coordinates, JensenTopHatGraph6_Y_coordinates, label="Jensen Model")
    plt.plot(JensenCosineGraph6_X_coordinates, JensenCosineGraph6_Y_coordinates, label = "Jensen Cosine Model")
    plt.plot(JensenScatterGraph6_X_coordinates, JensenScatterGraph6_Y_coordinates, 'r*', label = "Datapoints")

    # Add some labels and features to the plot.
    plt.xlabel(r'$\Delta\theta$ (deg)')
    plt.ylabel(r'$v/u$')
    plt.annotate(r'$x/r_o=6$', xy = (20.0, 1.0), xytext = (22, 0.95), xycoords = 'data')

    # Tell the program to show the plot now.
    plt.show()

    print v_over_u_cosine.shape
    print v_over_u_cosine

    # Save the theta_vector as a 2D array so it can be a row vector. Note that now I have to use two indices to
    # access elements from this vector. The purpose of this is so I can store it with the v_over_u data into a data
    # file.
    theta_vector = np.array([theta_vector])
    print 'theta_vector shape is: ', theta_vector.shape

    # Stack the theta_vector underneath the v_over_u_cosine rows of data. Again, this allows us to just store data
    # from one 2D matrix to the data file.
    v_over_u_cosine = np.vstack([v_over_u_cosine, theta_vector])

    # Create file name strings that I can use to create different files with each iteration.
    filenameStrings = ['MyJensenCosineModelVelocity16.txt', 'MyJensenCosineModelVelocity10.txt',
                       'MyJensenCosineModelVelocity6.txt', 'MyJensenModelThetaVector.txt']

    # Outer loop cycles from 0 to 3, the rows in each column. Inner loop cycles from 0 to 999, the columns in each row.
    for i in range(len(v_over_u_cosine[:, 1])):

        # Create and write to a new text file to store the necessary data.
        velocityFile = open(filenameStrings[i], 'w+')

        for j in range(len(v_over_u_cosine[1, :])):
            velocityFile.write('%f\n' % v_over_u_cosine[i, j])

        # Close the file once I'm done writing to it.
        velocityFile.close()

# ADD THE SPECIAL FUNCTION THAT ALLOWS THIS PROGRAM TO WORK AS EITHER A STAND-ALONE PROGRAM OR AS A MODULE FOR ANOTHER PROGRAM.
if __name__ == '__main__':
    main()
