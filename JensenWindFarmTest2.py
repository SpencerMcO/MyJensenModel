"""
Filename: JensenWindFarmTest1.py
Author: Spencer McOmber
Created: September 14, 2018
Description: The purpose for this Python file is to estimate the annual energy production (AEP) of a wind farm using my Jensen Wake Model that I created last week. I will use the Amalia wind farm windrose data to achieve this.
"""

# Import modules.
from JensenWakeModelTopHatTest3 import getFullVelocityDeficit
import numpy as np
import matplotlib.pyplot as plt

# Define a function to convert windrose directions to standard directions. Assuming that we're finding the standard direction in which the wind is GOING, not the direction from which the wind is COMING. Assuming that windroseDirection is in degrees, not radians.
def get_standard_direction(windroseDirection):

    # Equation I derived in my notes to convert directions.
    standardDirection = (-windroseDirection) + 270.0

    # Try to get an angle in the range of 0-359 degrees (i.e., don't want negative angles).
    if(standardDirection < 0.0):
        standardDirection += 360.0

    return standardDirection

# The main function that will run when this program is called.
def main():

    # INITIALIZE VARIABLES.

    # Set the AEPPower to be zero. This is because I will be summing to it in a loop later.
    AEPPower = 0

    # Initialize the rotorRadius variable. An example of 20 meters is given in Jensen's paper, so I'll use that. Units are in meters.
    rotorRadius = 20.0

    # Initialize a turbineSpacing variable. This variable stores the distance that should be between each turbine. Jensen's paper specified that this should at least be four times the rotor's diameter, so that's what I calculated. Units in meters.
    turbineSpacing = 4.0 * (rotorRadius * 2.0)

    # Initialize an integer variable storing the number of turbines being used.
    nTurbines = 16

    # Specify x and y coordinates for each of the 16 turbines. Store these coordinates in x and y arrays.
    xCoordinates = np.linspace(0, (15 * turbineSpacing), nTurbines)
    yCoordinates = np.linspace(0, (15 * turbineSpacing), nTurbines)

    # DONE INITIALIZING VARIABLES.

    # Read the windrose file.
    windroseInfo = np.loadtxt('windrose_amalia_directionally_averaged_speeds_36dir.txt', delimiter=' ')

    # Slice the imported information into separate columns. First column should be direction wind is coming FROM,
    # second column should be average speed of wind in the corresponding direction, third column should be
    # probability the wind will blow FROM the corresponding direction.
    windroseDirection = windroseInfo[:, 0]
    windroseAveSpeed = windroseInfo[:, 1]
    windroseProbability = windroseInfo[:, 2]

    # Obtain windrose direction. This will eventually be replaced by reading in the file "windrose_amalia_directionally_averaged_speeds_36dir.txt" (is this the right file?).
    # windroseDirection = float(raw_input('Enter a windrose direction (0-359 degrees): '))

    # Initialize the standardDirection array to store all the new standard directions the wind is GOING TO.
    standardDirection = np.zeros(len(windroseDirection))

    # Convert windrose direction (WD) to standard direction (SD).
    for i in range(0, len(standardDirection)):
        standardDirection[i] = get_standard_direction(windroseDirection[i])

# PSEUDO-CODE
"""
get turbines in wake function NOT SURE IF I'M ACTUALLY GOING TO USE THIS
    depends on positions of turbines and angle of wake expansion
    assuming 4x4 square wind farm
    for each wind direction in windrose
        rotate wind farm to match wind direction (where wind direction is coming in from the left)
        for each turbine in wind farm
            calculate lines of wake spreading out from the rotor
            for each turbine in wind farm
                if turbine's x and y coordinates are within the lines of the wake, then add another velocity term to that turbine's wind velocity deficit equation
    returns nothing? (void?) Or returns the updated velocity deficit equation for the given turbine?


check if turbine is in wake function
    idea 1: create lines with slopes calculated from alpha (wake decay constant?) and starting points determined by
    turbine location and rotor radius. Calculate y-coordinates of these lines from x-coordinate of current turbine.
    If y-coordinate of current turbine is between the two y-coordinates of the two lines, then the current turbine is within the wake of the upstream turbine. If these conditions are met, the function returns true.
    BETTER IDEA 2: find the delta_y (separation distance in y) between current turbine and wake-causing-turbine,
    then calculate the wake width for the wake-causing-turbine. If wake width > delta_y, then we're in the wake.
    Otherwise, current turbine is outside of the wake. This is assuming that the entire turbine can be modeled as a
    single point (i.e., the hub).

get velocity function()
    for each turbine in wind farm
        if ith turbine's wake covers position where current turbine is at
            calculate velocity deficit for current turbine from ith turbine's wake
    v_over_u = 1 - sqrt(sum((1 - ith_velocity_over_u)^2))
    returns wind velocity entering the current turbine

turbine power function
    get velocity of air at the turbine (based on the wakes)
    power = (1.0/2.0) * air_density * rotor_area * power_coefficient * velocity^3

rotate windfarm function
    rotation matrix applied to each turbine's x and y coordinates? LOOK ONLINE.
    return rotated coordinates of turbines?

convert to standard direction function
    standard_direction = fancy conversion equation
    do I need to convert this from direction wind is COMING FROM to direction wind is GOING TO? YES!!!

main function
    AEPPower initialized to zero
    x and y coordinates for each turbine are specified (such that the smallest distance between each turbine is at least 4 rotor diameters)
    read windrose file
    get number of directions in windrose
    for each direction in windrose
        convert direction from windrose direction to standard angle measurement direction
        rotate wind farm so that wind is coming in from left to windfarm on right (LOOP THROUGH TURBINES,
        transform each turbine individually)
        get probability of wind in that direction
        for each turbine in farm
            calculate power of turbine
        AEPPower += probability[ith direction] * power[ith direction]
    AEP = AEPPower * hours_in_year
    display or return AEP result
"""

# Add this if-statement to make this program work as a stand-alone program or as a module for another program.
if __name__ == '__main__':
    main()
