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

# Test to check if the Jensen single wind turbine model is working fine.
# random_variable = getFullVelocityDeficit('c', 0.1, 25.0, 320.0, 0.0)
# print "random_variable is %f" % random_variable

# Define a function to convert windrose directions to standard directions. Assuming that we're still just defining what direction the wind is COMING FROM, not the direction it is GOING TO. Assuming that windroseDirection is in degrees, not radians.
def get_standard_direction(windroseDirection):

    # Equation I derived in my notes to convert directions.
    standardDirection = (-windroseDirection) + 90

    # Try to get an angle in the range of 0-359 degrees (i.e., don't want negative angles).
    if(standardDirection < 0):
        standardDirection += 360

    return standardDirection

# The main function that will run when this program is called.
def main():

    # Obtain windrose direction. This will eventually be replaced by reading in the file "windrose_amalia_directionally_averaged_speeds.txt" (is this the right file?).
    windroseDirection = float(raw_input('Enter a windrose direction (0-359 degrees): '))

    # Convert windrose direction (WD) to standard direction (SD).
    standardDirection = get_standard_direction(windroseDirection)

    # Display the result.
    print "The standard direction based on the windrose direction is %f degrees." % standardDirection

# PSEUDO-CODE
"""
get velocity function(needs array of )
    get turbines within the current turbine's wake
    calculate the wind velocity deficit for each of those turbines in the wake
    v_over_u = 1 - sqrt(sum((1 - ith_velocity_over_u)^2))
    returns wind velocity entering the current turbine

get turbines in wake function
    depends on positions of turbines and angle of wake expansion
    assuming 4x4 square wind farm
    for each wind direction in windrose
        rotate wind farm to match wind direction (where wind direction is coming in from the left)
        for each turbine in wind farm
            calculate lines of wake spreading out from the rotor
            for each turbine in wind farm
                if turbine's x and y coordinates are within the lines of the wake, then add another velocity term to that turbine's wind velocity deficit equation
    returns nothing? (void?) Or returns the updated velocity deficit equation for the given turbine?

convert direction function
    standard_direction = fancy conversion equation

turbine power function
    get velocity of air at the turbine (based on the wakes)
    power = (1.0/2.0) * air_density * rotor_area * power_coefficient * velocity^3

main function
    AEP_power initialized to zero
    x and y coordinates for each turbine are specified (such that the smallest distance between each turbine is at least 4 rotor diameters)
    read windrose file
    get number of directions in windrose
    for each direction in windrose
        convert direction from windrose direction to standard angle measurement direction
        get probability of wind in that direction
    for each direction in windrose
        for each turbine in farm
            calculate power of turbine
        AEP_power += frequency[i] * power[i]
    AEP = AEP_power * hours_in_year
    display or return AEP result
"""

# Add this if-statement to make this program work as a stand-alone program or as a module for another program.
if __name__ == '__main__':
    main()
