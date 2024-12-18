# README: Ground Tracking with manual TLE input

## Overview

This script allows users to manually input the two-line element (TLE) data for a satellite and processes it to create a satellite object using the `sgp4` library. The script can calculate orbital parameters or be extended for additional use cases such as generating ground tracks or satellite tracking.

## Features

- Accepts manual user input for TLE lines.
- Validates the format of the entered TLE lines.
- Parses the TLE data to create a satellite object using `sgp4`.
- Includes robust error handling for malformed or missing TLE input.

## References

- Vallado, D. A. (2013). Application: Converting IJK (ECEF) To Latitude and Longitude. In 
Fundamentals of Astrodynamics and Applications (4th ed., pp. 169–172). essay, Microcosm 
Press. Retrieved November 3, 2024, from https://archive.org/details/FundamentalsOfAstrodynamicsAndApplications/page/n8/mode/1up. 


