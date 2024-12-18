import math
import numpy as np
import datetime as dt
import pytz
import matplotlib.pyplot as plt

from sgp4.earth_gravity import wgs72
from sgp4.io import twoline2rv

def convert_seconds_to_hours(time_period):
    hours = int(time_period // 3600)
    time_period = time_period - hours * 3600
    minutes = int(time_period // 60.)
    seconds = time_period - minutes * 60
    return hours, minutes, seconds

def convert_jd_to_GMST(datetime_object):
    obj = datetime_object
    jd_day = convert_epoch_to_jd(obj.year, obj.month, obj.day, obj.hour, obj.minute, obj.second)
    return Calculate_GMST(jd_day)


def convert_epoch_to_jd(year, month, day, hour, minute, second):
    JD = 367 * year - int((7 * (year + int((month + 9) / 12))) / 4.0) + int((275.0 * month) / 9.0) + day + 1721013.5 + (
                (((second / 60.0) + minute) / 60 + hour) / 24.0)
    return JD

def convert_zero_to_twopi(rotation_angle):
    wrapping_angle = rotation_angle / (2 * math.pi)
    return wrapping_angle

def convert_ECI_to_ECEF(ECI, theta):
    ECEF = np.zeros([3], dtype=np.float64)
    ECEF = np.dot(rotate_ECI_ECEF(theta), ECI)
    return ECEF

def Calculate_GMST(JD):
    T_UT1 = (JD - 2451545.0) / 36525.0
    theta_GMST = 67310.54841 + (876600.0 * 60 * 60 + 8640184.812866) * T_UT1 + 0.093104 * T_UT1 ** 2 - 6.2e-6 * T_UT1 ** 3

    while theta_GMST > 86400.0:
        theta_GMST = theta_GMST - 86400

    theta_GMST = theta_GMST / 240.0
    theta_GMST = theta_GMST - 360  # in degrees not radian
    return theta_GMST

def rotate_ECI_ECEF(alpha_rad):
    T = np.array([[math.cos(alpha_rad), math.sin(alpha_rad), 0],
                  [-math.sin(alpha_rad), math.cos(alpha_rad), 0],
                  [0, 0, 1]], dtype=np.float64)
    return T

def convert_cartesian_into_lat_lon(R):
    r_delta = np.linalg.norm(R[0:2])
    sinA = R[1] / r_delta
    cosA = R[0] / r_delta

    lon = math.atan2(sinA, cosA)
    if lon < -math.pi:
        lon = lon + 2 * math.pi

    lat = math.asin(R[2] / np.linalg.norm(R))
    return lat, lon

# Read coastline data
def read_coastline_data(filename):
    lons, lats = [], []
    with open(filename, 'r') as file:
        for line in file:
            if line.strip():
                parts = line.split()
                lon, lat = float(parts[0]), float(parts[1])
                lons.append(lon)
                lats.append(lat)
    return np.array(lons), np.array(lats)

# TLE data for any LEO satellite
tle_line1 = '1 62008U 24213A   24349.83334491 -.00008862  00000+0 -50208-4 0  9999'
tle_line2 = '2 62008  53.1565 101.7473 0001160  60.5010 220.3147 15.76727001  5455'
line1 = (tle_line1)
line2 = (tle_line2)

satellite_obj = twoline2rv(line1, line2, wgs72)
print('Satellite number:', satellite_obj.satnum)
print('Epoch year:', satellite_obj.epochyr)
print('Epoch days:', satellite_obj.epochdays)
print('JD sat epoch:', satellite_obj.jdsatepoch)
print('Epoch:', satellite_obj.epoch)
print('Inclination:', math.degrees(satellite_obj.inclo))
print('RAAN:', math.degrees(satellite_obj.nodeo))
print('Eccentricity:', satellite_obj.ecco)
print('Argument of perigee:', math.degrees(satellite_obj.argpo))
print('Mean anomaly:', math.degrees(satellite_obj.mo))

delta_t = 1  # [s]
simulation_period = 95 * 60 * 2  # [s]
timevec = np.arange(0, simulation_period + delta_t, delta_t, dtype=np.float64)
x_state = np.zeros([6, len(timevec)], dtype=np.float64)
xecef = np.zeros([3, len(timevec)], dtype=np.float64)
lat = np.zeros([len(timevec)], dtype=np.float64)
lon = np.zeros([len(timevec)], dtype=np.float64)

index = 0
current_time = timevec[index]
hrs, mins, secs = convert_seconds_to_hours(current_time + (satellite_obj.epoch.hour * 60 * 60) + (satellite_obj.epoch.minute * 60) + satellite_obj.epoch.second)
dys = satellite_obj.epoch.day + int(math.ceil(hrs / 24))
if hrs >= 24:
    hrs = hrs - 24 * int(math.ceil(hrs / 24))

satpos, satvel = satellite_obj.propagate(satellite_obj.epoch.year, satellite_obj.epoch.month, dys, hrs, mins, secs + (1e-6) * satellite_obj.epoch.microsecond)
x_state[0:3, index] = np.asarray(satpos)
x_state[3:6, index] = np.asarray(satvel)

tle_epoch_test = dt.datetime(year=satellite_obj.epoch.year, month=satellite_obj.epoch.month, day=int(dys), hour=int(hrs), minute=int(mins), second=int(secs), microsecond=0, tzinfo=pytz.utc)
theta_GMST = math.radians(convert_jd_to_GMST(tle_epoch_test))
theta_GMST = convert_zero_to_twopi(theta_GMST)
xecef[:, index] = convert_ECI_to_ECEF(x_state[0:3, index], theta_GMST)
lat[index], lon[index] = convert_cartesian_into_lat_lon(xecef[:, index])

for index in range(1, len(timevec)):
    current_time = timevec[index]
    hrs, mins, secs = convert_seconds_to_hours(current_time + (satellite_obj.epoch.hour * 60 * 60) + (satellite_obj.epoch.minute * 60) + satellite_obj.epoch.second)
    dys = satellite_obj.epoch.day + int(math.ceil(hrs / 24))

    if hrs >= 24:
        hrs = hrs - 24 * int(math.ceil(hrs / 24))

    satpos, satvel = satellite_obj.propagate(satellite_obj.epoch.year, satellite_obj.epoch.month, dys, hrs, mins, secs + (1e-6) * satellite_obj.epoch.microsecond)
    x_state[0:3, index] = np.asarray(satpos)
    x_state[3:6, index] = np.asarray(satvel)

    tle_epoch_test = dt.datetime(year=satellite_obj.epoch.year, month=satellite_obj.epoch.month, day=int(dys), hour=int(hrs), minute=int(mins), second=int(secs), microsecond=0, tzinfo=pytz.utc)
    theta_GMST = math.radians(convert_jd_to_GMST(tle_epoch_test))
    theta_GMST = convert_zero_to_twopi(theta_GMST)
    xecef[:, index] = convert_ECI_to_ECEF(x_state[0:3, index], theta_GMST)
    lat[index], lon[index] = convert_cartesian_into_lat_lon(xecef[:, index])

tle_epoch_test = dt.datetime(year=satellite_obj.epoch.year, month=satellite_obj.epoch.month, day=satellite_obj.epoch.day,
                             hour=satellite_obj.epoch.hour, minute=satellite_obj.epoch.minute, second=satellite_obj.epoch.second,
                             microsecond=0, tzinfo=pytz.utc)

tle_epoch_test = tle_epoch_test.astimezone(pytz.timezone('Europe/Istanbul'))
groundtrack_title = tle_epoch_test.strftime("%B %d, %Y %H:%M:%S")

# Load coastline data
coastline_file = 'C:/Users/ekcgi/Downloads/Coastline.txt'
coastline_lon, coastline_lat = read_coastline_data(coastline_file)

# Plot on 2D world map
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_title("Starlink-11433 Ground Track on %s" % groundtrack_title, fontsize=16)

# Plot coastlines
ax.plot(coastline_lon, coastline_lat, 'k-', linewidth=0.5)

# Plot the satellite track
ax.plot(np.degrees(lon), np.degrees(lat), 'b.', markersize=1)

# Add text annotation at a specific location
ax.text(0.95, 0.05, "Giray Ekici", transform=ax.transAxes, fontsize=8,
        verticalalignment='bottom', horizontalalignment='right',
        bbox=dict(facecolor='white', alpha=0.5, boxstyle='round,pad=0.2'))

plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.xlim(-180, 180)
plt.ylim(-90, 90)

plt.show()
fig.savefig('iss_ground_track_map.pdf', format='pdf', bbox_inches='tight', pad_inches=0.01, dpi=1200)
