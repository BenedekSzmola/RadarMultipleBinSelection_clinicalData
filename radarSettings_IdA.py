"""
Radar related parameters, to be used by the analysis scripts.
"""

from scipy.constants import speed_of_light #m/s

# Config of Radar
radar_tx_num = 3 # number of active transmitters antennas
radar_rx_num = 4 # number of active receivers antennas
radar_loop_num = 20  #  in one frame
radar_adcsamp_num =  64 


radar_samp_freq = 20  # Hz
radar_samp_time = 0.05 #s
radar_freq_min = 60000000000 # Hz
radar_bandwidth = 2995.2 * 10**6   # Hz
radar_freq_mean = ((radar_freq_min+radar_bandwidth) + radar_freq_min)/2  # Hz - "The  'centre  operating  frequency'  equals  one  half  of  the  sum  of  the  highest  plus  the  lowest  specified  operating frequencies"
radar_wavelength_mean = speed_of_light / radar_freq_mean  # m ;  Check wether or not to take the mean values

radar_frame_time =  8058 * 10**-6 # s ; Time until the end of the frame after every chirp
radar_time_adc_on = 25.6 * 10**-6  # Time that one measurement in a chirp takes
radar_idle_time = (1/radar_samp_freq) - radar_time_adc_on  # If the time between between two chirps is necessary (e.g. for phase unwrapping) then everything before and after the adcOn time counts as idleTime

radar_dist_res = (speed_of_light/(2*radar_bandwidth))   # in m - distance resolution
radar_velo_res = radar_wavelength_mean / (2*radar_frame_time) # m/s  - speed resolution
# distance  = (Phase * speed_of_light) / (radar_min_freq * 4*scipy.constants.pi ) 

radarSettings = {
    "radar_tx_num": radar_tx_num,
    "radar_rx_num": radar_rx_num,
    "radar_loop_num": radar_loop_num,
    "radar_adcsamp_num": radar_adcsamp_num,
    "radar_samp_freq": radar_samp_freq,
    "radar_samp_time": radar_samp_time,
    "radar_freq_min": radar_freq_min,
    "radar_bandwidth": radar_bandwidth,
    "radar_freq_mean": radar_freq_mean,
    "radar_wavelength_mean": radar_wavelength_mean,
    "radar_frame_time": radar_frame_time,
    "radar_time_adc_on": radar_time_adc_on,
    "radar_idle_time": radar_idle_time,
    "radar_dist_res": radar_dist_res,
    "radar_velo_res": radar_velo_res
}
