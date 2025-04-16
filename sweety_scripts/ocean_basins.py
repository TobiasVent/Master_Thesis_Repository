import pandas as pd

def get_zoned_df(appended_data):
    '''
    returns multiple dataframes corresponding to 4 different basins
    '''
    
    zone_ARCTIC = appended_data.loc[appended_data['nav_lat'] > 70.0]
    zone_ARCTIC['zone'] = 'ARCTIC'
        
    zone_NORTH_ATLANTIC= appended_data.loc[(appended_data['nav_lon'] >= -75.0) & (appended_data['nav_lon'] <= 0.0)]
    zone_NORTH_ATLANTIC = zone_NORTH_ATLANTIC.loc[(zone_NORTH_ATLANTIC['nav_lat'] >= 10) & (zone_NORTH_ATLANTIC['nav_lat'] <= 70)]
    zone_NORTH_ATLANTIC['zone'] = 'NORTH_ATLANTIC'
    
    zone_EQ= appended_data.loc[(appended_data['nav_lat'] >= -10.0) & (appended_data['nav_lat'] <= 10.0)]
    zone_EQ_PACIFIC_1 = zone_EQ.loc[(zone_EQ['nav_lon'] >= 105.0) & (zone_EQ['nav_lon'] <= 180.0)]
    zone_EQ_PACIFIC_2 = zone_EQ.loc[(zone_EQ['nav_lon'] >= -180.0) & (zone_EQ['nav_lon'] <= -80.0)]
    zone_EQ_PACIFIC = pd.concat([zone_EQ_PACIFIC_1, zone_EQ_PACIFIC_2])
    zone_EQ_PACIFIC['zone'] = 'EQ_PACIFIC'
    
    zone_SOUTHERN_OCEAN = appended_data.loc[appended_data['nav_lat'] <= -45]
    zone_SOUTHERN_OCEAN['zone'] = 'SOUTHERN_OCEAN'
    
    return zone_ARCTIC, zone_NORTH_ATLANTIC, zone_EQ_PACIFIC, zone_SOUTHERN_OCEAN

def assign_basins(nav_lat, nav_lon):
    '''
    Assigns basins to individual latitude and longitude values. 
    !!! Better use it as a lambda function.
    '''
    if nav_lat > 70.0:
        return 'ARCTIC'
    elif -75.0 <= nav_lon <= 0.0 and 10 <= nav_lat <= 70: 
        return 'NORTH_ATLANTIC'
    elif -10.0 <= nav_lat <= 10.0:
        if 105.0 <= nav_lon <= 180.0 or -180.0 <= nav_lon <= -80.0:
            return 'EQ_PACIFIC'
    elif nav_lat <= -45:
        return 'SOUTHERN_OCEAN'
    else:
        return 'OTHER'

data_df = pd.read_pickle("filepath/filename.pkl")
data_df['basin_name'] = data_df.apply(lambda row: assign_basins(row['nav_lat'], row['nav_lon']), axis=1)

