# data preprocessing for ML side
import numpy as np
import pandas as pd

# to do:
# - find top 20 most popular pitchers + numbers

# extract data from only top 20 pitchers
raw_data = pd.read_csv('pitches-2019-regular-04.csv')
top_20_pitchers = raw_data['pitcher'].value_counts()[:20].index.tolist()
df = raw_data[raw_data['pitcher'].isin(top_20_pitchers)]

# record pitcher names and ids for reference
df = df.assign(pitcher=lambda x: x['pitcher_first'] + ' ' + x['pitcher_last'])

# get rid of batter and pitcher names because they're useless
df = df.drop(['game_date', 'release_spin_rate', 'batter_first', 'pitch_type', 'batter_last', 'batter',
              'pitcher_first', 'pitcher_last', 'stand', 'p_throws'], axis=1)


# normalize the rest
normalized = ['release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
              'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'effective_speed', 'release_extension']
normalized_mean_stdev = [['variable', 'mean', 'stdev']]
for col in normalized:
    df = df.copy()
    index = df.columns.get_loc(col)
    mean = df.loc[:, col].mean()
    std = df.loc[:, col].std()
    df.loc[:, col] = (df[col] - mean)/std
    normalized_mean_stdev.append([col, mean, std])

# get dummies:
# pitchType
# pitcher

df = df.rename({
    "pitch_name": "Pitch Name",
    "pitcher": "Pitcher",
    "release_speed": "Release Speed",
    "release_pos_x": "Release Position (x)",
    "release_pos_y": "Release Position (y)",
    "release_pos_z": "Release Position (z)",
    "pfx_x": "PITCHf/x (x)",
    "pfx_z": "PITCHf/x (z)",
    "plate_x": "Plate Position (x)",
    "plate_z": "Plate Position (z)",
    "vx0": "Initial velocity (x)",
    "vy0": "Initial velocity (y)",
    "vz0": "Initial velocity (z)",
    "ax": "Acceleration (x)",
    "ay": "Acceleration (y)",
    "az": "Acceleration (z)",
    "effective_speed": "Effective Speed",
    "release_extension": "Release Extension"
})

np.savetxt("normalized_mean_stdev.csv",
           normalized_mean_stdev, delimiter=", ", fmt='% s')
df.to_csv('prepared_data.csv', index=False)
