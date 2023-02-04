import numpy as np
import pandas as pd

# to do:
# - find top 20 most popular pitchers + numbers

# extract data from only top 20 pitchers
raw_data = pd.read_csv('pitches-2019-regular-04.csv')
top_20_pitchers = raw_data['pitcher'].value_counts()[:20].index.tolist()
df = raw_data[raw_data['pitcher'].isin(top_20_pitchers)]

# record pitcher names and ids for reference
nameList = []
for num_id in top_20_pitchers:
    firstName = df[df['pitcher'] == num_id]['pitcher_first'].values[0]
    lastName = df[df['pitcher'] == num_id]['pitcher_last'].values[0]
    name = f"{firstName} {lastName}"
    nameList.append([num_id, name])

# get dummies:
# p_throws
# pitchType
# stand
# pitcher

# do weird formatting with date (percentiles)
dateList = np.sort(df['game_date'].unique())
df = df.copy()
df['game_date'] = df['game_date'].map(lambda d: np.where(dateList == d)[0][0]/len(dateList))

# get rid of batter and pitcher names because they're useless
df = df.drop(['release_spin_rate', 'batter_first', 'pitch_name', 'batter_last', 'batter',
              'pitcher_first', 'pitcher_last'], axis=1)

# normalize the rest
normalized = ['release_speed', 'release_pos_x', 'release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z',
              'plate_x', 'plate_z', 'vx0', 'vy0', 'vz0', 'ax', 'ay', 'az', 'effective_speed', 'release_extension']
normalized_mean_stdev = []
for col in normalized:
    df = df.copy()
    index = df.columns.get_loc(col)
    mean = df.loc[:, col].mean()
    std = df.loc[:, col].std()
    df.loc[:, col] = (df[col] - mean)/std
    normalized_mean_stdev.append([col, mean, std])

#df = pd.get_dummies(
#    df, columns=['p_throws', 'pitch_type', 'stand', 'pitcher'], drop_first=True)

df.to_csv('prepared_data.csv', index=False)
