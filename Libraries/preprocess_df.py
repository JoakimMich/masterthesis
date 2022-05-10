import json
import pandas as pd
from Libraries import signality_utilities as sign_u
import numpy as np

def exception_numbers(number, team, game_id):
    if game_id == '13299200-2faa-11ec-aa86-5be90a5520ac':
        if number == 39 and team == 'away':
            number = 77
    return number

def aux_flatten(players, team, game_id):
    players_dict = {}
    new_players = []
    for p in players:
        number = exception_numbers(p['jersey_number'], team, game_id)
        if number != -1:
            players_dict[str(number)] = {'x': p['position'][0], 'y': p['position'][1]}
            new_players.append(p)
        
    return players_dict, new_players


def preprocess_signality(to_analyse, data_folder, phases=['1','2'], interpolate_players=True):
    df_list = []
    
    info_dir = data_folder+'/'+to_analyse+'.1-info_live.json'
    with open(info_dir) as json_data:
        game_info = json.load(json_data)
        
    
    for i in phases:
        tracks_file = data_folder+'/'+to_analyse+'.'+i+'-tracks.json' 
        print("Loading phase "+i)
        with open(tracks_file) as f:
            tracks_json = json.load(f)
        if 'error' in tracks_json:
            print(tracks_json['error']['message'])
            return pd.DataFrame(), game_info, False
        for track in tracks_json:
            players_dict, players_list = aux_flatten(track['home_team'], 'home', game_info['id'])
            track['home_team_aux'] = players_dict
            track['home_team'] = players_list
            
            players_dict, players_list = aux_flatten(track['away_team'], 'away', game_info['id'])
            track['away_team_aux'] = players_dict
            track['away_team'] = players_list

        df_list.append(pd.json_normalize(tracks_json))

        
    tracking_df = pd.concat(df_list) 

    
        
    # Find attack directions
    with open(data_folder+'/'+to_analyse+'.1-events.json') as json_data:
        signality_events = json.load(json_data)
        
    pitch_length = game_info['calibration']['pitch_size'][0]
    pitch_width = game_info['calibration']['pitch_size'][1]
    
    tr_copy = tracking_df.copy()
    
    def aux_normalize_x(signality_x, pitch_length):
        return ((signality_x+pitch_length/2)/pitch_length*106)-106/2

    def aux_normalize_y(signality_y, pitch_width):
        return (((signality_y+pitch_width/2)/pitch_width*68)-68/2)*(-1)

    x_columns = [c for c in tr_copy.columns if c[-2:].lower()=='.x']
    y_columns = [c for c in tr_copy.columns if c[-2:].lower()=='.y']
    

    tr_copy = tracking_df.copy()
    tr_copy[x_columns] = ((tr_copy[x_columns]+pitch_length/2)/pitch_length*106)-106/2
    tr_copy[y_columns] = (((tr_copy[y_columns]+pitch_width/2)/pitch_width*68)-68/2)*(-1)
    tr_copy['ball'] = [{'x': aux_normalize_x(x[0], pitch_length), 'y': aux_normalize_y(x[1], pitch_width)} if x != None else {'x': None, 'y': None} for x in tr_copy['ball.position']]
    tr_copy = tr_copy.reset_index(drop=True)
    
    if 'team_home_is_left' in signality_events[0]:
        game_info['home_attack_direction_1'] = 'right' if signality_events[0]['team_home_is_left'] == True else 'left'
    else:
        for i,r in tr_copy[tr_copy.phase==1].iterrows():
            home_players = []
            away_players = []
            for x_value in x_columns:
                if pd.isna(r[x_value]) == False:
                    if 'home' in x_value:
                        home_players.append(r[x_value])
                    else:
                        away_players.append(r[x_value])
            if len(home_players) >= 5 and len(away_players) >= 5:
                if np.mean(home_players) < np.mean(away_players):
                    game_info['home_attack_direction_1'] = 'right'
                else:
                    game_info['home_attack_direction_1'] = 'left'
                break
    
    if interpolate_players == True:
        for i in range(len(x_columns)):
            tr_copy[[x_columns[i], y_columns[i], 'phase']] = tr_copy[[x_columns[i], y_columns[i], 'phase']].groupby('phase').apply(lambda group: group.interpolate(method='linear', limit_area='inside'))

    
    tr_copy = sign_u.calc_player_velocities(tr_copy,smoothing=True,filter_='moving_average')
    
    home_keepers = []
    away_keepers = []

    for i,r in tr_copy.iterrows():
        for p in r.home_team:
            if p['role']==1:
                home_keepers.append(p['jersey_number'])
                break
        for p in r.away_team:
            if p['role']==1:
                away_keepers.append(p['jersey_number'])
                break
        break
        
    for i,r in tr_copy[::-1].reset_index().iterrows():
        for p in r.home_team:
            if p['role']==1:
                if p['jersey_number'] not in home_keepers:
                    home_keepers.append(p['jersey_number'])
                break
        for p in r.away_team:
            if p['role']==1:
                if p['jersey_number'] not in away_keepers:
                    away_keepers.append(p['jersey_number'])
                break
        break
    
    game_info['home_keepers'] = home_keepers
    game_info['away_keepers'] = away_keepers
    
    return tr_copy, game_info, True