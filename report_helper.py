from IPython.display import HTML
from base64 import b64encode
from bisect import bisect_left
import ast
import numpy as np
from Libraries import signality_utilities as sign_u
import pandas as pd

def take_closest(myList, myNumber):
    """
    Assumes myList is sorted. Returns closest value to myNumber.

    If two numbers are equally close, return the smallest number.
    """
    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before
    
def show_video(video_path, video_width = 600):
    video_file = open(video_path, "r+b").read()

    video_url = f"data:video/mp4;base64,{b64encode(video_file).decode()}"
    return HTML(f"""<video width={video_width} controls><source src="{video_url}"></video>""")

def preprocess_wyscout(wyscout_df, team_home_signality, team_away_signality):
    offset_1 = float(wyscout_df[wyscout_df.matchPeriod=='1H'].iloc[0].matchTimestamp.split(':')[-1])
    offset_2 = float(wyscout_df[wyscout_df.matchPeriod=='2H'].iloc[0].matchTimestamp.split(':')[-1])
    wyscout_df['offset'] = [offset_1 if x == '1H' else offset_2 for x in wyscout_df.matchPeriod]
    wyscout_df['approx_match_time'] = [(x*60+float(y.split(':')[-1])-z-45*60*(int(a.replace('H',''))-1))*1000 for x,y,z,a in zip(wyscout_df.minute, wyscout_df.matchTimestamp, wyscout_df.offset, wyscout_df.matchPeriod)]
    wyscout_df['isGoal'] = [x['isGoal'] if x != None else False for x in wyscout_df.shot]
    wyscout_df['attack'] = [1 if x != None and 'attack' in x['types'] else 0 for x in wyscout_df.possession]
    wyscout_df['possession_id'] = [x['id'] if x != None else -1 for x in wyscout_df.possession]
    wyscout_df['possession_team_id'] = [x['team']['id'] if x != None else -1 for x in wyscout_df.possession]
    wyscout_df['possession_team_name'] = [x['team']['name'] if x != None else -1 for x in wyscout_df.possession]
    wyscout_df['possession_team'] = [0 if x == -1 else 'home' if wyscout_to_signality[x] == team_home_signality else 'away' for x in wyscout_df.possession_team_name] #wyscout_to_signality
    wyscout_df['possession_duration'] = [x['duration'] if x != None else -1 for x in wyscout_df.possession]
    wyscout_df['set_piece'] = [1 if x != None and 'set_piece_attack' in x['types'] else 0 for x in wyscout_df.possession]
    wyscout_df['throw_in'] = [1 if x != None and 'throw_in' in x['types'] else 0 for x in wyscout_df.possession]
    wyscout_df['sp_ti'] = [1 if x == 1 or y == 1 else 0 for x,y in zip(wyscout_df.set_piece, wyscout_df.throw_in)]

signality_to_wyscout = {
    'Hammarby': 'Hammarby',
    'MjällbyAIF': 'Mjällby',
    'BKHäcken': 'Häcken',
    'IKSiriusFK': 'Sirius',
    'Djurgården': 'Djurgården',
    'Halmstad': 'Halmstad',
    'DegerforsIF': 'Degerfors',
    'IFKNorrköpingFK': 'IFK Norrköping',
    'IFElfsborg': 'Elfsborg',
    'MalmöFF': 'Malmö FF',
    'VarbergsBoISFC': 'Varberg',
    'IFKGöteborg': 'IFK Göteborg',
    'AIK': 'AIK',
    'Östersund': 'Östersunds FK',
    'Örebro': 'Örebro',
    'KalmarFF': 'Kalmar',
    'HelsingborgsIF': 'Helsingborg'
}

wyscout_to_signality = {v: k for k, v in signality_to_wyscout.items()}

def addTeamPossession(wyscout_df,signality_df):
    """
    Creates a new column in the wyscout data and signality data named possession_team and is 'home' or 'away'
    Parameters
    ----------
    wyscout_df : a regular wyscout Df
    signality_df : preprocessed signality df
    Returns
    None
    """
    #Now we have team in possession described as home and away for the wyscout data 
    # The only thing to do now is loop over the times
    for half in [1,2]:
        wyscout = wyscout_df[wyscout_df['matchPeriod'] == str(half)+"H"]
        signality = signality_df[signality_df['phase'] == half]
        possessions = set(wyscout['possession_id'])
        for i in possessions:
            poss_team = wyscout[wyscout['possession_id'] == i].iloc[0]['possession_team']
            if poss_team in ['home','away']:
                start = wyscout[wyscout['possession_id'] == i].iloc[0]['approx_match_time']
                end = wyscout[wyscout['possession_id'] == i].iloc[-1]['approx_match_time']
                pause_rows = signality[signality['match_time'].between(start,end)]
                indexes = pause_rows.index
                signality_df.loc[indexes, 'possession_team'] = poss_team
                
            else: 
                continue
                
def addGameState(wyscout_df_full, signality_df_full):
    """
    "{'primary': 'game_interruption', 'secondary': ['ball_out']}" is the game interruption most important
    There are also some "whistle's" that should be considered
    
    All in all Primary_type = game_interruption can be considered "not running" until the next event
    The infraction primary_type is also almost always followed by a break in play (>99% of time) so include these
    Offside certainly are always followed by free kicks, so include these
    
    Other things to study are: shot, shot_against, 
    # The time between a shot that results in a goal (which can be found through isGoal column)
    and the next pass is dead time (celebrating goal). 
    
    
    Summary: 
        1. Game_interruption can be compared to ball coordinates being unknown position in signality data for better accuracy
        2. Offside and infraction are the same in the sense that I look at the timestamp of the following event
        3. Goals are different as I look at shot with isGoal = "True" and then take the time of the next 'pass' event
    Parameters
    ----------
    wyscout_df : The wyscout data for the game which I want to find game state for 
    signality_df : Signality data that has gone through the preprocess written by Joakim
    Returns: The same signality dataframe but with 'state' having changed now to 'paused' when ball is not in play
    -------
    None.
    """
    addGoodColumns(wyscout_df_full)

    for half in [1,2]:
        finalarray = np.array([[0,0,0]])
        wyscout_df = wyscout_df_full[wyscout_df_full['matchPeriod'] == str(half)+"H"]
        signality_df = signality_df_full[signality_df_full['phase'] == half]        
    
        # Handling the game_interruption 
        interrupted = wyscout_df[wyscout_df['primary_type']=='game_interruption']
        start_times = interrupted['approx_match_time'].tolist()
        indexes = list(interrupted.index)
        end_times = []
        for i in indexes:
            if i+1 == len(wyscout_df_full):
                del start_times[-1]
                break
            end_times.append( float( wyscout_df.at[i+1,'approx_match_time'] ) )
        failures = []
        final_start_times = []
        #final_end_times = []
        for event in start_times:
            timerange = 1*40
            temp_df = signality_df[signality_df['match_time'].between(event-timerange, event+timerange, inclusive = True)]
            if len(temp_df) == 0:
                failures.append(start_times.index(event))
                continue
            while (temp_df.iloc[0]['ball.position']!= None) & (temp_df.iloc[-1]['ball.position']!= None):
                timerange = timerange+(10*40)
                temp_df = signality_df[signality_df['match_time'].between(event-timerange, event+timerange, inclusive = True)]
                if timerange > (200*40):    #in case there is no ball position
                    #print("Failed for the case with guessed time: ", event)
                    # Decide what to do: Add the guessed time? Or continue saying game is running?
                    break
            #return temp_df # This instantly doesn't work because ball is in stupid position
            try: 
                final_start_times.append(temp_df[temp_df['ball.position'].isna()].iloc[0]['match_time'])
            except:
                failures.append(start_times.index(event))
                #del end_times[start_times_new.index(event)]
                #start_times_new.remove(event)
                #print(temp_df['ball.position'])
                #sys.exit()
            #temp_time_diff = final_start_times[-1]-event
            #final_end_times
        # One could make the argument for differing between goal-kicks/corners and throw ins here and
        # handle the end-times accordingly. for now  though I'm just gonna not care
        removes = [end_times[a] for a in failures]
        for i in removes:
            end_times.remove(i)
        array = np.transpose(np.array([final_start_times, end_times, [half]*len(end_times)]))
        finalarray = np.concatenate((finalarray,array),axis = 0)
        
        # Same formula handles the offsides and infractions - but this is less precise as I can't use the ball coordinates
        interrupted = wyscout_df[wyscout_df['primary_type'].isin(['offside', 'infraction'])]
        start_times = interrupted['approx_match_time'].tolist()
        indexes = list(interrupted.index)
        end_times = []
        for i in indexes:
            if i+1 == len(wyscout_df_full) or (half==1 and i+1 == len(wyscout_df)):
                del start_times[-1]
                break
            end_times.append( float( wyscout_df.at[i+1,'approx_match_time'] ) )
        array = np.transpose(np.array([start_times, end_times, [half]*len(end_times)]))
        finalarray = np.concatenate((finalarray,array),axis = 0)
        
        # Handles the goals
        goals = wyscout_df[wyscout_df['isGoal'] == 'True']
        start_times = goals['approx_match_time'].tolist()
        indexes = list(goals.index)
        end_times = []
        for i in indexes:
            r = i+1
            while wyscout_df.at[r,'primary_type']!='pass':
                r = r+1
            end_times.append(float(wyscout_df.at[r,'approx_match_time']))
        array = np.transpose(np.array([start_times, end_times, [half]*len(end_times)]))
        finalarray = np.concatenate((finalarray,array),axis = 0)
        finalarray = np.delete(finalarray,0, axis = 0)
        
        for row in finalarray:
            start, end, half = row[0], row[1],row[2]
            pause_rows = signality_df[signality_df['match_time'].between(start,end)]
            indexes = pause_rows.index
            signality_df_full.loc[indexes, 'state'] = 'paused'
            
def addGoodColumns(wyscout_df, losses = False, passing = True):     
    for i, row in wyscout_df.iterrows():
        location = row['location'] ## Here's where I'm working
        event = row['type']
        primary_type = event['primary']
        tempteam = row['team']
        try:
            teamname = tempteam['name']
            wyscout_df.at[i,'teamname'] = teamname
        except:
            pass
        pass_spec = row['pass']
        wyscout_df.at[i,'teamId'] = int(tempteam['id'])
        player = row['player']
        playerId = player['id']
        playername = player['name']
        try: 
            wyscout_df.at[i,'playerId'] = int(playerId)
        except: 
            pass
        if playername == None:
            playername = 'Unknown Player'
        wyscout_df.at[i, 'playername'] = playername
        wyscout_df.at[i,'primary_type'] = primary_type
        try: 
            wyscout_df.at[i, 'loc_x'] = int(location['x'])
            wyscout_df.at[i, 'loc_y'] = int(location['y'])
        except: 
            pass
        
        if (primary_type == 'pass') & passing:
            wyscout_df.at[i,'pass_accurate'] = pass_spec['accurate']
            wyscout_df.at[i,'recipient'] = pass_spec['recipient']['id']
            wyscout_df.at[i,'end_x'] = int(pass_spec['endLocation']['x'])
            wyscout_df.at[i,'end_y'] = int(pass_spec['endLocation']['y'])
            
        if row['type']['primary'] == 'touch':
            if 'secondary' in row['type'] and 'loss' in row['type']['secondary']:
                wyscout_df.at[i, 'loss'] = 'True'
        if row['type']['primary'] == 'duel':
            if 'secondary' in row['type'] and 'loss' in row['type']['secondary']:
                wyscout_df.at[i, 'loss'] = 'True'
                

def compute_metrics(df_downsampled, attacks_df, attack_id, match_time_values, game_info, params):
    attacks_df_current = attacks_df[attacks_df.possession_id == attack_id].copy()
    attack_start_time = int(attacks_df_current.approx_match_time.iloc[0])
    attack_end_time = int(attacks_df_current.approx_match_time.iloc[-1])
    attack_phase = int(attacks_df_current.matchPeriod.iloc[0].replace('H',''))
    match_times = match_time_values[attack_phase-1]
    team = attacks_df_current.possession_team.iloc[0]
    team_name = attacks_df_current.iloc[0].possession['team']['name']

    frame_start = match_times[match_times == take_closest(match_times.values, attack_start_time)].index[0]
    frame_end = match_times[match_times == take_closest(match_times.values, attack_end_time)].index[0]

    df_chain = df_downsampled.iloc[frame_start:frame_end+1].copy()

    attack_metrics_frames = []
    cached_contributions, freeze_frames, max_generation, max_run = {}, {}, {}, {}     
    x_columns = [c for c in df_chain.columns if c[-2:].lower()=='.x']
    players_on_pitch = df_chain.iloc[0][x_columns].notnull().sum()
    players_on_pitch = players_on_pitch if players_on_pitch > 16 else 17 # nr 17 is arbitrary
    if df_chain['total_team_len'].mean() < players_on_pitch-0.02*players_on_pitch:
        return

    x_columns_team = [c for c in df_downsampled.columns if c[-2:].lower()=='.x' and team in c] 
    players_contribution = {}
    players_generation = {}

    for i in range(frame_start, frame_end+1):
        track_frame = i
        close_to_ball = 0
        if df_downsampled.iloc[track_frame]['ball.position'] == None:
            continue
        ball_position = np.array([df_downsampled.iloc[track_frame]['ball']['x'], df_downsampled.iloc[track_frame]['ball']['y']])
        for col in x_columns_team:
            x_col = col
            y_col = col[:-1]+'y'
            player_position = np.array([df_downsampled.iloc[track_frame][x_col], df_downsampled.iloc[track_frame][y_col]])
            if np.linalg.norm(ball_position-player_position) <= 2:
                close_to_ball = 1
                break
        if close_to_ball == 0:
            continue

        new_generation, new_contribution, cached_contributions = sign_u.pcpi_generation(df_downsampled, track_frame, team, game_info, params, downsample_factor=4, cached_contributions=cached_contributions)
        for p in new_contribution:
            if p not in players_contribution:
                players_contribution[p] = ([new_contribution[p]], [track_frame])
            else:
                players_contribution[p][0].append(new_contribution[p])
                players_contribution[p][1].append(track_frame)

        for p in new_generation:
            if p not in players_generation:
                players_generation[p] = ([new_generation[p]], [track_frame])
            else:
                players_generation[p][0].append(new_generation[p])
                players_generation[p][1].append(track_frame)

    players_contribution_final = greedy_metric_combination(players_contribution)
    players_generation_final = greedy_metric_combination(players_generation)

    for p in players_contribution_final:
        for p_run in players_contribution_final[p]:
            run_value = p_run[0]
            run_frame = p_run[1]

            player_x = df_downsampled.iloc[run_frame][team+'_team_aux.'+str(p)+'.x']
            player_y = df_downsampled.iloc[run_frame][team+'_team_aux.'+str(p)+'.y']

            to_append = {'player_nr': p, 'team': team, 'team_name': wyscout_to_signality[team_name], 'x': player_x, 'y': player_y, 'run': run_value, 'generation': 0, 'run_frame': run_frame, 'generation_frame': -1, 'possession_id': attack_id}
            attack_metrics_frames.append(to_append)

    for p in players_generation_final:
        for p_generation in players_generation_final[p]:
            generation_value = p_generation[0]
            generation_frame = p_generation[1]

            player_x = df_downsampled.iloc[generation_frame][team+'_team_aux.'+str(p)+'.x']
            player_y = df_downsampled.iloc[generation_frame][team+'_team_aux.'+str(p)+'.y']

            to_append = {'player_nr': p, 'team': team, 'team_name': wyscout_to_signality[team_name], 'x': player_x, 'y': player_y, 'run': 0, 'generation': generation_value, 'run_frame': -1, 'generation_frame': generation_frame, 'possession_id': attack_id}
            attack_metrics_frames.append(to_append)

    df_metrics = pd.DataFrame(attack_metrics_frames)
    df_metrics['run_frame'] = df_metrics['run_frame']*4
    df_metrics['generation_frame'] = df_metrics['generation_frame']*4
    
    return df_metrics

def greedy_metric_combination(players_contribution):
    players_contribution_final = {}

    for p in players_contribution:
        contribution_values = np.array(players_contribution[p][0])
        contribution_frames = np.array(players_contribution[p][1])
        players_contribution_final[p] = []
        first_id = np.argmax(contribution_values)
        
        for direction in ['left', 'right']:
            move = 1
            current_id = first_id
            
            while move == 1:
                current_value = contribution_values[current_id]
                current_frame = contribution_frames[current_id]

                if current_id != first_id:
                    players_contribution_final[p].append((current_value, current_frame))
                elif direction == 'left':
                    players_contribution_final[p].append((current_value, current_frame))

                if direction == 'right':
                    next_frames = contribution_frames[current_id+1:]
                    next_frames = next_frames[np.where((next_frames >= current_frame+19))]
                else:
                    next_frames = contribution_frames[:current_id]
                    next_frames = next_frames[np.where((next_frames <= current_frame-19))]

                if len(next_frames) == 0:
                    move = 0
                else:
                    if direction == 'right':
                        frames_next_new = next_frames[np.where((next_frames <= current_frame+25))]
                    else:
                        frames_next_new = next_frames[np.where((next_frames >= current_frame-25))]
                    if len(frames_next_new) == 0:
                        if direction == 'right':
                            current_id = np.where(contribution_frames==next_frames[0])[0][0]
                        else:
                            current_id = np.where(contribution_frames==next_frames[-1])[0][0]
                    else:
                        first_in_right = np.where(np.isin(contribution_frames,frames_next_new))[0][0]
                        last_in_right = np.where(np.isin(contribution_frames,frames_next_new))[0][-1]
                        new_values = contribution_values[first_in_right:last_in_right+1]
                        new_frames = contribution_frames[first_in_right:last_in_right+1]
                        highest_value_id = np.argmax(new_values)
                        new_frame = new_frames[highest_value_id]

                        current_id = np.where(contribution_frames==new_frame)[0][0]
    return players_contribution_final

def convert_metrics_frames(df_metrics):
    ff = {}
        
    for i,r in df_metrics.iterrows():
        if r.run_frame == -4:
            metric = 'generation'
        else:
            metric = 'run'
        run = r.run
        generation = r.generation
        run = round(run, 2)
        generation = round(generation, 2)
        x = r.x
        y = r.y
        text = ""
        frame_add = r[metric+'_frame']
        threshold_reached = 0
        
        if metric == 'run' and run >= 0.05:
            text += "Run: "+ str(run)
            threshold_reached = 1
        if metric == 'generation' and generation >= 0.05:
            text += " Generation: " + str(generation)
            threshold_reached = 1
        
        if threshold_reached == 1:
            if frame_add in ff:
                ff[frame_add].append((text, [x,y]))
            else:
                ff[frame_add] = [(text, [x,y])]
    return ff