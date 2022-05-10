import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import matplotlib.patheffects as PathEffects
import scipy.signal as signal
import pandas as pd
import numpy as np
import math
from numba import njit
from numba import NumbaTypeSafetyWarning
import warnings
"""
Created on Mon Apr  6 14:52:19 2020
@author: Laurie Shaw (@EightyFivePoint)
Additional and modified code to work with Signality tracking data
"""

def calc_player_velocities(track_df, smoothing=True, filter_='moving_average', window=7, polyorder=1, maxspeed = 12):
    """ calc_player_velocities( tracking_data )
    
    Calculate player velocities in x & y direciton, and total player speed at each timestamp of the tracking data
    
    Parameters
    -----------
        team: the tracking DataFrame for home or away team
        smoothing: boolean variable that determines whether velocity measures are smoothed. Default is True.
        filter: type of filter to use when smoothing the velocities. Default is moving average
        window: smoothing window size in # of frames
        polyorder: order of the polynomial for the Savitzky-Golay filter. Default is 1 - a linear fit to the velcoity, so gradient is the acceleration
        maxspeed: the maximum speed that a player can realisitically achieve (in meters/second). Speed measures that exceed maxspeed are tagged as outliers and set to NaN. 
        
    Returrns
    -----------
       team : the tracking DataFrame with columns for speed in the x & y direction and total speed added
    """    
    # Get the player ids
    player_ids = np.unique( [ c[:-2] for c in track_df.columns if c[:4] in ['home','away'] and 'aux' in c] )
    # Calculate the timestep from one frame to the next. Should always be 0.04 within the same half
    dt = track_df['match_time'].diff()/1000
    
    # index of first frame in second half
    second_half_idx = track_df[track_df.phase==2].iloc[0].name
    
    # estimate velocities for players in team
    for player in player_ids: # cycle through players individually
        # difference player positions in timestep dt to get unsmoothed estimate of velicity
        vx = track_df[player+".x"].diff() / dt
        vy = track_df[player+".y"].diff() / dt

        if maxspeed>0:
            # remove unsmoothed data points that exceed the maximum speed (these are most likely position errors)
            raw_speed = np.sqrt( vx**2 + vy**2 )
            vx[ raw_speed>maxspeed ] = np.nan
            vy[ raw_speed>maxspeed ] = np.nan
            
        if smoothing:
            if filter_=='Savitzky-Golay':
                # calculate first half velocity
                print(vx.loc[:second_half_idx] )
                vx.loc[:second_half_idx] = signal.savgol_filter(vx.loc[:second_half_idx],window_length=window,polyorder=polyorder)
                vy.loc[:second_half_idx] = signal.savgol_filter(vy.loc[:second_half_idx],window_length=window,polyorder=polyorder)        
                # calculate second half velocity
                vx.loc[second_half_idx:] = signal.savgol_filter(vx.loc[second_half_idx:],window_length=window,polyorder=polyorder)
                vy.loc[second_half_idx:] = signal.savgol_filter(vy.loc[second_half_idx:],window_length=window,polyorder=polyorder)
            elif filter_=='moving average':
                ma_window = np.ones( window ) / window 
                # calculate first half velocity
                vx.loc[:second_half_idx] = np.convolve( vx.loc[:second_half_idx] , ma_window, mode='same' ) 
                vy.loc[:second_half_idx] = np.convolve( vy.loc[:second_half_idx] , ma_window, mode='same' )      
                # calculate second half velocity
                vx.loc[second_half_idx:] = np.convolve( vx.loc[second_half_idx:] , ma_window, mode='same' ) 
                vy.loc[second_half_idx:] = np.convolve( vy.loc[second_half_idx:] , ma_window, mode='same' ) 
                
        
        # put player speed in x,y direction, and total speed back in the data frame
        track_df[player + ".vx"] = vx
        track_df[player + ".vy"] = vy
        track_df[player + ".speed"] = np.sqrt( vx**2 + vy**2 )

    return track_df

def plot_pitch( field_dimen = (106.0,68.0), field_color ='white', linewidth=2, markersize=20):
    """ plot_pitch
    
    Plots a soccer pitch. All distance units converted to meters.
    
    Parameters
    -----------
        field_dimen: (length, width) of field in meters. Default is (106,68)
        field_color: color of field. options are {'green','white'}
        linewidth  : width of lines. default = 2
        markersize : size of markers (e.g. penalty spot, centre spot, posts). default = 20
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    fig,ax = plt.subplots(figsize=(12,8)) # create a figure 
    # decide what color we want the field to be. Default is green, but can also choose white
    if field_color=='green':
        ax.set_facecolor('mediumseagreen')
        lc = 'whitesmoke' # line color
        pc = 'w' # 'spot' colors
    elif field_color=='white':
        lc = 'k'
        pc = 'k'
    # ALL DIMENSIONS IN m
    border_dimen = (3,3) # include a border arround of the field of width 3m
    meters_per_yard = 0.9144 # unit conversion from yards to meters
    half_pitch_length = field_dimen[0]/2. # length of half pitch
    half_pitch_width = field_dimen[1]/2. # width of half pitch
    signs = [-1,1] 
    # Soccer field dimensions typically defined in yards, so we need to convert to meters
    goal_line_width = 8*meters_per_yard
    box_width = 20*meters_per_yard
    box_length = 6*meters_per_yard
    area_width = 44*meters_per_yard
    area_length = 18*meters_per_yard
    penalty_spot = 12*meters_per_yard
    corner_radius = 1*meters_per_yard
    D_length = 8*meters_per_yard
    D_radius = 10*meters_per_yard
    D_pos = 12*meters_per_yard
    centre_circle_radius = 10*meters_per_yard
    # plot half way line # center circle
    ax.plot([0,0],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
    ax.scatter(0.0,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
    y = np.linspace(-1,1,50)*centre_circle_radius
    x = np.sqrt(centre_circle_radius**2-y**2)
    ax.plot(x,y,lc,linewidth=linewidth)
    ax.plot(-x,y,lc,linewidth=linewidth)
    for s in signs: # plots each line seperately
        # plot pitch boundary
        ax.plot([-half_pitch_length,half_pitch_length],[s*half_pitch_width,s*half_pitch_width],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length],[-half_pitch_width,half_pitch_width],lc,linewidth=linewidth)
        # goal posts & line
        ax.plot( [s*half_pitch_length,s*half_pitch_length],[-goal_line_width/2.,goal_line_width/2.],pc+'s',markersize=6*markersize/20.,linewidth=linewidth)
        # 6 yard box
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[box_width/2.,box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*box_length],[-box_width/2.,-box_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*box_length,s*half_pitch_length-s*box_length],[-box_width/2.,box_width/2.],lc,linewidth=linewidth)
        # penalty area
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[area_width/2.,area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length,s*half_pitch_length-s*area_length],[-area_width/2.,-area_width/2.],lc,linewidth=linewidth)
        ax.plot([s*half_pitch_length-s*area_length,s*half_pitch_length-s*area_length],[-area_width/2.,area_width/2.],lc,linewidth=linewidth)
        # penalty spot
        ax.scatter(s*half_pitch_length-s*penalty_spot,0.0,marker='o',facecolor=lc,linewidth=0,s=markersize)
        # corner flags
        y = np.linspace(0,1,50)*corner_radius
        x = np.sqrt(corner_radius**2-y**2)
        ax.plot(s*half_pitch_length-s*x,-half_pitch_width+y,lc,linewidth=linewidth)
        ax.plot(s*half_pitch_length-s*x,half_pitch_width-y,lc,linewidth=linewidth)
        # draw the D
        y = np.linspace(-1,1,50)*D_length # D_length is the chord of the circle that defines the D
        x = np.sqrt(D_radius**2-y**2)+D_pos
        ax.plot(s*half_pitch_length-s*x,y,lc,linewidth=linewidth)
        
    # remove axis labels and ticks
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_xticks([])
    ax.set_yticks([])
    # set axis limits
    xmax = field_dimen[0]/2. + border_dimen[0]
    ymax = field_dimen[1]/2. + border_dimen[1]
    ax.set_xlim([-xmax,xmax])
    ax.set_ylim([-ymax,ymax])
    ax.set_axisbelow(True)
    return fig,ax

def plot_frame( frame, figax=None, team_colors=('r','b'), field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=15, PlayerAlpha=0.7, annotate=[] ):
    """ plot_frame( hometeam, awayteam )
    
    Plots a frame of Signality  tracking data (player positions and the ball) on a football pitch. All distances should be in meters.
    
    Parameters
    -----------
        frame: row of the tracking data frame
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot, 
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        annotate: list of which teams to annotate ('home' or 'away' possible values)
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    if figax is None: # create new pitch 
        fig,ax = plot_pitch( field_dimen = field_dimen )
    else: # overlay on a previously generated pitch
        fig,ax = figax # unpack tuple
    # plot home & away teams in order
    for team,color in zip( ['home', 'away'], team_colors) :
        x_values = []
        y_values = []
        vx_values = []
        vy_values = []
        jersey_values = []
        player_ids = np.unique( [ int(c.split('.')[1]) for c in frame.keys() if c[-2:] == '.x' and team+'_team' in c] )
        for p in player_ids:
            j_number = str(p)
            if np.isnan(frame[team+'_team_aux.'+str(j_number)+'.x']) == False:
                jersey_values.append(j_number)
                x_values.append(frame[team+'_team_aux.'+j_number+'.x'])
                y_values.append(frame[team+'_team_aux.'+j_number+'.y'])
                vx_values.append(frame[team+'_team_aux.'+j_number+'.vx'])
                vy_values.append(frame[team+'_team_aux.'+j_number+'.vy'])
        ax.plot( x_values, y_values, color=color, marker='o', linestyle='None', markeredgecolor='black', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
        if include_player_velocities:
                ax.quiver( x_values, y_values, vx_values, vy_values, color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
        if team in annotate:
            for i in range(len(jersey_values)):
                ax.annotate(int(jersey_values[i]),(x_values[i]+1.2,y_values[i]-1),size='x-large')
    # plot ball
    if frame.ball['x'] != None:
        ax.plot( frame.ball['x'], frame.ball['y'], 'ko', markersize=10, alpha=1.0, linewidth=0)
    return fig,ax

    
def save_match_clip(frames, fpath, fname='clip_test', figax=None, frames_per_second=25, team_colors=('r','b'), field_dimen = (106.0,68.0), include_player_velocities=False, PlayerMarkerSize=15, PlayerAlpha=0.7, freeze_frames={}, annotate=[]):
    """ save_match_clip( frames, fpath )
    
    Generates a movie from Signality tracking data, saving it in the 'fpath' directory with name 'fname'
    
    Parameters
    -----------
        frames: the rows in the dataframe to plot, eg df.iloc[20:120] for frames 20 to 120
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.gif'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Default is 0.7
        freeze_frames: dictionary of frames which should be frozen in the animation, eg {123: 'Pass'} as input will pause frame 123 and plot text 'pass'
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    INTERVAL = 40  # ms
    HOLD_MS  = 1000
    HOLD_COUNT = HOLD_MS // INTERVAL

    def frame_generator(frames, freeze_frames):
        counter = 0
        for frame in frames.iterrows():
            # Yield the frame first
            yield counter
            # If we should "sleep" here, yield None HOLD_COUNT times
            if frame[0] in freeze_frames:
                for _ in range(HOLD_COUNT):
                    yield None
            counter += 1
                
    # create football pitch
    if figax is None:
        fig,ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
    # Generate movie
    print("Generating movie...",end='')
    def animate(i, begin_frame):
        if i is None:
            return
        
        row = frames.iloc[i]
        for figobj in figobjs:
            figobj.remove()
            figobjs.remove(figobj)
        
        #figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
        for team,color in zip( ['home', 'away'], team_colors) :
            x_values = []
            y_values = []
            jersey_values = []
            vx_values = []
            vy_values = []
            player_ids = np.unique( [ int(c.split('.')[1]) for c in row.keys() if c[-2:] == '.x' and team+'_team' in c] )
            for p in player_ids:
                j_number = str(p)
                if np.isnan(row[team+'_team_aux.'+str(j_number)+'.x']) == False:
                    jersey_values.append(j_number)
                    x_values.append(row[team+'_team_aux.'+j_number+'.x'])
                    y_values.append(row[team+'_team_aux.'+j_number+'.y'])
                    vx_values.append(row[team+'_team_aux.'+j_number+'.vx'])
                    vy_values.append(row[team+'_team_aux.'+j_number+'.vy'])
                
            objs, = ax.plot( x_values, y_values, color=color, marker='o', linestyle='None', markeredgecolor='black', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
            figobjs.append(objs)
            if include_player_velocities:
                objs = ax.quiver( x_values, y_values, vx_values, vy_values, color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                figobjs.append(objs)
            if team in annotate:
                for j in range(len(jersey_values)):
                    objs = ax.annotate(int(jersey_values[j]),(x_values[j]+1.2,y_values[j]-1),size='x-large')
                    figobjs.append(objs)
        # plot ball
        if row.ball['x'] != None:
            objs, = ax.plot( row.ball['x'], row.ball['y'], 'ko', markersize=10, alpha=1.0, linewidth=0)
            figobjs.append(objs)
        # include match time at the top
        frame_minute =  int( (row['match_time']/1000)/60. )
        frame_second =  ( (row['match_time']/1000.)/60. - frame_minute ) * 60
        if row['phase'] == 2:
            frame_minute += 45
        timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
        timestring += " Frame: "+str(i+begin_frame)
        objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
        figobjs.append(objs)
        
        if (i+begin_frame) in freeze_frames:
            for freeze_frame in freeze_frames[i+begin_frame]:
                if 'Generation' not in freeze_frame[0] or 'Run' not in freeze_frame[0]:
                    text_to_plot = freeze_frame[0]
                    text_to_plot = text_to_plot.replace('Run', 'Control')
                    objs = ax.annotate(text_to_plot,(freeze_frame[1][0]-2.40,freeze_frame[1][1]+1.25),size='medium')
                    objs.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='white')])
                    figobjs.append(objs)
                else:
                    objs = ax.annotate(freeze_frame[0].split(' Generation')[0],(freeze_frame[1][0]-2.40,freeze_frame[1][1]+1.25),size='medium')
                    objs.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='white')])
                    figobjs.append(objs)
                    objs = ax.annotate("Generation"+ freeze_frame[0].split(' Generation')[1],(freeze_frame[1][0]-4,freeze_frame[1][1]+2.75),size='medium')
                    objs.set_path_effects([PathEffects.withStroke(linewidth=0.75, foreground='white')])
                    figobjs.append(objs)
                    

            
    figobjs = []
    save_count = len(list(frame_generator(frames, freeze_frames)))
    anim = animation.FuncAnimation(fig, animate, fargs=[frames.iloc[0].name], frames=frame_generator(frames, freeze_frames), interval=40,save_count=save_count)
    FFwriter = animation.FFMpegWriter(fps=25)
    anim.save(fname+".mp4", writer=FFwriter)
    print("done")
    plt.clf()
    plt.close(fig)       
    
def check_offsides( attacking_players, defending_players, ball_position, frame, game_info, attacking_team, verbose=False, tol=0.2):
    """
    check_offsides( attacking_players, defending_players, ball_position, GK_numbers, verbose=False, tol=0.2):
    
    checks whetheer any of the attacking players are offside (allowing for a 'tol' margin of error). Offside players are removed from 
    the 'attacking_players' list and ignored in the pitch control calculation.
    
    Parameters
    -----------
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_position: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        frame: the row of the desired frame
        game_info: metadata of the Signality tracking data
        attacking_team: home or away depending on the attacking team
        verbose: if True, print a message each time a player is found to be offside
        tol: A tolerance parameter that allows a player to be very marginally offside (up to 'tol' m) without being flagged offside. Default: 0.2m
            
    Returrns
    -----------
        attacking_players: list of 'player' objects for the players on the attacking team with offside players removed
    """    
    
    #find attacking direction and assign -1 or 1 depending on that
    if find_attacking_direction(frame, game_info, attacking_team) == 'right':
        defending_half = 1
    else:
        defending_half = -1
    # find the x-position of the second-deepest defeending player (including GK)
    second_deepest_defender_x = sorted( [defending_half*p.position[0] for p in defending_players], reverse=True )[1]
    # define offside line as being the maximum of second_deepest_defender_x, ball position and half-way line
    offside_line = max(second_deepest_defender_x,defending_half*ball_position[0],0.0)+tol
    # any attacking players with x-position greater than the offside line are offside
    if verbose:
        for p in attacking_players:
            if p.position[0]*defending_half>offside_line:
                print("player %s in %s team is offside" % (p.id, p.playername) )
    attacking_players = [p for p in attacking_players if p.position[0]*defending_half<=offside_line]
    return attacking_players

class player(object):
    """
    player() class
    
    Class defining a player object that stores position, velocity, time-to-intercept and pitch control contributions for a player
    
    __init__ Parameters
    -----------
    player_dict: dictionary of a player with jersey number, role, position and velocity
    teamname: team name
    playername: player name
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
    
    methods include:
    -----------
    simple_time_to_intercept(r_final): time take for player to get to target position (r_final) given current position
    probability_intercept_ball(T): probability player will have controlled ball at time T given their expected time_to_intercept
    
    """
    # player object holds position, velocity, time-to-intercept and pitch control contributions for each player
    def __init__(self,p_id, is_keeper, teamname, playername, position, velocity,params):
        self.id = p_id
        self.is_gk = is_keeper
        self.teamname = teamname
        self.playername = playername
        self.vmax = params['max_player_speed'] # player max speed in m/s. Could be individualised
        self.reaction_time = params['reaction_time'] # player reaction time in 's'. Could be individualised
        self.tti_sigma = params['tti_sigma'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_att = params['lambda_att'] # standard deviation of sigmoid function (see Eq 4 in Spearman, 2018)
        self.lambda_def = params['lambda_gk'] if self.is_gk else params['lambda_def'] # factor of 3 ensures that anything near the GK is likely to be claimed by the GK
        self.position = position
        self.velocity = velocity
        self.PPCF = 0. # initialise this for later
    
    def simple_time_to_intercept(self, r_final):
        self.PPCF = 0. # initialise this for later
        # Time to intercept assumes that the player continues moving at current velocity for 'reaction_time' seconds
        # and then runs at full speed to the target position.
        r_reaction = self.position + self.velocity*self.reaction_time
        self.time_to_intercept = self.reaction_time + np.linalg.norm(r_final-r_reaction)/self.vmax
        return self.time_to_intercept

    def probability_intercept_ball(self,T):
        # probability of a player arriving at target location at time 'T' given their expected time_to_intercept (time of arrival), as described in Spearman 2018
        f = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/self.tti_sigma * (T-self.time_to_intercept) ) )
        return f

def default_model_params(time_to_control_veto=3):
    """
    default_model_params()
    
    Returns the default parameters that define and evaluate the model. See Spearman 2018 for more details.
    
    Parameters
    -----------
    time_to_control_veto: If the probability that another team or player can get to the ball and control it is less than 10^-time_to_control_veto, ignore that player.
    
    
    Returns
    -----------
    
    params: dictionary of parameters required to determine and calculate the model
    
    """
    # key parameters for the model, as described in Spearman 2018
    params = {}
    # model parameters
    params['max_player_accel'] = 7. # maximum player acceleration m/s/s, not used in this implementation
    params['max_player_speed'] = 5. # maximum player speed m/s
    params['reaction_time'] = 0.7 # seconds, time taken for player to react and change trajectory. Roughly determined as vmax/amax
    params['tti_sigma'] = 0.45 # Standard deviation of sigmoid function in Spearman 2018 ('s') that determines uncertainty in player arrival time
    params['kappa_def'] =  1. # kappa parameter in Spearman 2018 (=1.72 in the paper) that gives the advantage defending players to control ball, I have set to 1 so that home & away players have same ball control probability
    params['lambda_att'] = 4.3 # ball control parameter for attacking team
    params['lambda_def'] = 4.3 * params['kappa_def'] # ball control parameter for defending team
    params['lambda_gk'] = params['lambda_def']*3.0 # make goal keepers must quicker to control ball (because they can catch it)
    params['average_ball_speed'] = 15. # average ball travel speed in m/s
    # numerical parameters for model evaluation
    params['int_dt'] = 0.04 # integration timestep (dt)
    params['max_int_time'] = 10 # upper limit on integral time
    params['model_converge_tol'] = 0.01 # assume convergence when PPCF>0.99 at a given location.
    # The following are 'short-cut' parameters. We do not need to calculated PPCF explicitly when a player has a sufficient head start. 
    # A sufficient head start is when the a player arrives at the target location at least 'time_to_control' seconds before the next player
    params['time_to_control_att'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_att'])
    params['time_to_control_def'] = time_to_control_veto*np.log(10) * (np.sqrt(3)*params['tti_sigma']/np.pi + 1/params['lambda_def'])
    return params
    
def initialise_players(row,game_info,params):
    """
    initialise_players(team,teamname,params)
    
    create a list of player objects that holds their positions and velocities from the tracking data dataframe 
    
    Parameters
    -----------
    
    row: row (i.e. instant) of tracking Dataframe
    game_info: metadata from signality data
    params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returns
    -----------
    
    home_team_players: list of home team player objects for the team at at given instant
    away_team_players: list of away team player objects for the team at at given instant
    
    """    
    home_team_players = []
    away_team_players = []
    
    for team in ['home', 'away']:
        if team == 'home':
            player_ids = np.unique( [ int(c.split('.')[1]) for c in row.keys() if c[-2:] == '.x' and 'home_team' in c] )
            players_names = game_info['team_home_players']
            keepers = game_info['home_keepers']
        else:
            player_ids = np.unique( [ int(c.split('.')[1]) for c in row.keys() if c[-2:] == '.x' and 'away_team' in c] )
            players_names = game_info['team_away_players']
            keepers = game_info['away_keepers']
        
        for p_id in player_ids:
            if p_id == -1:
                continue
            if len(list(filter(lambda player_name: player_name['jersey_number'] == p_id, players_names))) == 0:
                continue
            player_name = list(filter(lambda player_name: player_name['jersey_number'] == p_id, players_names))[0]['name']
            player_position = np.array( [ row[team+'_team_aux.'+str(p_id)+'.x'], row[team+'_team_aux.'+str(p_id)+'.y'] ] )
            if np.isnan(player_position[0]):
                continue
            player_velocity = np.array( [ row[team+'_team_aux.'+str(p_id)+'.vx'], row[team+'_team_aux.'+str(p_id)+'.vy'] ] )
            is_keeper = False
            if p_id in keepers:
                is_keeper = True
            to_add = player(p_id, is_keeper, game_info['team_'+team+'_name'], player_name, player_position, player_velocity,params)
            
            if team == 'home':
                home_team_players.append(to_add)
            else:
                away_team_players.append(to_add)

  
    
    return home_team_players, away_team_players

def generate_pitch_control_for_frame(df, frame_id_start, params, game_info, team, field_dimen = (106.,68.,), n_grid_cells_x = 50, offsides=True, x_threshold=-1):
    """ generate_pitch_control_for_frame
    
    Evaluates pitch control surface over the entire field at the moment of the given event (determined by the index of the event passed as an input)
    
    Parameters
    -----------
        event_id: Index (not row) of the event that describes the instant at which the pitch control surface should be calculated
        events: Dataframe containing the event data
        tracking_home: tracking DataFrame for the Home team
        tracking_away: tracking DataFrame for the Away team
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        GK_numbers: tuple containing the player id of the goalkeepers for the (home team, away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        n_grid_cells_x: Number of pixels in the grid (in the x-direction) that covers the surface. Default is 50.
                        n_grid_cells_y will be calculated based on n_grid_cells_x and the field dimensions
        offsides: If True, find and remove offside atacking players from the calculation. Default is True.
        
    UPDATE (tutorial 4): Note new input arguments ('GK_numbers' and 'offsides')
        
    Returrns
    -----------
        PPCFa: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team,
               where 1 = full control for attacking team and 0 = full control for defending team
        xgrid: Positions of the pixels in the x-direction (field length)
        ygrid: Positions of the pixels in the y-direction (field width)
        PPCF_players_att: the pitch control surface containing PC probability for each player in attacking team, similar to PPCFa but on
                          an individual level
    """
    frame = df.iloc[frame_id_start]
    
    
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y
    xgrid = np.arange(n_grid_cells_x)*dx - field_dimen[0]/2. + dx/2.
    ygrid = np.arange(n_grid_cells_y)*dy - field_dimen[1]/2. + dy/2.
    # initialise pitch control grids for attacking and defending teams 
    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
        
    if frame['ball.position'] == None:
        phase = frame.phase
        ball_df = df[(df.index<frame_id_start) & (df['ball.position'].isna() == False) & (df.phase==phase)].copy()
        if len(ball_df) > 0:
            ball_start_pos = np.array([ball_df.iloc[-1]['ball.position'][0], ball_df.iloc[-1]['ball.position'][1]])
        else:
            ball_start_pos = np.array([0,0])
    else:
        ball_start_pos = np.array([frame['ball.position'][0],frame['ball.position'][1]])
    
    home_players, away_players = initialise_players(frame,game_info,params)
    
    # initialise player positions and velocities for pitch control calc (so that we're not repeating this at each grid cell position)
    if team=='home':
        attacking_players = home_players
        defending_players = away_players
    elif team=='away':
        defending_players = home_players
        attacking_players = away_players
    else:
        assert False, "Team in possession must be either home or away"
        
    attacking_players_ids = [p.id for p in attacking_players]
    PPCF_players_att = {}

    for p in attacking_players_ids:
        PPCF_players_att[p] = np.zeros( shape = (len(ygrid), len(xgrid)) )
            
    # find any attacking players that are offside and remove them from the pitch control calculation
    skip_x = np.array([])
    if offsides:
        attacking_players = check_offsides( attacking_players, defending_players, ball_start_pos, frame, game_info, team, verbose=False)
    if x_threshold != -1:
        direction_multiplier = 1
        if find_attacking_direction(frame, game_info, team) == 'left':
            direction_multiplier = -1
        xgrid_temp = xgrid * direction_multiplier
        skip_x = np.where(xgrid_temp<=x_threshold)[0]
         
    attacking_players_target = []
    defending_players_target = []
    dT_list_targets = []
    xy_counter = 0
    approx_dT = np.arange(15-params['int_dt'],15+params['max_int_time'],params['int_dt']) 
    approx_dT = int(approx_dT.shape[0] + approx_dT.shape[0]*0.2)
    for i in range( len(ygrid) ):
        for j in range( len(xgrid) ):
            if j in skip_x:
                continue
            target_position = np.array( [xgrid[j], ygrid[i]] )
            if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
                ball_travel_time = 0.0 
            else:
                # ball travel time is distance to target position from current ball position divided assumed average ball speed
                ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
            tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
            tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
            dT_target = np.ones(approx_dT)
            dT_target *= -1
            
            attacking_players_target_temp = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
            defending_players_target_temp = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]

            attacking_players_target_ids = np.zeros(len(attacking_players))
            attacking_players_target_att = np.zeros(len(attacking_players))
            attacking_players_target_tti = np.zeros(len(attacking_players))
            attacking_players_target_intercept = np.zeros(len(attacking_players))

            defending_players_target_ids = np.zeros(len(defending_players))
            defending_players_target_def = np.zeros(len(defending_players))
            defending_players_target_tti = np.zeros(len(defending_players))
            defending_players_target_intercept = np.zeros(len(defending_players))
            p_counter = 0
            for p in attacking_players_target_temp:
                attacking_players_target_ids[p_counter] = p.id
                attacking_players_target_att[p_counter] = p.lambda_att
                attacking_players_target_tti[p_counter] = p.tti_sigma
                attacking_players_target_intercept[p_counter] = p.time_to_intercept
                p_counter += 1
            p_counter = 0
            for p in defending_players_target_temp:
                defending_players_target_ids[p_counter] = p.id
                defending_players_target_def[p_counter] = p.lambda_def
                defending_players_target_tti[p_counter] = p.tti_sigma
                defending_players_target_intercept[p_counter] = p.time_to_intercept
                p_counter += 1
                
            attacking_players_target.append(np.array([attacking_players_target_att, attacking_players_target_tti, attacking_players_target_intercept, attacking_players_target_ids]))
            defending_players_target.append(np.array([defending_players_target_def, defending_players_target_tti, defending_players_target_intercept, defending_players_target_ids]))
            dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
            dT_target[:dT_array.shape[0]] = dT_array
            dT_list_targets.append(dT_target)
            xy_counter += 1
    # check probabilitiy sums within convergence
    checksum = np.sum( PPCFa + PPCFd ) / float(n_grid_cells_y*n_grid_cells_x ) 
    #assert 1-checksum < params['model_converge_tol'], "Checksum failed: %1.3f" % (1-checksum)
    warnings.simplefilter("ignore", category=NumbaTypeSafetyWarning)
    PPCFa,PPCF_players_att = calculate_pitch_control_parallel(np.array(attacking_players_target), np.array(defending_players_target), xgrid, ygrid, np.array(dT_list_targets), params['model_converge_tol'], params['int_dt'], np.array(attacking_players_ids), skip_x)

    return PPCFa,xgrid,ygrid,dict(PPCF_players_att)
    
@njit()
def calculate_pitch_control_parallel(attacking_players, defending_players, xgrid, ygrid, dT_list_targets, model_converge_tol, int_dt, attacking_players_ids, skip_x):

    PPCFa = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCFd = np.zeros( shape = (len(ygrid), len(xgrid)) )
    PPCF_players_att = {}
    
    for p in attacking_players_ids:
        PPCF_players_att[p] = np.zeros( shape = (len(ygrid), len(xgrid)) )
    xy_counter = 0
    for i in range( len(ygrid) ):
        for j in range( len(xgrid) ):
            if j in skip_x:
                continue
            dT_array = dT_list_targets[xy_counter]
            PPCFatt = np.zeros_like( dT_array )
            PPCFdef = np.zeros_like( dT_array )
            players_att = np.zeros(len(attacking_players[xy_counter][0]))
            players_def = np.zeros(len(defending_players[xy_counter][0]))
            k = 1
            ptot = 0.0
            while 1-ptot>model_converge_tol and k<dT_array.size: 
                T = dT_array[k]
                if T == -1: 
                    break
                for p in range(len(attacking_players[xy_counter][0])):
                    lambda_att = attacking_players[xy_counter][0][p]
                    tti_sigma = attacking_players[xy_counter][1][p]
                    intercept = attacking_players[xy_counter][2][p]
                    p_id = attacking_players[xy_counter][3][p]
                    if p_id == 0:
                        break
                    
                    proba_intercept = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/tti_sigma * (T-intercept) ) )
                    dPPCFdT = (1-PPCFatt[k-1]-PPCFdef[k-1])*proba_intercept*lambda_att
                    assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                    players_att[p] += (dPPCFdT*int_dt)
                    PPCFatt[k] += players_att[p]
                    PPCF_players_att[np.int32(p_id)][i,j] = players_att[p]
                for p in range(len(defending_players[xy_counter][0])):
                    lambda_def = defending_players[xy_counter][0][p]
                    tti_sigma = defending_players[xy_counter][1][p]
                    intercept = defending_players[xy_counter][2][p]
                    p_id = defending_players[xy_counter][3][p]
                    if p_id == 0:
                        break
                    
                    proba_intercept = 1/(1. + np.exp( -np.pi/np.sqrt(3.0)/tti_sigma * (T-intercept) ) )
                    dPPCFdT = (1-PPCFatt[k-1]-PPCFdef[k-1])*proba_intercept*lambda_def
                    assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                    players_def[p] += (dPPCFdT*int_dt)
                    PPCFdef[k] += players_def[p]
                
                ptot = PPCFdef[k]+PPCFatt[k]
                k += 1
            PPCFa[i,j] = PPCFatt[k-1]
            xy_counter += 1
    return PPCFa,PPCF_players_att

def calculate_pitch_control_at_target(target_position, attacking_players, defending_players, ball_start_pos, params):
    """ calculate_pitch_control_at_target
    
    Calculates the pitch control probability for the attacking and defending teams at a specified target position on the ball.
    
    Parameters
    -----------
        target_position: size 2 numpy array containing the (x,y) position of the position on the field to evaluate pitch control
        attacking_players: list of 'player' objects (see player class above) for the players on the attacking team (team in possession)
        defending_players: list of 'player' objects (see player class above) for the players on the defending team
        ball_start_pos: Current position of the ball (start position for a pass). If set to NaN, function will assume that the ball is already at the target position.
        params: Dictionary of model parameters (default model parameters can be generated using default_model_params() )
        
    Returrns
    -----------
        PPCFatt: Pitch control probability for the attacking team
        PPCFdef: Pitch control probability for the defending team ( 1-PPCFatt-PPCFdef <  params['model_converge_tol'] )
        attacking_dict: dictionary of each attacking player's contribution to the overall pitch control probability
        defending_dict: dictionary of each defending player's contribution to the overall pitch control probability
    """
    # calculate ball travel time from start position to end position.
    if ball_start_pos is None or any(np.isnan(ball_start_pos)): # assume that ball is already at location
        ball_travel_time = 0.0 
    else:
        # ball travel time is distance to target position from current ball position divided assumed average ball speed
        ball_travel_time = np.linalg.norm( target_position - ball_start_pos )/params['average_ball_speed']
    
    # first get arrival time of 'nearest' attacking player (nearest also dependent on current velocity)
    tau_min_att = np.nanmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )
    tau_min_def = np.nanmin( [p.simple_time_to_intercept(target_position ) for p in defending_players] )
    
    # check whether we actually need to solve equation 3
    if tau_min_att-max(ball_travel_time,tau_min_def) >= params['time_to_control_def']:
        # if defending team can arrive significantly before attacking team, no need to solve pitch control model
        closest_player = defending_players[np.argmin( [p.simple_time_to_intercept(target_position) for p in defending_players] )].id
        return 0., 1., {}, {closest_player: 1}
    elif tau_min_def-max(ball_travel_time,tau_min_att) >= params['time_to_control_att']:
        # if attacking team can arrive significantly before defending team, no need to solve pitch control model
        closest_player = attacking_players[np.argmin( [p.simple_time_to_intercept(target_position) for p in attacking_players] )].id
        return 1., 0., {closest_player: 1}, {}
    else: 
        # solve pitch control model by integrating equation 3 in Spearman et al.
        # first remove any player that is far (in time) from the target location
        attacking_players = [p for p in attacking_players if p.time_to_intercept-tau_min_att < params['time_to_control_att'] ]
        defending_players = [p for p in defending_players if p.time_to_intercept-tau_min_def < params['time_to_control_def'] ]
        # set up integration arrays
        dT_array = np.arange(ball_travel_time-params['int_dt'],ball_travel_time+params['max_int_time'],params['int_dt']) 
        PPCFatt = np.zeros_like( dT_array )
        PPCFdef = np.zeros_like( dT_array )
        
        attacking_dict = {}
        defending_dict = {}
        # integration equation 3 of Spearman 2018 until convergence or tolerance limit hit (see 'params')
        ptot = 0.0
        i = 1
        while 1-ptot>params['model_converge_tol'] and i<dT_array.size: 
            T = dT_array[i]
            for player in attacking_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_att
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid attacking player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFatt[i] += player.PPCF # add to sum over players in the attacking team (remembering array element is zero at the start of each integration iteration)
                attacking_dict[player.id] = player.PPCF
            for player in defending_players:
                # calculate ball control probablity for 'player' in time interval T+dt
                dPPCFdT = (1-PPCFatt[i-1]-PPCFdef[i-1])*player.probability_intercept_ball( T ) * player.lambda_def
                # make sure it's greater than zero
                assert dPPCFdT>=0, 'Invalid defending player probability (calculate_pitch_control_at_target)'
                player.PPCF += dPPCFdT*params['int_dt'] # total contribution from individual player
                PPCFdef[i] += player.PPCF # add to sum over players in the defending team
                defending_dict[player.id] = player.PPCF
            ptot = PPCFdef[i]+PPCFatt[i] # total pitch control probability 
            i += 1
        if i>=dT_array.size:
            print("Integration failed to converge: %1.3f" % (ptot) )
        return PPCFatt[i-1], PPCFdef[i-1], attacking_dict, defending_dict
    
def plot_pitchcontrol_for_frame( frame, PPCF, team, alpha = 0.7, include_player_velocities=False, annotate=[], field_dimen = (106.0,68)):
    """ plot_pitchcontrol_for_frame( frame, PPCF, team )
    
    Plots the pitch control surface at the instant of the event given by the event_id. Player and ball positions are overlaid.
    
    Parameters
    -----------
        frame: the row of the dataframe containing the frame to plot, eg df.iloc[100] for frame 100 
        PPCF: Pitch control surface (dimen (n_grid_cells_x,n_grid_cells_y) ) containing pitch control probability for the attcking team (as returned by the generate_pitch_control_for_event in Metrica_PitchControl)
        alpha: alpha (transparency) of player markers. Default is 0.7
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        annotate: Boolean variable that determines with player jersey numbers are added to the plot (default is False)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """        
    # plot frame and event
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    plot_frame( frame, figax=(fig,ax), team_colors=('green','red'), PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    
    # plot pitch control surface
    if team=='home':
        cmap = 'RdYlGn'
    else:
        cmap = 'RdYlGn_r'
    ax.imshow(np.flipud(PPCF), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)

    return fig,ax


def save_match_clip_pitchcontrol(df, frame_id_start, frame_id_end, fpath, game_info, params, attacking_team, fname='clip_test', figax=None, frames_per_second=25, team_colors=('green','red'), field_dimen = (106.0,68.0), include_player_velocities=True, PlayerMarkerSize=15, PlayerAlpha=0.7,annotate=[]):
    """ save_match_clip( frames, fpath, game_info, params, attacking_team )
    
    Generates a movie from Metrica tracking data, saving it in the 'fpath' directory with name 'fname'
    
    Parameters
    -----------
        frames: the rows in the dataframe to plot, eg df.iloc[20:120] for frames 20 to 120
        fpath: directory to save the movie
        fname: movie filename. Default is 'clip_test.mp4'
        fig,ax: Can be used to pass in the (fig,ax) objects of a previously generated pitch. Set to (fig,ax) to use an existing figure, or None (the default) to generate a new pitch plot,
        frames_per_second: frames per second to assume when generating the movie. Default is 25.
        team_colors: Tuple containing the team colors of the home & away team. Default is 'r' (red, home team) and 'b' (blue away team)
        field_dimen: tuple containing the length and width of the pitch in meters. Default is (106,68)
        include_player_velocities: Boolean variable that determines whether player velocities are also plotted (as quivers). Default is False
        PlayerMarkerSize: size of the individual player marlers. Default is 10
        PlayerAlpha: alpha (transparency) of player markers. Defaault is 0.7
        
    Returrns
    -----------
       fig,ax : figure and aixs objects (so that other data can be plotted onto the pitch)
    """
    frames = df.iloc[frame_id_start:frame_id_end]
    # plot pitch control surface
    if attacking_team=='home':
        cmap = 'RdYlGn'
    else:
        cmap = 'RdYlGn_r'
    
    # create football pitch
    if figax is None:
        fig,ax = plot_pitch(field_dimen=field_dimen)
    else:
        fig,ax = figax
    fig.set_tight_layout(True)
    im = plt.imshow(np.flipud(np.zeros((32,50))), extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),interpolation='spline36',vmin=0.0,vmax=1.0,cmap=cmap,alpha=0.5)
    
    # Generate movie
    print("Generating movie...",end='')
    def init_anim():
        pass
        
    def animate(i, begin_frame):
        row = frames.iloc[i]
        for figobj in figobjs:
            figobj.remove()
            figobjs.remove(figobj)
        
        #figobjs = [] # this is used to collect up all the axis objects so that they can be deleted after each iteration
        for team,color in zip( ['home', 'away'], team_colors) :
            jersey_values = []
            x_values = []
            y_values = []
            vx_values = []
            vy_values = []
            player_ids = np.unique( [ int(c.split('.')[1]) for c in row.keys() if c[-2:] == '.x' and team+'_team' in c] )

            for p in player_ids:
                j_number = str(p)
                if np.isnan(row[team+'_team_aux.'+str(j_number)+'.x']) == False:
                    jersey_values.append(j_number)
                    x_values.append(row[team+'_team_aux.'+j_number+'.x'])
                    y_values.append(row[team+'_team_aux.'+j_number+'.y'])
                    vx_values.append(row[team+'_team_aux.'+j_number+'.vx'])
                    vy_values.append(row[team+'_team_aux.'+j_number+'.vy'])
                
            objs, = ax.plot( x_values, y_values, color=color, marker='o', linestyle='None', markeredgecolor='black', markersize=PlayerMarkerSize, alpha=PlayerAlpha ) # plot player positions
            figobjs.append(objs)
            if include_player_velocities:
                objs = ax.quiver( x_values, y_values, vx_values, vy_values, color=color, scale_units='inches', scale=10.,width=0.0015,headlength=5,headwidth=3,alpha=PlayerAlpha)
                figobjs.append(objs)
            if team in annotate:
                for j in range(len(jersey_values)):
                    objs = ax.annotate(int(jersey_values[j]),(x_values[j]+1.2,y_values[j]-1),size='x-large')
                    figobjs.append(objs)
            
        # plot ball
        if row.ball['x'] != None:
            objs, = ax.plot( row.ball['x'], row.ball['y'], 'ko', markersize=10, alpha=1.0, linewidth=0)
            figobjs.append(objs)
        # include match time at the top
        frame_minute =  int( (row['match_time']/1000)/60. )
        frame_second =  ( (row['match_time']/1000.)/60. - frame_minute ) * 60
        if row['phase'] == 2:
            frame_minute += 45
        timestring = "%d:%1.2f" % ( frame_minute, frame_second  )
        objs = ax.text(-2.5,field_dimen[1]/2.+1., timestring, fontsize=14 )
        figobjs.append(objs)
        PPCF,xgrid,ygrid,_ = generate_pitch_control_for_frame(df, i+begin_frame, params, game_info, attacking_team, field_dimen = field_dimen, n_grid_cells_x = 50)
        im.set_array(np.flipud(PPCF))
        return [im]
        
            
    figobjs = []
    anim = animation.FuncAnimation(fig, animate, fargs=[frames.iloc[0].name], frames=len(frames), interval=40, init_func=init_anim)
    FFwriter = animation.FFMpegWriter(fps=25)
    anim.save(fname+'.mp4', writer = FFwriter)

    print("done")
    plt.clf()
    plt.close(fig)
    
# Pass impact model by Twelve Football
def pass_impact_twelve(coord_origin,coord_destination,attack_direction,cross,throughBall,pullBack,chanceCreated,flickOn):

    ''' Variables x and y take values from 0 to 100 '''

    if attack_direction=='right':
        x1 = coord_origin[0]
        x2 = coord_destination[0]

        y1 = coord_origin[1]
        y2 = coord_destination[1]

        adj_y1 = y1 if y1<50 else 100-y1
        adj_y2 = y2 if y2<50 else 100-y2

    else:
        x1 = 100-coord_origin[0]
        x2 = 100-coord_destination[0]

        y1 = 100-coord_origin[1]
        y2 = 100-coord_destination[1]

        adj_y1 = y1 if y1<50 else 100-y1
        adj_y2 = y2 if y2<50 else 100-y2

    #coefficient = -3.720748+6.037517e-03*x1+3.584304e-02*x2+2.617176e-02*adj_y2-6.156030e-04*x1*adj_y1+5.304036e-04*math.pow(adj_y1, 2)-3.027166e-04*math.pow(x1, 2)-7.325353e-06*math.pow(adj_y1, 3)+4.716508e-06*math.pow(x1, 3)-4.951233e-04*x2*adj_y2-4.466221e-04*math.pow(x2, 2)+1.128160e-04*math.pow(adj_y2, 2)-4.959944e-06*math.pow(adj_y2, 3)+4.849165e-06*math.pow(x2, 3)+2.196781e-04*x1*x2-4.024221e-04*adj_y1*adj_y2+1.057939e-05*x1*adj_y1*x2+4.241092e-06*x1*x2*adj_y2-8.232459e-08*x1*adj_y1*x2*adj_y2-6.654025e-06*math.pow(x1, 2)*x2+4.668466e-06*math.pow(x2, 2)*adj_y2+7.636041e-06*math.pow(adj_y2, 2)*adj_y1
    #logistic = (math.exp(coefficient)/(1+math.exp(coefficient)))
    #linear = 7.628156e-03*x1+7.996155e-03*x2+1.445358e-03*adj_y2-1.368979*cross+8.410532e-01*throughBall+7.921517e-02*pullBack-9.274986e-02*chanceCreated-7.581955e-05*x1*adj_y1-4.716755e-05*math.pow(x1, 2)-9.534056e-05*x2*adj_y2-6.851437e-05*math.pow(x2, 2)+7.821691e-07*math.pow(x2, 3)-2.111737e-04*x1*x2+5.654900e-05*adj_y1*adj_y2+5.308242e-07*x1*adj_y1*x2+2.328050e-07*x1*x2*adj_y2+1.423698e-02*cross*x2-5.765683e-03*throughBall*x2+3.073823e-03*flickOn*x2-3.470719e-03*flickOn*x1+1.094886e-01*chanceCreated*cross+7.758500e-02*chanceCreated*flickOn-9.178206e-02*chanceCreated*throughBall+4.158375e-07*math.pow(x1, 2)*x2+7.818592e-07*math.pow(x2, 2)*x1+3.818770e-07*math.pow(x1, 2)*adj_y1+8.122093e-07*math.pow(x2, 2)*adj_y2-4.910344e-07*math.pow(adj_y2, 2)*adj_y1
    
    coefficient = -3.720748 + 6.037517e-03 * x1 + 3.584304e-02 * x2 + 2.617176e-02 * adj_y2 - 6.156030e-04 * x1 * adj_y1 + 5.304036e-04 * math.pow(adj_y1, 2) - 3.027166e-04 * math.pow(x1, 2) - 7.325353e-06 * math.pow(adj_y1,3) + 4.716508e-06 * math.pow(x1, 3) - 4.951233e-04 * x2 * adj_y2 - 4.466221e-04 * math.pow(x2, 2) + 1.128160e-04 * math.pow(adj_y2,2) - 4.959944e-06 * math.pow(adj_y2, 3) + 4.849165e-06 * math.pow(x2,3) + 2.196781e-04 * x1 * x2 - 4.024221e-04 * adj_y1 * adj_y2 + 1.057939e-05 * x1 * adj_y1 * x2 + 4.241092e-06 * x1 * x2 * adj_y2 - 8.232459e-08 * x1 * adj_y1 * x2 * adj_y2 - 6.654025e-06 * math.pow(x1, 2) * x2 + 4.668466e-06 * math.pow(x2, 2) * adj_y2 + 7.636041e-06 * math.pow(adj_y2,2) * adj_y1
    logistic = (math.exp(coefficient) / (1 + math.exp(coefficient)))
    linear = 7.628156e-03 * x1 + 7.996155e-03 * x2 + 1.445358e-03 * adj_y2 - 1.368979 * cross + 8.410532e-01 * throughBall + 7.921517e-02 * pullBack - 9.274986e-02 * chanceCreated - 7.581955e-05 * x1 * adj_y1 - 4.716755e-05 * math.pow(x1, 2) - 9.534056e-05 * x2 * adj_y2 - 6.851437e-05 * math.pow(x2, 2) + 7.821691e-07 * math.pow(x2,3) - 2.111737e-04 * x1 * x2 + 5.654900e-05 * adj_y1 * adj_y2 + 5.308242e-07 * x1 * adj_y1 * x2 + 2.328050e-07 * x1 * x2 * adj_y2 + 1.423698e-02 * cross * x2 - 5.765683e-03 * throughBall * x2 + 3.073823e-03 * flickOn * x2 - 3.470719e-03 * flickOn * x1 + 1.094886e-01 * chanceCreated * cross + 7.758500e-02 * chanceCreated * flickOn - 9.178206e-02 * chanceCreated * throughBall + 4.158375e-07 * math.pow(x1, 2) * x2 + 7.818592e-07 * math.pow(x2, 2) * x1 + 3.818770e-07 * math.pow(x1,2) * adj_y1 + 8.122093e-07 * math.pow(x2, 2) * adj_y2 - 4.910344e-07 * math.pow(adj_y2, 2) * adj_y1
    
    return logistic*linear

def generate_pass_impact_for_frame(df, frame_id_start, game_info, attack_team, model = 'laurie', field_dimen = (106.,68.,), n_grid_cells_x = 50):
    """ generate_pass_impact_for_frame( frame, game_info, attack_team )
    
    Generates a pass impact surface for given frame based on Twelve Football's model
    
    Parameters
    -----------
        frame: row of DF for the desired frame
        game_info: metadata of the Signality tracking data
        attack_team: 'home' or 'away' depending on which team is attacking
        
    Returrns
    -----------
       PI: Pass Impact surface (dimen (n_grid_cells_x,n_grid_cells_y) )
    """
    frame = df.iloc[frame_id_start]
    attack_direction = find_attacking_direction(frame, game_info, attack_team)
    
    if model == 'laurie':
        EPV = np.loadtxt('EPV_grid.csv', delimiter=',')
        if attack_direction == 'left':
            EPV = np.fliplr(EPV)
        return EPV
    
    
    # break the pitch down into a grid
    n_grid_cells_y = int(n_grid_cells_x*field_dimen[1]/field_dimen[0])
    dx = field_dimen[0]/n_grid_cells_x
    dy = field_dimen[1]/n_grid_cells_y
    xgrid = (np.arange(n_grid_cells_x)*dx + dx/2.) / 106 * 100
    ygrid = (np.arange(n_grid_cells_y)*dy + dy/2.) / 68 * 100
    
    # initialise PI grid
    PI = np.zeros( shape = (len(ygrid), len(xgrid)) )

    if frame['ball.position'] == None:
        phase = frame.phase
        ball_df = df[(df.index<frame_id_start) & (df['ball.position'].isna() == False) & (df.phase==phase)].copy()
        if len(ball_df) > 0:
            ball_x = ball_df.iloc[-1]['ball.position'][0]
            ball_y = ball_df.iloc[-1]['ball.position'][1]
        else:
            ball_x = 0
            ball_y = 0
    else:
        ball_x = frame['ball.position'][0]
        ball_y = frame['ball.position'][1]
    origin_position = np.array([(ball_x+106/2)/106*100,(ball_y+68/2)/68*100])
    
    
    for i in range(len(ygrid)):
        for j in range(len(xgrid)):
            target_position = np.array( [xgrid[j], ygrid[i]] )
            PI[i][j] = pass_impact_twelve(origin_position,target_position,attack_direction,0,0,0,0,0)
    return PI



def plot_pcpi_for_frame( frame, PPCF, PI, cmap, alpha = 0.7, include_player_velocities=True, annotate=[], field_dimen = (106.0,68), vmax=0.7, team_colors = ('green', 'red')):
    """ plot_pcpi_for_frame( frame, PPCF, PI, cmap )
    
    Plots the Pitch Control * Pass Impact pitch
    
    Parameters
    -----------
        frame: row of DF for the desired frame
        PPCF: Pitch control probability matrix of the frame for the attacking team
        PI: Pass impact matrix of the frame for the attacking team
        
    Returrns
    -----------
       fig, ax: figure and axis objects
    """
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    plot_frame( frame, figax=(fig,ax), team_colors=team_colors, PlayerAlpha=alpha, include_player_velocities=include_player_velocities, annotate=annotate )
    
    PIxPPCF = PI*PPCF
            
    ax.imshow(np.flipud(PIxPPCF),extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),vmin=0,vmax=vmax,interpolation='spline36',cmap=cmap,alpha=0.7)
    
    return fig,ax

def plot_PI(PI,field_dimen=(106.0,68)):
    """ plot_pcpi_for_frame( PI )
    
    Plots the pass impact surface
    
    Parameters
    -----------
        PI: Pass impact matrix 
        
    Returrns
    -----------
       fig, ax: figure and axis objects
    """
    # plot a pitch
    fig,ax = plot_pitch(field_color='white', field_dimen = field_dimen)
    # overlap the EPV surface
    ax.imshow(PI, extent=(-field_dimen[0]/2., field_dimen[0]/2., -field_dimen[1]/2., field_dimen[1]/2.),vmin=0.0,vmax=0.7,cmap='Blues',alpha=0.6)
    return fig, ax
    
def find_attacking_direction(frame, game_info, attack_team):
    """ find_attacking_direction( frame, game_info, attack_team )
    
    Finds the attacking direction for the given team
    
    Parameters
    -----------
        frame: row of DF for the desired frame
        game_info: metadata of the Signality tracking data
        attack_team: desired team
        
    Returrns
    -----------
       attack_direction: the attacking direction for team attack_team
    """
    home_direction = game_info['home_attack_direction_1']
    if attack_team == 'home':
        attack_direction = home_direction
    else:
        if home_direction == 'right':
            attack_direction = 'left'
        else:
            attack_direction = 'right'
    
    if frame.phase == 2:
        if attack_direction == 'right':
            attack_direction = 'left'
        else:
            attack_direction = 'right'
            
    return attack_direction
    
    
def pcpi_contribution(frame, game_info, PPCF_players, PI, team, field_dimen = (106.0,68.0)):
    """ pcpi_contribution( frame, game_info, PPCF_players, PI, team )
    
    Finds the maximum PC*PI contribution for each attacking player in final third
    
    Parameters
    -----------
        frame: row of DF for the desired frame
        game_info: metadata of the Signality tracking data
        PPCF_players: the pitch control probability matrix for each player in a dictionary, where the key is the jersey number
                      and the value is the matrix
        PI: Pass impact matrix of the frame for the attacking team
        team: the attacking team
        
    Returrns
    -----------
       contributions: the maximum contribution of each attacking player in final third in terms of probabilities
    """
    final_third = (field_dimen[0]/3)*2-field_dimen[0]/2
    direction_multiplier = 1
    if find_attacking_direction(frame, game_info, team) == 'left':
        direction_multiplier = -1
    
    contributions = {}
    player_ids = np.unique( [ int(c.split('.')[1]) for c in frame.keys() if c[-2:] == '.x' and team+'_team' in c] )

    
    for j_number in player_ids:
        if np.isnan(frame[team+'_team_aux.'+str(j_number)+'.x']):
            continue
        if j_number == -1:
            continue
        if direction_multiplier * frame[team+'_team_aux.'+str(j_number)+'.x'] >= final_third:
            contributions[j_number] = np.amax(PPCF_players[j_number]*PI)
    return contributions
    
def maximum_box_pcpi(frame, game_info, PPCF, PI, team, xgrid, ygrid, field_dimen = (106.0,68.0)):
    att_direction = find_attacking_direction(frame, game_info, team)
    if att_direction == 'right':
        x_penalty = np.where(xgrid >= (53-16))
    else:
        x_penalty = np.where(xgrid <= -1*(53-16))
    y_penalty = np.where((ygrid >= -20) & (ygrid <= 20))
    
    PCPI = PPCF*PI
    PCPI_values = []
    
    for i in y_penalty[0]:
        for j in x_penalty[0]:
            PCPI_values.append(PCPI[i][j])
    max_PCPI = np.array(PCPI_values).max()
    
    contributions = {}
    
    player_ids = np.unique( [ int(c.split('.')[1]) for c in frame.keys() if c[-2:] == '.x' and team+'_team' in c] )

    
    for j_number in player_ids:
        if np.isnan(frame[team+'_team_aux.'+str(j_number)+'.x']):
            continue
        if j_number == -1:
            continue
        contributions[j_number] = max_PCPI
        
    return contributions
    
def pcpi_generation(df, frame_id_start, team, game_info, params, field_dimen = (106.0,68.0), downsample_factor=1, cached_contributions={}):
    # Params
    frames_ahead = int(75/downsample_factor) # 3 sec in future, dependent if df is downsampled
    meters_away = 4 # meters a player must've ran away from origin position to be considered for metric
    radius_space = 6 # radius of player in origin position to consider space generated
    
    final_third = (field_dimen[0]/3)*2-field_dimen[0]/2
    frame_start = df.iloc[frame_id_start]
    direction_multiplier = 1
    if find_attacking_direction(frame_start, game_info, team) == 'left':
        direction_multiplier = -1
        
    origin_positions = {}    
    player_ids = np.unique( [ int(c.split('.')[1]) for c in frame_start.keys() if c[-2:] == '.x' and team+'_team' in c] )
    
    for j_number in player_ids:
        if np.isnan(frame_start[team+'_team_aux.'+str(j_number)+'.x']):
            continue
        if j_number == -1:
            continue
        if direction_multiplier * frame_start[team+'_team_aux.'+str(j_number)+'.x'] >= final_third:
            origin_positions[j_number] = np.array([frame_start[team+'_team_aux.'+str(j_number)+'.x'], frame_start[team+'_team_aux.'+str(j_number)+'.y']])
            
    if len(origin_positions) == 0 or frame_start['possession_team'] != team or frame_start['state'] == 'paused':
        return {}, {}, cached_contributions
    current_phase = frame_start['phase']
    
    if frame_id_start+frames_ahead > len(df):
        frames_ahead = len(df)
    elif current_phase == 1 and df.iloc[frame_id_start+frames_ahead]['phase'] != current_phase:
        frames_ahead = df[df.phase==current_phase].iloc[-1].name - frame_id_start
        
    frames_check = {}
    players_check = set()
    for i,r in df.iloc[frame_id_start:frame_id_start+frames_ahead].iterrows():
        if r['ball.position'] == None:
            continue
        if r['possession_team'] != team:
            continue
        if r['state'] == 'paused':
            continue
            
        player_ids = np.unique( [ int(c.split('.')[1]) for c in r.keys() if c[-2:] == '.x' and team+'_team' in c] )
    
        for j_number in player_ids:
            if np.isnan(r[team+'_team_aux.'+str(j_number)+'.x']):
                continue
            if j_number == -1:
                continue
            if j_number in origin_positions:
                players_check.add(j_number)
                current_pos = np.array([r[team+'_team_aux.'+str(j_number)+'.x'], r[team+'_team_aux.'+str(j_number)+'.y']])
                if np.linalg.norm(current_pos-origin_positions[j_number]) >= meters_away:
                    if i in frames_check:
                        frames_check[i].append(j_number)
                    else:
                        frames_check[i] = [j_number]
                        
    players_origin_pcpi = {}
    
    if frame_id_start in cached_contributions:
        contributions_origin = cached_contributions[frame_id_start]
    else:
        PPCF,xgrid,ygrid,PPCF_players = generate_pitch_control_for_frame(df, frame_id_start, params, game_info, team, field_dimen = (106.,68.,), n_grid_cells_x = 50,x_threshold=0)
        PI = generate_pass_impact_for_frame(df, frame_id_start, game_info, team)
        contributions_origin = pcpi_contribution(frame_start, game_info, PPCF_players, PI, team)
        cached_contributions[frame_id_start] = contributions_origin
    
    for p in players_check:
        pcpi_origin = 0
        player_ids = np.unique( [ int(c.split('.')[1]) for c in frame_start.keys() if c[-2:] == '.x' and team+'_team' in c] )
    
        for j_number in player_ids:
            if np.isnan(frame_start[team+'_team_aux.'+str(j_number)+'.x']):
                continue
            if j_number == -1:
                continue
            if p != j_number:
                current_pos = np.array([frame_start[team+'_team_aux.'+str(j_number)+'.x'], frame_start[team+'_team_aux.'+str(j_number)+'.y']])
                if np.linalg.norm(current_pos-origin_positions[p]) <= radius_space and j_number in contributions_origin:
                    pcpi_origin += contributions_origin[j_number]
        players_origin_pcpi[p] = pcpi_origin
        
    players_final_pcpi = {}
    
    for frame in frames_check:
        pcpi_origin_dict = {}
        found_players = 0
        frame_current = df.iloc[frame]
        
        for p_check in frames_check[frame]:
            players_in_radius = []
            player_ids = np.unique( [ int(c.split('.')[1]) for c in frame_current.keys() if c[-2:] == '.x' and team+'_team' in c] )
    
            for j_number in player_ids:
                if np.isnan(frame_current[team+'_team_aux.'+str(j_number)+'.x']):
                    continue
                if j_number == -1:
                    continue
                if j_number != p_check:
                    current_pos = np.array([frame_current[team+'_team_aux.'+str(j_number)+'.x'], frame_current[team+'_team_aux.'+str(j_number)+'.y']])
                    if np.linalg.norm(current_pos-origin_positions[p_check]) <= radius_space and direction_multiplier * current_pos[0] >= final_third:
                        players_in_radius.append(j_number)
                        found_players = 1
            pcpi_origin_dict[p_check] = players_in_radius
        
        if found_players == 1:
            if frame in cached_contributions:
                contributions = cached_contributions[frame]
            else:
                PPCF,xgrid,ygrid,PPCF_players = generate_pitch_control_for_frame(df, frame, params, game_info, team, field_dimen = (106.,68.,), n_grid_cells_x = 50,x_threshold=0)
                PI = generate_pass_impact_for_frame(df, frame, game_info, team)
                contributions = pcpi_contribution(frame_current, game_info, PPCF_players, PI, team)
                cached_contributions[frame] = contributions
            
            for p_check in pcpi_origin_dict:
                pcpi_value = 0
                for p_radius in pcpi_origin_dict[p_check]:
                    pcpi_value += contributions[p_radius]
                    
                if pcpi_value > players_origin_pcpi[p_check]:
                    if p_check in players_final_pcpi:
                        if pcpi_value > players_final_pcpi[p_check]:
                            players_final_pcpi[p_check] = pcpi_value
                    else:
                        players_final_pcpi[p_check] = pcpi_value
                
    return players_final_pcpi, contributions_origin, cached_contributions
    