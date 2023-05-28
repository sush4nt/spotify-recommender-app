import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as components
import time
from model import *

def play_recomm():
    """
    generates recommendations based on playlist playlist's genres
    1. If there are any previous inputs, deletes the state
    2. If the # of new tracks from the playlist > 200, then update the dataset
    3. Create recommendations
    """
    # delete any previous state
    if 'rs' in st.session_state:
        del st.session_state.rs, st.session_state.err
    # if new tracks present, add and save them
    try:
        if len(pd.read_csv('Data/new_tracks.csv')) >= 200:
            with st.spinner('Updating the dataset...'):
                x=update_dataset()
                st.success('{} New tracks were added to the dataset.'.format(x))
    except:
        st.error("The dataset update failed.")
    # get recommendations
    with st.spinner('Getting Recommendations...'):
        res, err = playlist_model(st.session_state.p_url,
                                    st.session_state.model,
                                    st.session_state.genre,
                                    st.session_state.artist
                                    )
        st.session_state.rs = res
        st.session_state.err = err
    if len(st.session_state.rs)>=1:
        if st.session_state.model == 'Model 1' or st.session_state.model == 'Model 2':
            st.success('Go to the results page to view the top {} recommendations'.format(len(st.session_state.rs)))
        else:
            st.success('Go to the results page to view the recommendations')
    else:
        st.error('Model failed. Check the log for more information.')   

def art_recomm():
    """
    generates recommendations based on playlist's artists' top tracks
    """
    if 'rs' in st.session_state:
        del st.session_state.rs,st.session_state.err
    with st.spinner('Getting Recommendations...'):
        res,err = top_tracks(st.session_state.a_url, st.session_state.rg)
        st.session_state.rs=res
        st.session_state.err=err
    if len(st.session_state.rs)>=1:
        st.success("Go to the results page to view the artist's top tracks")
    else:
        st.error('Model failed. Check the log for more information.')

def song_recomm():
    """
    generates recommendations based on songs in the playlist's songs
    """
    if 'rs' in st.session_state:
        del st.session_state.rs,st.session_state.err
    with st.spinner('Getting Recommendations...'):
        res,err = song_model(st.session_state.s_url,
                                st.session_state.model,
                                st.session_state.genre,
                                st.session_state.artist
                                )
        st.session_state.rs=res
        st.session_state.err=err
    if len(st.session_state.rs)>=1:
        if st.session_state.model == 'Model 1' or st.session_state.model == 'Model 2':
            st.success('Go to the results page to view the top {} recommendations'.format(len(st.session_state.rs)))
        else:
            st.success('Go to the results page to view the  Spotify recommendations')
    else:
        st.error('Model failed. Check the log for more information.')

def playlist_page():
    """
    display iframe based on playlist URL
    """
    st.subheader("User Playlist")
    st.markdown('---')
    playlist_uri = (st.session_state.playlist_url).split('/')[-1].split('?')[0]
    uri_link = 'https://open.spotify.com/embed/playlist/' + playlist_uri
    components.iframe(uri_link, height=300)

def song_page():
    """
    display iframe based on song URL
    """
    st.subheader("User Song")
    st.markdown('---')
    song_uri = (st.session_state.song_url).split('/')[-1].split('?')[0]
    uri_link = 'https://open.spotify.com/embed/track/' + song_uri
    components.iframe(uri_link, height=200)

def artist_page():
    """
    display iframe based on artist URL
    """
    st.subheader("User Artist")
    st.markdown('---')
    artist_uri = (st.session_state.artist_url).split('/')[-1].split('?')[0]
    uri_link = 'https://open.spotify.com/embed/artist/' + artist_uri
    components.iframe(uri_link, height=300)


def spr_sidebar():
    """
    creates a sidebar with 4 pages:
        1. Home
        2. Result
        3. About
        4. Log
    """
    menu=option_menu(
        menu_title=None,
        options=['Home','Result','About','Log'],
        icons=['house','book','info-square','terminal'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal'
    )
    if menu=='Home':
        st.session_state.app_mode = 'Home'
    elif menu=='Result':
        st.session_state.app_mode = 'Result'
    elif menu=='About':
        st.session_state.app_mode = 'About'
    elif menu=='Log':
        st.session_state.app_mode = 'Log'

# radio "Feature"
if 'radio' not in st.session_state:
    st.session_state.feature="Playlist"
def update_radio0():
    st.session_state.feature=st.session_state.radio

# radio "Model"
if 'model' not in st.session_state:
    st.session_state.model = 'Model 1'
def update_radio2():
    st.session_state.model = st.session_state.radio2
if 'model2' not in st.session_state:
    st.session_state.model2 = 'Spotify Model'
def update_radio1():
    st.session_state.model2 = st.session_state.radio1

# selectBox "Region"
if 'region' not in st.session_state:
    st.session_state.rg="US"
def update_region():
    st.session_state.rg=st.session_state.region

# selectBox "Genre"
if 'genre' not in st.session_state:
    st.session_state.genre=3
def update_num_genre():
    st.session_state.genre=st.session_state.num_genre

# selectBox "Artist"
if 'artist' not in st.session_state:
    st.session_state.artist=5
def update_same_art():
    st.session_state.artist=st.session_state.same_art

# display URLs
if 'p_url' not in st.session_state:
    st.session_state.p_url = 'Example: https://open.spotify.com/playlist/37i9dQZF1DX0E9XMGembJo?si=dcd0716f53f64527'
if 's_url' not in st.session_state:
    st.session_state.s_url = 'Example: https://open.spotify.com/track/7165YDcOpr9yEypbdpU6fa?si=873f2e37b656495d'
if 'a_url' not in st.session_state:
    st.session_state.a_url = 'Example: https://open.spotify.com/artist/4Ai0pGz6GhQavjzaRhPTvz?si=qg1gMiCWSeWWrS4DM8Y4qw'

# updating URLS
def update_artist_url():
    st.session_state.a_url = st.session_state.artist_url
def update_playlist_url():
    st.session_state.p_url = st.session_state.playlist_url
def update_song_url():
    st.session_state.s_url = st.session_state.song_url

# define home page
def home_page():
    """
    configuring `Home` page
    """
    st.session_state.radio=st.session_state.feature
    st.session_state.radio2=st.session_state.model
    st.session_state.num_genre=st.session_state.genre
    st.session_state.same_art=st.session_state.artist
    st.session_state.region=st.session_state.rg

    # define title
    st.title('Spotify Recommendation App')
    # divide into 3 columns for 
    # 1. `Feature`
    # 2. `Model`
    # 3.  Dropdowns
    col1, col2, col3 = st.columns([2,2,3])

    # define `Feature`
    radio = col1.radio("Feature", 
                        options=(
                                "Playlist",
                                "Song",
                                "Artist Top Tracks"
                                ),
                        key='radio',
                        on_change=update_radio0)
    # change dropdowns for artist tracks
    if radio == "Artist Top Tracks":
        radio1 = col2.radio("Model",
                            options=["Spotify Model"],
                            key='radio1',
                            on_change=update_radio1
                            )
        # select box for region
        region=col3.selectbox("Please Choose Region",
                            index=58, # index for `US`
                            key='region',
                            on_change = update_region,
                            options=('AD', 'AR', 'AU', 'AT', 'BE', 'BO', 'BR', 
                                    'BG', 'CA', 'CL', 'CO', 'CR', 'CY', 'CZ', 
                                    'DK', 'DO', 'EC', 'SV', 'EE', 'FI', 'FR', 
                                    'DE', 'GR', 'GT', 'HN', 'HK', 'HU', 'IS', 
                                    'ID', 'IE', 'IT', 'JP', 'LV', 'LI', 'LT', 
                                    'LU', 'MY', 'MT', 'MX', 'MC', 'NL', 'NZ', 
                                    'NI', 'NO', 'PA', 'PY', 'PE', 'PH', 'PL', 
                                    'PT', 'SG', 'ES', 'SK', 'SE', 'CH', 'TW', 
                                    'TR', 'GB', 'US', 'UY'
                                    )
                            )
    elif radio == "Playlist" or radio == "Song" :
        radio2 = col2.radio("Model",
                            options=("Model 1", "Model 2", "Spotify Model"),
                            key='radio2',
                            on_change=update_radio2
                            )
        if st.session_state.radio2 == "Model 1" or st.session_state.radio2 == "Model 2":
            num_genre = col3.selectbox("Choose a number of genres to focus on",
                                        options=(1,2,3,4,5,6,7),
                                        index=2,
                                        key='num_genre',
                                        on_change=update_num_genre
                                        )
            same_art = col3.selectbox("How many recommendations by the same artist?",
                                        options=(1,2,3,4,5,7,10,15),
                                        index=3,
                                        key='same_art',
                                        on_change=update_same_art
                                        )
    st.markdown("<br>", unsafe_allow_html=True)
    # WHAT HAPPENS WHEN WE WANT RECOMMENDATIONS FOR PLAYLIST
    if radio == "Playlist" :
        st.session_state.playlist_url = st.session_state.p_url
        Url = st.text_input(label="Playlist URL",
                            key='playlist_url',
                            on_change=update_playlist_url
                            )
        # display playlist frame based on URL
        playlist_page()
        state = st.button('Get Recommendations')
        with st.expander("Here's how to find any Playlist URL in Spotify"):
            st.write(""" 
                - Search for Playlist on the Spotify app
                - Right Click on the Playlist you like
                - Click "Share"
                - Choose "Copy link to playlist"
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            st.image('spotify_get_playlist_url.png')
        # if clicked; get recommendations
        if state:
            play_recomm()
    # WHAT HAPPENS WHEN WE WANT RECOMMENDATIONS FOR SONG
    elif radio == "Song" :
        st.session_state.song_url = st.session_state.s_url
        Url = st.text_input(label="Song URL",
                            key='song_url',
                            on_change=update_song_url
                            )
        # display song frame based on URL
        song_page()
        state = st.button('Get Recommendations')
        with st.expander("Here's how to find any song URL in Spotify"):
            st.write(""" 
                - Search for Song on the Spotify app
                - Right Click on the Song you like
                - Click "Share"
                - Choose "Copy link to Song"
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            st.image('spotify_get_song_url.png')
        if state:
            song_recomm()
    # WHAT HAPPENS WHEN WE WANT RECOMMENDATIONS ARTIST'S TOP TRACKS
    elif radio == "Artist Top Tracks" :
        st.session_state.artist_url = st.session_state.a_url
        Url = st.text_input(label="Artist URL",
                            key='artist_url',
                            on_change=update_artist_url
                            )
        # display artist frame based on URL
        artist_page()
        state = st.button('Get Recommendations')
        with st.expander("Here's how to find any artist URL in Spotify"):
            st.write(""" 
                - Search for Artist on the Spotify app
                - Right Click on the Artist you like
                - Click "Share"
                - Choose "Copy link to Artist"
            """)
            st.markdown("<br>", unsafe_allow_html=True)
            st.image('spotify_get_artist_url.png')
        if state:
            art_recomm()

# define results page
def result_page():
    """
    configure results page
    """
    if 'rs' not in st.session_state:
        st.error('Please select a model on the `Home` page and run `Get Recommendations`')
    else:
        st.success('Top {} recommendations'.format(len(st.session_state.rs)))
        i=0
        for uri in st.session_state.rs:
            uri_link = "https://open.spotify.com/embed/track/" + uri + "?utm_source=generator&theme=0"
            components.iframe(uri_link, height=80)
            i+=1
            if i%5==0:
                time.sleep(1)

# define log page
def log_page():
    """
    configure logs page
    """
    log=st.checkbox('Display Output', 
                    value=True, 
                    key='display_output'
                    )
    if log == True:
        if 'err' in st.session_state:
            st.write(st.session_state.err)
    with open('Data/streamlit.csv') as f:
        st.download_button('Download Dataset', 
                            data=f,
                            file_name='streamlit.csv'
                        )

# define about page
def about_page():
    st.header('Development')
    """
    Check out the [repository](https://github.com/sush4nt) for the source code.
    """
    st.subheader('Spotify Million Playlist Dataset')
    """
    For this project, I'm using the Million Playlist Dataset, which, as its name implies, consists of one million playlists.
    contains a number of songs, and some metadata is included as well, such as the name of the playlist, duration, number of songs, number of artists, etc.
    """

    """
    It is created by sampling playlists from the billions of playlists that Spotify users have created over the years. 
    Playlists that meet the following criteria were selected at random:
    - Created by a user that resides in the United States and is at least 13 years old
    - Was a public playlist at the time the MPD was generated
    - Contains at least 5 tracks
    - Contains no more than 250 tracks
    - Contains at least 3 unique artists
    - Contains at least 2 unique albums
    - Has no local tracks (local tracks are non-Spotify tracks that a user has on their local device
    - Has at least one follower (not including the creator
    - Was created after January 1, 2010 and before December 1, 2017
    - Does not have an offensive title
    - Does not have an adult-oriented title if the playlist was created by a user under 18 years of age

    Information about the Dataset [here](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge)
    """
    st.subheader('Audio Features Explanation')
    """
    | Variable | Description |
    | :----: | :---: |
    | Acousticness | A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic. |
    | Danceability | Danceability describes how suitable a track is for dancing based on a combination of musical elements including tempo, rhythm stability, beat strength, and overall regularity. A value of 0.0 is least danceable and 1.0 is most danceable. |
    | Energy | Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity. Typically, energetic tracks feel fast, loud, and noisy. For example, death metal has high energy, while a Bach prelude scores low on the scale. Perceptual features contributing to this attribute include dynamic range, perceived loudness, timbre, onset rate, and general entropy. |
    | Instrumentalness | Predicts whether a track contains no vocals. "Ooh" and "aah" sounds are treated as instrumental in this context. Rap or spoken word tracks are clearly "vocal". The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content. Values above 0.5 are intended to represent instrumental tracks, but confidence is higher as the value approaches 1.0. |
    | Key | The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1. |
    | Liveness | Detects the presence of an audience in the recording. Higher liveness values represent an increased probability that the track was performed live. A value above 0.8 provides strong likelihood that the track is live. |
    | Loudness | The overall loudness of a track in decibels (dB). Loudness values are averaged across the entire track and are useful for comparing relative loudness of tracks. Loudness is the quality of a sound that is the primary psychological correlate of physical strength (amplitude). Values typically range between -60 and 0 db. |
    | Mode | Mode indicates the modality (major or minor) of a track, the type of scale from which its melodic content is derived. Major is represented by 1 and minor is 0. |
    | Speechiness | Speechiness detects the presence of spoken words in a track. The more exclusively speech-like the recording (e.g. talk show, audio book, poetry), the closer to 1.0 the attribute value. Values above 0.66 describe tracks that are probably made entirely of spoken words. Values between 0.33 and 0.66 describe tracks that may contain both music and speech, either in sections or layered, including such cases as rap music. Values below 0.33 most likely represent music and other non-speech-like tracks. |
    | Tempo | The overall estimated tempo of a track in beats per minute (BPM). In musical terminology, tempo is the speed or pace of a given piece and derives directly from the average beat duration. |
    | Time Signature | An estimated time signature. The time signature (meter) is a notational convention to specify how many beats are in each bar (or measure). The time signature ranges from 3 to 7 indicating time signatures of "3/4", to "7/4". |
    | Valence | A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track. Tracks with high valence sound more positive (e.g. happy, cheerful, euphoric), while tracks with low valence sound more negative (e.g. sad, depressed, angry). |
    
    Information about features: [here](https://developer.spotify.com/documentation/web-api/reference/#/operations/get-audio-features)
    """

def main():
    spr_sidebar()        
    if st.session_state.app_mode == 'Home':
        home_page()
    if st.session_state.app_mode == 'Result':
        result_page()
    if st.session_state.app_mode == 'About' :
        about_page()
    if st.session_state.app_mode == 'Log':
        log_page()
# Run main()
if __name__ == '__main__':
    main()