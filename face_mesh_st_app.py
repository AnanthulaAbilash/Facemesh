import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
import tempfile
import time
from PIL import Image


mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

DEMO_IMAGE = 'demo.jpg'
DEMO_VIDEO = 'people.mp4'


st.title('App For Landmark(mesh) Enabled Face Detection')

st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    
    </style>
    
    """,
    unsafe_allow_html=True,
)

st.sidebar.title('Mesh detection sidebar')
st.sidebar.subheader('parameters')

@st.cache()
def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h,w) = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
        
    else:
        r = width/float(w)
        dim = (width, int(h*r))
        
     
    return cv2.resize(image, dim, interpolation=inter)

app_mode = st.sidebar.selectbox('Choose the App mode', ['App Info:', 'Run on Image', 'Run on Video']
                                   )

   
if app_mode == 'App Info:':
    st.markdown('**Mediapipe** for generating face mesh: App rendered with **streamlit**')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    
    </style>
    
    """,
    unsafe_allow_html=True,
)
    
    st.video('https://www.youtube.com/watch?v=IFxzo4CpFzY')
    
elif app_mode == 'Run on Image':
    drawing_spec= mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
    st.sidebar.markdown('---')
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    
    </style>
    
    """,
    unsafe_allow_html=True,
)
   
    st.markdown(f"<p style = 'color: DarkBlue; font-weight: bold; font-size:24px; font-family: Franklin Gothic Medium; text-align: center'> Face Mesh detection on Image</p>", unsafe_allow_html=True)
    
    max_faces = st.sidebar.number_input("Maximum Number of Face", value =2, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.1, max_value=1.0, value=0.5)
    st.sidebar.markdown('---')
    
    img_file_loaded = st.sidebar.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])
    
       
    if img_file_loaded is not None:
        try:
            image = np.array(Image.open(img_file_loaded))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception() as e:
        #print('error: ', e)
            st.markdown('**Image type Error - upload another image**')
            st.markdown(f"<p style = 'color: brown;'> <strong>ERROR: </strong>  {e}</p>", unsafe_allow_html=True)
        
        
    else: 
        demo_image = DEMO_IMAGE
        image = np.array(Image.open(demo_image))
    
    height, width, chl = image.shape
    if width>1000:
        image_resize(image, width=1000)
        
    st.sidebar.text('Input Image')
    st.sidebar.image(image)
    
    face_count = 0
    
    ##Analysis
    with mp_face_mesh.FaceMesh (
        static_image_mode = True, # for unrelated images
        max_num_faces = max_faces,
        min_detection_confidence =detection_confidence
        
    ) as face_mesh:
               #Face Landmarks display
        try:
            results = face_mesh.process(image)
            out_image = image.copy()
            for face_landmarks in results.multi_face_landmarks:
                face_count+=1
                mp_drawing.draw_landmarks(
                    image = out_image,
                    landmark_list = face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = drawing_spec
                )
                
            #dashboard
            sta_text = st.markdown("0")
            sta_text.write(f"<h3 style = 'text-align: center; color:red;'> No of faces: {face_count} </h3>", unsafe_allow_html=True)
            
            st.subheader('Post Processed Image after Detection')
            st.image(out_image, use_column_width=True)

        except TypeError:
           
            st.markdown(f"<p style = 'color: brown; font-weight: bold'> Landmark detection failed or No faces in the uploaded image</p>", unsafe_allow_html=True)
            st.markdown(f"<p style =  font-weight: italic; font-size: 16px; '> 'You may try decreasing the Detection confidence from the <b style = 'color: green; text-decoration: underline;'>sidebar</b> on the left'</p>", unsafe_allow_html=True)
        
        
        except Exception as e:
            st.markdown('**Image detection Error for the uploaded file**')
            st.markdown(f"<p style = 'color: brown;'> <strong>ERROR: </strong>  {e}</p>", unsafe_allow_html=True)
        
        
        
     ###
elif app_mode == 'Run on Video':
    st.set_option('deprecation.showfileUploaderEncoding', False)
    use_webcam = st.sidebar.button('Use Webcam')
    record = st.sidebar.checkbox('Record Video')
    
    if record:
        st.checkbox("Recording", value=True)
    
    st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 350px
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 350px
        margin-left: -350px
    }
    
    </style>
    
    """,
    unsafe_allow_html=True,
)
   
    st.markdown(f"<p style = 'color: DarkBlue; font-weight: bold; font-size:24px; font-family: Franklin Gothic Medium; text-align: center'> Face Mesh detection on Video</p>", unsafe_allow_html=True)
    
    max_faces = st.sidebar.number_input("Maximum Number of Face", value =5, min_value=1)
    st.sidebar.markdown('---')
    detection_confidence = st.sidebar.slider('Min Detection Confidence', min_value=0.1, max_value=1.0, value=0.5)
    tracking_confidence = st.sidebar.slider('Min Tracking Confidence', min_value=0.1, max_value=1.0, value=0.5)
    
    st.sidebar.markdown('---')
    
    stframe =st.empty()
    vid_file_loaded = st.sidebar.file_uploader("Upload a Video", type=["mp4", "mov", "avi", "asf", "m4v"])
    temfile = tempfile.NamedTemporaryFile(delete = False)
    
    ###  video 
    if vid_file_loaded is not None:
        try:
            temfile.write(vid_file_loaded.read())
            video = cv2.VideoCapture(temfile.name)
        except Exception() as e:
        #print('error: ', e)
            st.markdown('**Video type Error - upload another video**')
            st.markdown(f"<p style = 'color: brown;'> <strong>ERROR: </strong>  {e}</p>", unsafe_allow_html=True)
      
    else:
        if use_webcam:
            video = cv2.VideoCapture(0)
         
        demo_video = DEMO_VIDEO
        video = cv2.VideoCapture(DEMO_VIDEO)
    
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_ip = int(video.get(cv2.CAP_PROP_FPS))
    
     ##record
    codec = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    vid_op = cv2.VideoWriter('vid_output.mp4', codec, fps_ip, (width, height))
    
            
    st.sidebar.text('Input Video')
    st.sidebar.video(temfile.name)
    
    drawing_spec= mp_drawing.DrawingSpec(thickness=2, circle_radius=1)
    
    kpi1, kpi2 = st.columns(2)
    
    with kpi1:
        st.markdown(f"<h6 style = 'text-align: center;'> Frame Rate </h6>", unsafe_allow_html=True)
        kpi1_text = st.markdown("0")
    
    with kpi2:
        st.markdown(f"<h6 style = 'text-align: center;'> No of Faces detected </h6>", unsafe_allow_html=True)
        kpi2_text = st.markdown("0")
        

     
    ##Mesh Analysis
    with mp_face_mesh.FaceMesh (
        max_num_faces = max_faces,
        min_detection_confidence =detection_confidence,
        min_tracking_confidence =tracking_confidence
        
    ) as face_mesh:
               #Face Landmarks display
        prevTime = time.time()
        i, j =0, 0
        
        while video.isOpened():
            i += 1
            ret, frame = video.read()
            if not ret:
                continue
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            face_count = 0
            
            if results.multi_face_landmarks:
                j+=1
                
                for face_landmarks in results.multi_face_landmarks:
                    face_count+=1
                    
                    mp_drawing.draw_landmarks(
                        image = frame,
                        landmark_list = face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec = drawing_spec,
                        connection_drawing_spec = drawing_spec
                    )
                
                currTime = time.time()
                fps = 1/(currTime-prevTime)
                prevTime = currTime
                
                if record:
                    vid_op.write(frame)
                    
                #Dashboard
                kpi1_text.write(f"<h3 style = 'text-align: center; color:red;'> {int(fps)} </h3>", unsafe_allow_html=True)
                
                kpi2_text.write(f"<h3 style = 'text-align: center; color:red;'> {face_count} </h3>", unsafe_allow_html=True)
                
                #frame = cv2.resize(frame, (0,0), fx=0.8, fy=0.8)
                frame = image_resize(frame, width=640)
                
                stframe.image(frame, channels='BGR', use_column_width=True)
            
            
    if j==0:        
        st.markdown(f"<p style = 'color: brown; font-weight: bold'> Landmark detection failed or No faces in the uploaded image</p>", unsafe_allow_html=True)
        st.markdown(f"<p style =  font-weight: italic; font-size: 16px; '> 'You may try decreasing the Detection confidence from the <b style = 'color: green; text-decoration: underline;'>sidebar</b> on the left or choose another video source/file'</p>", unsafe_allow_html=True)
    

        

