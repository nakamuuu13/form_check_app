import mediapipe as mp
import cv2
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from lime.lime_tabular import LimeTabularExplainer


# 腰を減点に座標を修正
def coordinate_transformation(left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y \
                             ,left_hip_x, left_hip_y, right_hip_x, right_hip_y \
                             ,left_knee_x, left_knee_y, right_knee_x, right_knee_y \
                             ,left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y \
                             ,left_heel_x, left_heel_y, right_heel_x, right_heel_y \
                             ,left_foot_index_x, left_foot_index_y, right_foot_index_x, right_foot_index_y):

    hip_center_x = (left_hip_x + right_hip_x) / 2
    hip_center_y = (left_hip_y + right_hip_y) / 2

    left_shoulder_x = left_shoulder_x - hip_center_x
    left_shoulder_y = left_shoulder_y - hip_center_y
    right_shoulder_x = right_shoulder_x - hip_center_x
    right_shoulder_y = right_shoulder_y - hip_center_y
    left_hip_x = left_hip_x - hip_center_x
    left_hip_y = left_hip_y - hip_center_y
    right_hip_x = right_hip_x - hip_center_x
    right_hip_y = right_hip_y - hip_center_y
    left_knee_x = left_knee_x - hip_center_x
    left_knee_y = left_knee_y - hip_center_y
    right_knee_x = right_knee_x - hip_center_x
    right_knee_y = right_knee_y - hip_center_y
    right_ankle_x = right_ankle_x - hip_center_x
    right_ankle_y = right_ankle_y - hip_center_y
    left_ankle_x = left_ankle_x - hip_center_x
    left_ankle_y = left_ankle_y - hip_center_y
    right_ankle_x = right_ankle_x - hip_center_x
    right_ankle_y = right_ankle_y - hip_center_y
    left_heel_x = left_heel_x - hip_center_x
    left_heel_y = left_heel_y - hip_center_y
    right_heel_x = right_heel_x - hip_center_x
    right_heel_y = right_heel_y - hip_center_y
    left_foot_index_x = left_foot_index_x - hip_center_x
    left_foot_index_y = left_foot_index_y - hip_center_y
    right_foot_index_x = right_foot_index_x - hip_center_x
    right_foot_index_y = right_foot_index_y - hip_center_y


    return left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y \
          ,left_hip_x, left_hip_y, right_hip_x, right_hip_y \
          ,left_knee_x, left_knee_y, right_knee_x, right_knee_y \
          ,left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y \
          ,left_heel_x, left_heel_y, right_heel_x, right_heel_y \
          ,left_foot_index_x, left_foot_index_y, right_foot_index_x, right_foot_index_y

# PIL型をOpenCV型に変換
def pil2cv(image):
    new_image = np.array(image, dtype=np.uint8)
    if new_image.ndim == 2:  # モノクロ
        pass
    elif new_image.shape[2] == 3:  # カラー
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    elif new_image.shape[2] == 4:  # 透過
        new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
    return new_image

# LIME 用の予測関数
def predict_fn(x):
    if len(x.shape) == 1:
        return loaded_model.predict_proba(x.reshape(1, -1))[0]
    else:
        return loaded_model.predict_proba(x)



st.title('kick form check app!')

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mesh_drawing_spec = mp_drawing.DrawingSpec(thickness=2, color=(0, 255, 0))
mark_drawing_spec = mp_drawing.DrawingSpec(thickness=3, circle_radius=3, color=(0, 0, 255))

uploaded_img = st.file_uploader('Choose a kick form image', type='png')
if uploaded_img is not None:
      img = Image.open(uploaded_img)
      img = pil2cv(img)
      
      with mp_pose.Pose(
            min_detection_confidence=0.5,
            static_image_mode=True) as pose_detection:

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = pose_detection.process(rgb_img)

            annotated_img = rgb_img.copy()

            left_shoulder_x = results.pose_landmarks.landmark[11].x #* width
            left_shoulder_y = results.pose_landmarks.landmark[11].y #* height
            right_shoulder_x = results.pose_landmarks.landmark[12].x #* width
            right_shoulder_y = results.pose_landmarks.landmark[12].y #* height
            left_hip_x = results.pose_landmarks.landmark[23].x #* width
            left_hip_y = results.pose_landmarks.landmark[23].y #* height
            right_hip_x = results.pose_landmarks.landmark[24].x #* width
            right_hip_y = results.pose_landmarks.landmark[24].y #* height
            left_knee_x = results.pose_landmarks.landmark[25].x #* width
            left_knee_y = results.pose_landmarks.landmark[25].y #* height
            right_knee_x = results.pose_landmarks.landmark[26].x #* width
            right_knee_y = results.pose_landmarks.landmark[26].y #* height
            left_ankle_x = results.pose_landmarks.landmark[27].x #* width
            left_ankle_y = results.pose_landmarks.landmark[27].y #* height
            right_ankle_x = results.pose_landmarks.landmark[28].x #* width
            right_ankle_y = results.pose_landmarks.landmark[28].y #* height
            left_heel_x = results.pose_landmarks.landmark[29].x #* width
            left_heel_y = results.pose_landmarks.landmark[29].y #* height
            right_heel_x = results.pose_landmarks.landmark[30].x #* width
            right_heel_y = results.pose_landmarks.landmark[30].y #* height
            left_foot_index_x = results.pose_landmarks.landmark[31].x #* width
            left_foot_index_y = results.pose_landmarks.landmark[31].y #* height
            right_foot_index_x = results.pose_landmarks.landmark[32].x #* width
            right_foot_index_y = results.pose_landmarks.landmark[32].y #* height


            left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y \
            ,left_hip_x, left_hip_y, right_hip_x, right_hip_y \
            ,left_knee_x, left_knee_y, right_knee_x, right_knee_y \
            ,left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y \
            ,left_heel_x, left_heel_y, right_heel_x, right_heel_y \
            ,left_foot_index_x, left_foot_index_y, right_foot_index_x, right_foot_index_y \
            = coordinate_transformation(left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y \
                                    ,left_hip_x, left_hip_y, right_hip_x, right_hip_y \
                                    ,left_knee_x, left_knee_y, right_knee_x, right_knee_y \
                                    ,left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y \
                                    ,left_heel_x, left_heel_y, right_heel_x, right_heel_y \
                                    ,left_foot_index_x, left_foot_index_y, right_foot_index_x, right_foot_index_y)


            mp_drawing.draw_landmarks(
                  image=annotated_img,
                  landmark_list=results.pose_landmarks,
                  connections=mp_pose.POSE_CONNECTIONS,
                  landmark_drawing_spec=mark_drawing_spec,
                  connection_drawing_spec=mesh_drawing_spec
                  )


            columns_name = ['left_shoulder_x', 'left_shoulder_y', 'right_shoulder_x', 'right_shoulder_y'\
                        ,'left_hip_x', 'left_hip_y', 'right_hip_x', 'right_hip_y'\
                        ,'left_knee_x', 'left_knee_y', 'right_knee_x', 'right_knee_y'\
                        ,'left_ankle_x', 'left_ankle_y', 'right_ankle_x', 'right_ankle_y'\
                        ,'left_heel_x', 'left_heel_y', 'right_heel_x', 'right_heel_y'\
                        ,'left_foot_index_x', 'left_foot_index_y', 'right_foot_index_x', 'right_foot_index_y' \
                        ]


            predict_df = pd.DataFrame(data=np.array([[left_shoulder_x, left_shoulder_y, right_shoulder_x, right_shoulder_y \
                                                ,left_hip_x, left_hip_y, right_hip_x, right_hip_y \
                                                ,left_knee_x, left_knee_y, right_knee_x, right_knee_y \
                                                ,left_ankle_x, left_ankle_y, right_ankle_x, right_ankle_y \
                                                ,left_heel_x, left_heel_y, right_heel_x, right_heel_y \
                                                ,left_foot_index_x, left_foot_index_y, right_foot_index_x, right_foot_index_y \
                                                      ]]),
                                    columns=columns_name)

      # 標準化
      std_scaler = StandardScaler()
      x_train = pd.read_csv('form_clustering_std_x_train.csv')
      x_train = x_train.drop(columns=['Unnamed: 0'])
      std_scaler.fit(x_train)
      predict_std = std_scaler.transform(predict_df)

      # モデルのロード
      loaded_model = pickle.load(open('form_clustering_model.sav', 'rb'))

      # 推論
      predict = loaded_model.predict(predict_std)

      col1, col2= st.columns(2)

      with col1:
            if predict == 0:
                  st.header('Your form is good!!')
                  st.image(annotated_img, use_column_width=True)
            else:
                  st.write('Your form is bad.')
                  st.image(annotated_img, use_column_width=True)
      with col2:
            st.header('Good example')
            st.image("sample.jpg", use_column_width=True)

      # LIME の実装
      # インスタンス化
      explainer = LimeTabularExplainer(x_train.values, class_names=['Good', 'Bad'], feature_names = x_train.columns)

      exp = explainer.explain_instance(predict_std[0], predict_fn, num_features=24)

      # 可視化
      as_list = exp.as_list()
      as_map = exp.as_map()
      as_figure = exp.as_pyplot_figure()
      st.pyplot(as_figure)