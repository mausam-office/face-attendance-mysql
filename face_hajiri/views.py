from email.mime import image
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from django.http import QueryDict

from .models import Registration, Attendance
from .serializers import RegistrationSerializer, FaceVerificationSerializer, AttendanceSerializer, UserDetailsSerializer

import os
import io
import base64
from PIL import Image
import numpy as np
import cv2
import face_recognition
from datetime import datetime
import json
from json import JSONEncoder
import time

from core.utils import gen_encoding, automatic_brightness_and_contrast



stored_encodings = None
attendee_names = None
attendee_ids = None
duration = 60   # in seconds

def base64_img(img_str):
    image = base64.b64decode((img_str))
    image = Image.open(io.BytesIO(image))
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # cv2.imwrite('detections/image.jpg', image)
    return image


def img_preprocessing(img_str):
    image = base64_img(img_str)
    # can do register only when single face is encountered
    # Returns list of tuples: one for each face
    # if more than one face appears return 
    face_cropped = face_recognition.face_locations(image, model = "cnn")
    if len(face_cropped) > 1:
        return Response({'Acknowledge' : 'Multiple Faces Detected.'}, status=status.HTTP_206_PARTIAL_CONTENT)
    else:
        return gen_encoding([image])

def get_user_data():
    reg = Registration.objects.all()
    reg_serializer = RegistrationSerializer(reg, many=True)
    stored_encodings = []
    attendee_names = []
    attendee_ids = []
    for row in reg_serializer.data:
        face_embedding = row['face_embedding']['face_embedding']
        face_embedding = json.loads(str(face_embedding))
        decoded_face_embedding = np.asarray(face_embedding)
        attendee_name = row['attendee_name']
        attendee_id = row['attendee_id']
        stored_encodings.append(decoded_face_embedding)
        attendee_names.append(attendee_name)
        attendee_ids.append(attendee_id)
    return stored_encodings, attendee_names, attendee_ids


def construct_dict(name, id, device, current_time, state):
    query_dict = QueryDict('', mutable=True)
    if state == 'in':
        data = {
            'attendee_name' : name,
            'attendee_id' : id,
            'device' : device,
            'in_time' : current_time
        }
        query_dict.update(data)
        return query_dict
    else:
        return query_dict


def store_in_time(name, id, device, current_time, state):
    query_dict = construct_dict(name, id, device, current_time, state)
    if query_dict:
        serializer_attendance = AttendanceSerializer(data = query_dict)
        if serializer_attendance.is_valid():
            # print('attendance serializer is valid in else-if')
            serializer_attendance.save()
        else:
            print('attendance serializer is invalid')


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class RegistrationView(APIView):
    # def img_preprocessing(self, img_str):
    #     image = base64_img(img_str)
    #     # can do register only when single face is encountered
    #     # Returns list of tuples: one for each face
    #     # if more than one face appears return 
    #     face_cropped = face_recognition.face_locations(image, model = "cnn")
    #     print('faces num: ', len(face_cropped), type(len(face_cropped)))
    #     if len(face_cropped) > 1:
            
    #         return Response({'Acknowledge' : 'Multiple Faces Detected.'}, status=status.HTTP_206_PARTIAL_CONTENT)
    #     else:
    #         return gen_encoding([image])

    def post(self, request):
        global stored_encodings
        global attendee_names
        global attendee_ids

        
        # data grabbing
        try:
            data = request.data
            # print("try")
        except:
            data = request.POST
            # print('ex')
        # print(data)

        if data['attendee_id'] in attendee_ids:
            return Response({'Acknowledge' : 'ID already exists'}, status=status.HTTP_208_ALREADY_REPORTED)
        
        # generation of face_encoding
        if data['image_base64']:
            try:
                face_encoding = img_preprocessing(data['image_base64'])
                print('face encoding completed')
                face_encoding = {"face_embedding": face_encoding}
                print('dict built')
                encoded_face_encoding = json.dumps(face_encoding, cls=NumpyArrayEncoder)
                print("json built")
            except:
                return Response({'Acknowledge':'invalid image data'}, status=status.HTTP_206_PARTIAL_CONTENT)
        
        # dict used in generation of query dict
        data_ = {
            'attendee_name' : data['attendee_name'].strip(),
            'attendee_id' : data['attendee_id'].strip(),
            'registration_device' : data['registration_device'].strip(),
            'department' : data['department'],
            'image_base64' : data['image_base64'],
            'face_embedding' : encoded_face_encoding,
        }
        query_dict = QueryDict('', mutable=True)
        query_dict.update(data_)

        serializer = RegistrationSerializer(data=query_dict)
        if serializer.is_valid():
            serializer.save()
            stored_encodings, attendee_names, attendee_ids = get_user_data()
            
            return Response({'Acknowledge':'User Created successfully'}, status=status.HTTP_200_OK)
        print(serializer.errors)
        return Response({'Acknowledge':serializer.errors}, status=status.HTTP_400_BAD_REQUEST)




class VerificationView(APIView):
    def post(self, request):
        global stored_encodings
        global attendee_names
        global attendee_ids
        
        if stored_encodings is None or attendee_names is None or attendee_ids is None:
            stored_encodings, attendee_names, attendee_ids = get_user_data()
        
        start = time.time()
        try:
            for filename in os.listdir('/'.join([os.getcwd(), 'media/'])):
                filepath = '/'.join([os.getcwd(), f"media/{filename}"])
                os.remove(filepath)
        except:
            pass

        try:
            data = request.data
            # print("try")
        except:
            data = request.POST
            # print('ex')
        # print(f'data arrived {data}')
        
        serializer = FaceVerificationSerializer(data=data)
        if serializer.is_valid():
            # print('valid')
            current_time = datetime.now()
            # serializer_data = serializer.data
            
            # added
            serializer_data = serializer.initial_data
            serializer.save()

            # commented
            # img_str = serializer_data['image_base64']
            # image = base64_img(img_str)

            # changed to
            filepath = '/'.join([os.getcwd(), 'media/face.jpg'])
            # image = cv2.imread(filepath)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # OR Check the image reading with PIL 
            image = Image.open(filepath)
            image = np.array(image)


            face_cropped = face_recognition.face_locations(image, model = "cnn")
            # print(f'Number of faces {len(face_cropped)}')
            encoded_face_in_frame = face_recognition.face_encodings(image, face_cropped)
            recognized_faces = []
            for encode_face, face_loc in zip(encoded_face_in_frame, face_cropped):
                matches = face_recognition.compare_faces(stored_encodings, encode_face, tolerance=0.45)
                face_dist = face_recognition.face_distance(stored_encodings, encode_face)
                try:
                    match_index = np.argmin(face_dist)
                    # matched_face_distance = face_dist[match_index]
                    if matches[match_index]:
                        # aggregate matched user data
                        name = attendee_names[match_index].upper()
                        id = attendee_ids[match_index]
                        dist = face_dist[match_index]
                        # recognized_faces.append({'name':name, 'id':id})

                        # write data to database
                        rows = Attendance.objects.filter(date=datetime.now().date()).filter(attendee_name=name, attendee_id=id)

                        if rows:
                            '''If any rocord is found'''
                            row = rows.order_by('-in_time')[:1].get()
                            if row.out_time is None:
                                # save only if duration exceeds threshold
                                diff = datetime.now() - row.in_time
                                if diff.total_seconds() > duration:
                                    row.out_time = current_time
                                    row.save()
                                    recognized_faces.append({'name':name, 'id':id, 'state':'out', 'current_time':current_time, 'dist':dist})
                            elif isinstance(row.in_time, datetime):
                                diff = datetime.now() - row.out_time
                                if diff.total_seconds() > duration:
                                    store_in_time(name, id, serializer_data['device'], current_time, state='in')
                                    recognized_faces.append({'name':name, 'id':id, 'state':'in', 'current_time':current_time, 'dist':dist})
                            
                        else:
                            '''If no rocord is found'''
                            store_in_time(name, id, serializer_data['device'], current_time, state='in')
                            recognized_faces.append({'name':name, 'id':id, 'state':'in', 'current_time':current_time, 'dist':dist})

                    #         y1, x2, y2, x1 = face_loc
                    #         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    #         cv2.putText(image, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    # cv2.imwrite('detections/detection.jpg', image)   
                except:
                    return Response({'Acknowledge':'no any registered faces'})

            # print(recognized_faces)          
            # print(f'Verification time: {time.time()-start}')
            return Response({'Acknowledge':recognized_faces})
        return Response({'Acknowledge':'error'})

class UserDetailsView(APIView):
    def get(self, request):
        global stored_encodings
        global attendee_names
        global attendee_ids
        
        if stored_encodings is None or attendee_names is None or attendee_ids is None:
            stored_encodings, attendee_names, attendee_ids = get_user_data()
            
        reg_data = Registration.objects.all()#.distinct()       # distinct changes sequential order
        reg_data = UserDetailsSerializer(reg_data, many=True)

        return Response({'Acknowledge': reg_data.data, 'ids':attendee_ids})

