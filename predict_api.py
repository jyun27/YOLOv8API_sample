#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 13:58:02 2023

@author: henry
"""



from flask import Flask, render_template, Response, request
import json
import argparse
import os
import sys
from pathlib import Path

from ultralytics import YOLO
from ultralytics.utils.checks import cv2, print_args
from utils.general import update_options

# Initialize paths
FILE = Path(__file__).resolve() # 현재 파일의 절대 경로 저장 
ROOT = FILE.parents[0] # 현재 파일의 부모 디렉토리를 저장. parents[0]는 현재 파일이 속한 디렉토리를 의미 
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd())) # 상대 경로로 변환(현재 작업 디렉 기준)

# Initialize Flask API
app = Flask(__name__)

def predict(opt):
    """
    predict 함수는 객체 감지를 수행하는 함수로, 옵션(opt)에 따라 결과를 JSON 형식 또는 이미지 바이트 형식으로 반환

    Perform object detection using the YOLO model and yield results.
    
    Parameters:
    - opt (Namespace): A namespace object that contains all the options for YOLO object detection,
        including source, model path, confidence thresholds, etc.
    
    Yields:
    - JSON: If opt.save_txt is True, yields a JSON string containing the detection results.
    - bytes: If opt.save_txt is False, yields JPEG-encoded image bytes with object detection results plotted.
    """
    
    results = model(**vars(opt), stream=True)

    # For saving JSON results if opt.save_txt is True
    if opt.save_txt:
        results_list = []
        for result in results:
            result_json = json.loads(result.tojson())
            results_list.append(result_json)
            # Stream image
            im0 = result.plot()
            _, jpeg = cv2.imencode('.jpg', im0)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Save the JSON file
        json_file_path = Path('result.json')
        with open(json_file_path, 'w') as f:
            json.dump(results_list, f, indent=4)
    else:
        for result in results:
            im0 = result.plot()
            _, jpeg = cv2.imencode('.jpg', im0)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    """
    Video streaming home page.
    """
    
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def video_feed():
    if request.method == 'POST':
        uploaded_file = request.files.get('myfile') # 클라이언트에서 파일이 업로드된 경우, request.files.get('myfile')을 통해 파일을 가져옴 
        save_txt = request.form.get('save_txt', 'T')  # Default to 'F' if save_txt is not provided

        if uploaded_file:
            source = Path(__file__).parent / raw_data / uploaded_file.filename
            uploaded_file.save(source)
            opt.source = source # 업로드된 파일 경로 
        else:
            opt.source, _ = update_options(request)
            
        opt.save_txt = True if save_txt == 'T' else False
            
    elif request.method == 'GET':
        opt.source, opt.save_txt = update_options(request)
       # GET 요청에 대한 응답 반환
        return Response(f"Source: {opt.source}", mimetype='text/plain')
    
    # 기본 응답
    #return Response({f'message': {opt.source}}), 400
    print(opt.save_txt)

    return Response(predict(opt), mimetype='multipart/x-mixed-replace; boundary=frame')
'''
     # JSON 파일로 결과를 저장하는 부분
    if opt.save_txt:
        result = next(predict(opt))  # 첫 번째 결과만 가져옴
        result_data = json.loads(result)
        json_file_path = Path('result.json')
        with open(json_file_path, 'w') as f:
            json.dump(result_data, f, indent=4)
        return jsonify(result_data)
'''
    



if __name__ == '__main__':
    # Input arguments
    # Flask 애플리케이션을 실행될 때 사용되는 input parameter을 설명하는 부분 
    # 여러 개의 인수를 추가하여 사용자로부터 다양한 입력을 받을 수 있도록 한다. 
    parser = argparse.ArgumentParser()
    parser.add_argument('--model','--weights', type=str, default=ROOT / '/Users/kimjyun/Desktop/2024/SeoulReChat/cloneCode/YOLOv8API/best.pt', help='model path or triton URL')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='source directory for images or videos')
    parser.add_argument('--conf','--conf-thres', type=float, default=0.25, help='object confidence threshold for detection')
    parser.add_argument('--iou', '--iou-thres', type=float, default=0.7, help='intersection over union (IoU) threshold for NMS')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='image size as scalar or (h, w) list, i.e. (640, 480)')
    parser.add_argument('--half', action='store_true', help='use half precision (FP16)')
    parser.add_argument('--device', default='', help='device to run on, i.e. cuda device=0/1/2/3 or device=cpu')
    parser.add_argument('--show','--view-img', default=False, action='store_true', help='show results if possible')
    parser.add_argument('--save', action='store_true', help='save images with results')
    parser.add_argument('--save_txt','--save-txt', action='store_true', help='save results as .txt file')
    parser.add_argument('--save_conf', '--save-conf', action='store_true', help='save results with confidence scores')
    parser.add_argument('--save_crop', '--save-crop', action='store_true', help='save cropped images with results')
    parser.add_argument('--show_labels','--show-labels', default=True, action='store_true', help='show labels')
    parser.add_argument('--show_conf', '--show-conf', default=True, action='store_true', help='show confidence scores')
    parser.add_argument('--max_det','--max-det', type=int, default=300, help='maximum number of detections per image')
    parser.add_argument('--vid_stride', '--vid-stride', type=int, default=1, help='video frame-rate stride')
    parser.add_argument('--stream_buffer', '--stream-buffer', default=False, action='store_true', help='buffer all streaming frames (True) or return the most recent frame (False)')
    parser.add_argument('--line_width', '--line-thickness', default=None, type=int, help='The line width of the bounding boxes. If None, it is scaled to the image size.')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize model features')
    parser.add_argument('--augment', default=False, action='store_true', help='apply image augmentation to prediction sources')
    parser.add_argument('--agnostic_nms', '--agnostic-nms', default=False, action='store_true', help='class-agnostic NMS')
    parser.add_argument('--retina_masks', '--retina-masks', default=False, action='store_true', help='whether to plot masks in native resolution')
    parser.add_argument('--classes', type=list, help='filter results by class, i.e. classes=0, or classes=[0,2,3]') # 'filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--boxes', default=True, action='store_false', help='Show boxes in segmentation predictions')
    parser.add_argument('--exist_ok', '--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--raw_data', '--raw-data', default=ROOT / 'data/raw', help='save raw images to data/raw')
    parser.add_argument('--port', default=5002, type=int, help='port deployment')
    opt, unknown = parser.parse_known_args()

 
    # print used arguments
    print_args(vars(opt))

    # Get por to deploy
    port = opt.port
    delattr(opt, 'port')
    
    # Create path for raw data
    raw_data = Path(opt.raw_data)
    raw_data.mkdir(parents=True, exist_ok=True)
    delattr(opt, 'raw_data')
    
    # Load model (Ensemble is not supported)
    model = YOLO(str(opt.model))

    # 모델 구조 및 경로 출력
    #print(model)
    print(f"Loading model from: {opt.model}")

    # Run app
    app.run(host='0.0.0.0', port=port, debug=True) # Don't use debug=True, model will be loaded twice (https://stackoverflow.com/questions/26958952/python-program-seems-to-be-running-twice)
    


    