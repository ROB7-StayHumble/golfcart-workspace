#!/usr/bin/env python3

import os
from yolov3.models import *  # set ONNX_EXPORT in models.py
from yolov3.utils.datasets import *
from yolov3.utils.utils import *

folder = 'src/persondetection/include/yolov3/'
CFG =       folder + 'cfg/yolov3-tiny.cfg'
WEIGHTS =   folder + 'weights/yolov3-tiny.weights'
SOURCE =    folder + 'data/samples'
OUTPUT =    folder + 'output'
DATA =      folder + 'data/coco.data'

HALF = False
VIEW_IMG = True
CONF_THRESH = 0.2
NMS_THRESH = 0.5

def detect_from_folder(save_txt=False, save_img=False):
    with torch.no_grad():
        img_size = 416
        out, source, weights, half, view_img = OUTPUT, SOURCE, WEIGHTS, HALF, VIEW_IMG
        webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')

        # Initialize
        device = torch_utils.select_device(device='')
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder

        # Initialize model
        model = Darknet(CFG, img_size)

        # Load weights
        attempt_download(weights)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        else:  # darknet format
            _ = load_darknet_weights(model, weights)

        # Second-stage classifier
        classify = False
        if classify:
            modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        model.to(device).eval()

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        # if webcam:
        #     view_img = True
        #     torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        #     dataset = LoadStreams(source, img_size=img_size, half=half)
        # else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size, half=half)
        print(dataset)

        # Get classes and colors
        # classes = load_classes(parse_data_cfg(DATA)['names'])
        classes = ['person']
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        # Run inference
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            t = time.time()

            # Get detections
            img = torch.from_numpy(img).to(device)
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            pred = model(img)[0]

            if HALF:
                pred = pred.float()

            # Apply NMS
            pred = non_max_suppression(pred, CONF_THRESH, NMS_THRESH)

            # Apply
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0 = path[i], '%g: ' % i, im0s[i]
                else:
                    p, s, im0 = path, '', im0s

                save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %ss, ' % (n, classes[int(c)])  # add to string

                    # Write results
                    for *xyxy, conf, _, cls in det:
                        if save_txt:  # Write to file
                            with open(save_path + '.txt', 'a') as file:
                                file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

                        if save_img or view_img:  # Add bbox to image
                            label = '%s %.2f' % (classes[int(cls)], conf)
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

                print('%sDone. (%.3fs)' % (s, time.time() - t))

                # Stream results
                if view_img:
                    cv2.imshow(p, im0)

                # Save results (image with detections)
                if save_img:
                    if dataset.mode == 'images':
                        cv2.imwrite(save_path, im0)

        if save_txt or save_img:
            print('Results saved to %s' % os.getcwd() + os.sep + out)

        print('Done. (%.3fs)' % (time.time() - t0))

def detect_from_img(img):
    print("Running detection")
    with torch.no_grad():
        img_size = 416
        out, source, weights, half, view_img = OUTPUT, SOURCE, WEIGHTS, HALF, VIEW_IMG

        # Initialize
        device = torch_utils.select_device(device='')

        # Initialize model
        model = Darknet(CFG, img_size)

        # Load weights
        # attempt_download(weights)
        _ = load_darknet_weights(model, weights)

        # Fuse Conv2d + BatchNorm2d layers
        # model.fuse()

        # Eval mode
        model.to(device).eval()

        # Half precision
        half = half and device.type != 'cpu'  # half precision only supported on CUDA
        if half:
            model.half()

        # Set Dataloader
        save_img = False

        # Get classes and colors
        # classes = load_classes(parse_data_cfg(DATA)['names'])
        classes = ['person']
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(classes))]

        # img0s = img  # BGR
        img0 = img
        im0 = img
        assert img0 is not None, 'Image is None'
        # Padded resize
        img = letterbox(img0, new_shape=img_size)[0]

        # Normalize RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB
        img = np.ascontiguousarray(img, dtype=np.float16 if HALF else np.float32)  # uint8 to fp16/fp32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0


        # Run inference
        t0 = time.time()

        t = time.time()

        # Get detections
        img = torch.from_numpy(img).to(device)
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = model(img)[0]

        if HALF:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, CONF_THRESH, NMS_THRESH)

        boxes = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            s = ''
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                for *xyxy, conf, _, cls in det:
                    if view_img and int(cls) == 0:  # Add bbox to image
                        label = '%.2f' % conf
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                        box = {'coords':[int(x) for x in [*xyxy]], 'conf':float(conf)}
                        boxes.append(box)

            print('%sDone. (%.3fs)' % ('', time.time() - t))

            # Stream results
            # if view_img:
            #     cv2.imshow('yolo', im0)
            #     cv2.waitKey(0)

    return im0, boxes
