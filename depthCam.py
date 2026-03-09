import pyrealsense2 as rs
import numpy as np
import cv2

# Configure pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

try:
    pipeline.start(config)
    # Reduce frame queue size to 1
    depth_sensor = pipeline.get_active_profile().get_device().first_depth_sensor()
    depth_sensor.set_option(rs.option.frames_queue_size, 1)
    
    while True:
        frames = pipeline.wait_for_frames()
        align_to = rs.stream.color
        align = rs.align(align_to)
        frames = align.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            continue
        
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Resize for faster display
        resized_color = cv2.resize(color_image, (640, 480))
        resized_depth = cv2.resize(depth_image, (640, 480))
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(resized_depth, alpha=0.03), cv2.COLORMAP_JET) 

        # cv2.imshow('Color', resized_color)
        # cv2.imshow('Depth', cv2.applyColorMap(cv2.convertScaleAbs(resized_depth, alpha=0.03), cv2.COLORMAP_JET))
        cv2.imshow('RealSense', np.hstack((color_image, depth_colormap)))
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
