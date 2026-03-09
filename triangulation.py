import time
import traceback
import sys
import numpy as np
import cv2
import targeting_tools as tt

def run():

    try:
        # cameras variables
        pixel_width = 320
        pixel_height = 240
        angle_width = 70
        angle_height = 56
        frame_rate = 20
        camera_separation = 5.9

        # Configure based on dual camera module type
        # Try different approaches for dual cameras
        
        print("Attempting to connect to dual camera module...")
        camera_mode,idx = detect_camera_mode()
        
        if camera_mode == "SIDE_BY_SIDE":
            # Camera returns a single wide frame with both images side-by-side
            print("Detected side-by-side dual camera module")
            
            # Try to open the camera first to verify its accessible
            # Change: try index 0 first as it's more commonly available
            cap = cv2.VideoCapture(idx)  
            if not cap.isOpened():
                print("Failed to open camera at index 0, trying index 1...")
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    print("Failed to open camera at any index, exiting")
                    return
                camera_idx = 0
            else:
                camera_idx = idx
                
            # Check if we can actually read frames
            ret, test_frame = cap.read()
            if not ret or test_frame is None or test_frame.size == 0:
                print("Failed to read a valid frame from the camera")
                cap.release()
                return
                
            # Get actual frame dimensions
            actual_width = test_frame.shape[1]
            actual_height = test_frame.shape[0]
            print(f"Camera returned frame with dimensions: {actual_width}x{actual_height}")
            
            # If frame isn't wide enough to be split, adjust
            if actual_width < 2 * pixel_width:
                print(f"Warning: Camera frame width {actual_width} is less than expected {2*pixel_width}")
                # Adjust pixel width to half of actual width
                pixel_width = actual_width // 2
                print(f"Adjusting to use pixel width of {pixel_width}")
            
            cap.release()
            
            # MAJOR CHANGE: Instead of using two camera threads on the same camera,
            # use one thread and split the frames in the main loop
            # print("Using single camera approach with frame splitting")
            
            # Single camera thread
            ct_main = tt.Camera_Thread()
            ct_main.camera_number = camera_idx
            ct_main.camera_width = actual_width  # Use full width
            ct_main.camera_height = actual_height
            ct_main.camera_frame_rate = frame_rate
            ct_main.camera_source = ""  # No split in thread
            
            # Try different codecs
            try_codecs = [
                cv2.VideoWriter_fourcc(*"MJPG"),
                cv2.VideoWriter_fourcc(*"YUYV"),
                cv2.VideoWriter_fourcc(*"YUY2"),
                0  # Default codec
            ]
            
            # Use a single camera thread
            success = False
            for codec in try_codecs:
                print(f"Trying codec: {codec}")
                ct_main.camera_fourcc = codec
                
                try:
                    print("Starting camera...")
                    ct_main.start()
                    time.sleep(1.0)  # Give more time to initialize
                    
                    # Test if camera actually works - try multiple times
                    for _ in range(5):
                        frame = ct_main.next(black=True, wait=1)
                        if frame is not None and frame.size > 0:
                            print(f"Successfully got frame: {frame.shape}")
                            success = True
                            break
                        time.sleep(0.2)
                    
                    if success:
                        print(f"Successfully started camera with codec: {codec}")
                        break
                    else:
                        print("Failed to get valid frame")
                        ct_main.stop()
                        time.sleep(1.0)
                    
                except Exception as e:
                    print(f"Error starting camera with codec {codec}: {e}")
                    traceback.print_exc()
                    try:
                        ct_main.stop()
                    except:
                        pass
                    time.sleep(1.0)
            
            if not success:
                print("Failed to start camera with any codec")
                return
                
            # Define a function to split frames
            def split_frame(full_frame):
                h, w = full_frame.shape[:2]
                mid = w // 2
                left_frame = full_frame[:, :mid].copy()
                right_frame = full_frame[:, mid:].copy()
                return left_frame, right_frame
                
        elif camera_mode == "TWO_INDICES":
            # Two separate camera indices
            print("Detected dual camera as two separate indices")
            
            # left camera (likely index 0)
            ct1 = tt.Camera_Thread()
            ct1.camera_number = 0
            ct1.camera_source = ""  # No split needed
            ct1.camera_width = pixel_width
            ct1.camera_height = pixel_height
            ct1.camera_frame_rate = frame_rate
            
            # right camera (likely index 1)
            ct2 = tt.Camera_Thread()
            ct2.camera_number = 1
            ct2.camera_source = ""  # No split needed
            ct2.camera_width = pixel_width
            ct2.camera_height = pixel_height
            ct2.camera_frame_rate = frame_rate
            
            # Try different camera codecs
            try_codecs = [
                cv2.VideoWriter_fourcc(*"MJPG"),
                cv2.VideoWriter_fourcc(*"YUYV"),
                cv2.VideoWriter_fourcc(*"YUY2"),
                0  # Default codec
            ]
            
            print("Starting left camera...")
            if not start_camera_safely(ct1, try_codecs):
                print("Failed to start left camera with all codecs, exiting")
                return
                
            print("Starting right camera...")  
            if not start_camera_safely(ct2, try_codecs):
                print("Failed to start right camera with all codecs, exiting")
                ct1.stop()  # Stop the first camera if it started
                return
                
            # Define split_frame as identity for this mode
            def split_frame(dummy_frame):
                return None, None  # Not used in this mode
        else:
            print("Could not detect a working dual camera configuration")
            return
            
        print("Camera setup complete!")

        # set up angles 

        # cameras are the same, so only 1 needed
        angler = tt.Frame_Angles(pixel_width, pixel_height, angle_width, angle_height)
        angler.build_frame()

        # set up motion detection 

        # motion camera1
        targeter1 = tt.Frame_Motion()
        targeter1.contour_min_area = 1
        targeter1.targets_max = 1
        targeter1.target_on_contour = True # False = use box size
        targeter1.target_return_box = False # (x,y,bx,by,bw,bh)
        targeter1.target_return_size = True # (x,y,%frame)
        targeter1.contour_draw = True
        targeter1.contour_box_draw = False
        targeter1.targets_draw = True

        # motion camera2
        targeter2 = tt.Frame_Motion()
        targeter2.contour_min_area = 1
        targeter2.targets_max = 1
        targeter2.target_on_contour = True # False = use box size
        targeter2.target_return_box = False # (x,y,bx,by,bw,bh)
        targeter2.target_return_size = True # (x,y,%frame)
        targeter2.contour_draw = True
        targeter2.contour_box_draw = False
        targeter2.targets_draw = True

        # stabilize 

        # pause to stabilize
        print("Stabilizing cameras...")
        time.sleep(1.0)

        # targeting loop 

        # variables
        maxsd = 2 # maximum size difference of targets, percent of frame
        klen  = 3 # length of target queues, positive target frames required to reset set X,Y,Z,D

        # target queues
        x1k, y1k, x2k, y2k = [], [], [], []
        x1m, y1m, x2m, y2m = 0, 0, 0, 0

        # last positive target
        # from camera baseline midpoint
        X, Y, Z, D = 0, 0, 0, 0

        # Maximum consecutive errors before exiting
        max_errors = 5
        error_count = 0

        print("Starting triangulation loop...")
        # loop
        while 1:
            try:
                if camera_mode == "SIDE_BY_SIDE":
                    # Check if camera is still running
                    if not hasattr(ct_main, 'buffer') or ct_main.buffer is None:
                        print("Camera buffer is no longer valid - camera thread may have died")
                        break
                        
                    # get frame with error handling
                    try:
                        full_frame = ct_main.next(black=True, wait=1)
                        if full_frame is None or full_frame.size == 0:
                            print("Warning: Received invalid frame, skipping")
                            error_count += 1
                            if error_count > max_errors:
                                print(f"Too many invalid frames ({error_count}), exiting")
                                break
                            time.sleep(0.1)
                            continue
                    except Exception as e:
                        print(f"Error getting frame from camera: {e}")
                        traceback.print_exc()
                        error_count += 1
                        if error_count > max_errors:
                            print(f"Too many errors ({error_count}), exiting")
                            break
                        time.sleep(0.1)
                        continue
                    
                    # Split the frame
                    frame1, frame2 = split_frame(full_frame)
                    
                    # Extra check to ensure frames are the right size
                    h1, w1 = frame1.shape[:2]
                    h2, w2 = frame2.shape[:2]
                    if w1 != pixel_width or h1 != pixel_height or w2 != pixel_width or h2 != pixel_height:
                        print(f"Warning: Frame dimensions mismatch. Expected {pixel_width}x{pixel_height}, Got {w1}x{h1} and {w2}x{h2}")
                        # Resize if needed
                        frame1 = cv2.resize(frame1, (pixel_width, pixel_height))
                        frame2 = cv2.resize(frame2, (pixel_width, pixel_height))
                    
                    # Get FPS
                    fps1 = int(ct_main.current_frame_rate) if hasattr(ct_main, 'current_frame_rate') else 0
                    fps2 = fps1  # Same for both since it's one camera
                    
                else:  # TWO_INDICES mode
                    # Check if cameras are still running
                    if not hasattr(ct1, 'buffer') or ct1.buffer is None or not hasattr(ct2, 'buffer') or ct2.buffer is None:
                        print("Camera buffers are no longer valid - camera thread may have died")
                        break
                        
                    # get frames with error handling
                    try:
                        frame1 = ct1.next(black=True, wait=1)
                    except Exception as e:
                        print(f"Error getting frame from camera 1: {e}")
                        error_count += 1
                        if error_count > max_errors:
                            print(f"Too many errors ({error_count}), exiting")
                            break
                        time.sleep(0.1)
                        continue
                        
                    try:
                        frame2 = ct2.next(black=True, wait=1)
                    except Exception as e:
                        print(f"Error getting frame from camera 2: {e}")
                        error_count += 1
                        if error_count > max_errors:
                            print(f"Too many errors ({error_count}), exiting")
                            break
                        time.sleep(0.1)
                        continue

                    # Check if frames are valid
                    if frame1 is None or frame2 is None or frame1.size == 0 or frame2.size == 0:
                        print("Warning: Received invalid frame, skipping")
                        error_count += 1
                        if error_count > max_errors:
                            print(f"Too many invalid frames ({error_count}), exiting")
                            break
                        time.sleep(0.1)
                        continue
                        
                    # Get FPS
                    fps1 = int(ct1.current_frame_rate) if hasattr(ct1, 'current_frame_rate') else 0
                    fps2 = int(ct2.current_frame_rate) if hasattr(ct2, 'current_frame_rate') else 0
                
                # Reset error count since we got good frames
                error_count = 0

                # motion detection targets
                targets1 = targeter1.targets(frame1)
                targets2 = targeter2.targets(frame2)

                # check 1: motion in both frames
                if not (targets1 and targets2):
                    x1k, y1k, x2k, y2k = [], [], [], []  # reset
                else:
                    # split
                    x1, y1, s1 = targets1[0]
                    x2, y2, s2 = targets2[0]

                    # check 2: similar size
                    if abs(s1-s2) > maxsd:
                        x1k, y1k, x2k, y2k = [], [], [], []  # reset
                    else:
                        # update queues
                        x1k.append(x1)
                        y1k.append(y1)
                        x2k.append(x2)
                        y2k.append(y2)

                        # check 3: queues full
                        if len(x1k) >= klen:
                            # trim
                            x1k = x1k[-klen:]
                            y1k = y1k[-klen:]
                            x2k = x2k[-klen:]
                            y2k = y2k[-klen:]

                            # mean values
                            x1m = sum(x1k)/klen
                            y1m = sum(y1k)/klen
                            x2m = sum(x2k)/klen
                            y2m = sum(y2k)/klen
                                    
                            # get angles from camera centers
                            xlangle, ylangle = angler.angles_from_center(x1m, y1m, top_left=True, degrees=True)
                            xrangle, yrangle = angler.angles_from_center(x2m, y2m, top_left=True, degrees=True)
                            
                            # triangulate
                            X, Y, Z, D = angler.location(camera_separation, (xlangle, ylangle), (xrangle, yrangle), center=True, degrees=True)
            
                # display camera centers
                angler.frame_add_crosshairs(frame1)
                angler.frame_add_crosshairs(frame2)

                # display coordinate data
                text = f'X: {X:3.1f}\nY: {Y:3.1f}\nZ: {Z:3.1f}\nD: {D:3.1f}\nFPS: {fps1}/{fps2}'
                lineloc = 0
                lineheight = 30
                for t in text.split('\n'):
                    lineloc += lineheight
                    cv2.putText(frame1,
                                t,
                                (10, lineloc),  # location
                                cv2.FONT_HERSHEY_PLAIN,  # font
                                1.5,  # size
                                (0, 255, 0),  # color
                                1,  # line width
                                cv2.LINE_AA,  #
                                False)  #

                # display current target
                # if x1k:
                #     targeter1.frame_add_crosshairs(frame1, x1m, y1m, 48)            
                #     targeter2.frame_add_crosshairs(frame2, x2m, y2m, 48)            

                # display frame
                cv2.imshow("Left Camera", frame1)
                cv2.imshow("Right Camera", frame2)

                # detect keys
                key = cv2.waitKey(1) & 0xFF
                if cv2.getWindowProperty('Left Camera', cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif cv2.getWindowProperty('Right Camera', cv2.WND_PROP_VISIBLE) < 1:
                    break
                elif key == ord('q'):
                    break
                elif key != 255:
                    print('KEY PRESS:', [chr(key)])
                    
            except Exception as e:
                print(f"Error in main loop: {e}")
                traceback.print_exc()
                # Count errors
                error_count += 1
                if error_count > max_errors:
                    print(f"Too many errors ({error_count}), exiting")
                    break
                time.sleep(0.1)

    except Exception as e:
        print(f"Fatal error: {e}")
        print(traceback.format_exc())

    # close cameras based on mode
    try:
        if camera_mode == "SIDE_BY_SIDE":
            ct_main.stop()
        else:
            ct1.stop()
            ct2.stop()
    except Exception as e:
        print(f"Error stopping cameras: {e}")

    # kill frames
    cv2.destroyAllWindows()

    # done
    print('DONE')

def detect_camera_mode():
    """Detect the dual camera mode (side-by-side or two indices)"""
    print("Testing for side-by-side camera mode...")
    # First try side-by-side mode on each possible camera index
    for idx in range(2):  # Try camera index 0 and 1
        try:
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                print(f"Could not open camera at index {idx}")
                continue
                
            # Check if the width is approximately double the height or significantly wide
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            print(f"Camera {idx} dimensions: {width}x{height}")
            
            # Get a test frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                print(f"Could not read frame from camera {idx}")
                continue
                
            # If width is significantly larger than height, might be side-by-side
            if width > height * 1.5:
                print(f"Camera {idx} has wide format (width={width}, height={height}), likely side-by-side")
                return "SIDE_BY_SIDE",idx
                
        except Exception as e:
            print(f"Error checking camera {idx}: {e}")
    
    print("Testing for two separate camera indices...")
    # Check if we have two separate camera indices
    cap1 = cv2.VideoCapture(0)
    cap2 = cv2.VideoCapture(1)
    
    has_cam0 = cap1.isOpened()
    has_cam1 = cap2.isOpened()
    
    if has_cam0:
        ret1, frame1 = cap1.read()
        has_cam0 = ret1 and frame1 is not None
        
    if has_cam1:
        ret2, frame2 = cap2.read()
        has_cam1 = ret2 and frame2 is not None
        
    cap1.release()
    cap2.release()
    
    if has_cam0 and has_cam1:
        print("Found two working camera indices (0 and 1)")
        return "TWO_INDICES"
    
    # If we found at least one camera, try the side-by-side approach
    if has_cam0 or has_cam1:
        print("Found one camera, will try to use it as side-by-side")
        return "SIDE_BY_SIDE"
        
    print("Could not detect any working camera configuration")
    return "SIDE_BY_SIDE",1  # Default to try something

def start_camera_safely(camera_thread, codecs):
    """Try to start camera with various codecs"""
    for codec in codecs:
        try:
            camera_thread.camera_fourcc = codec
            camera_thread.start()
            
            # Test if camera actually works
            for _ in range(5):  # Try to get frames a few times
                frame = camera_thread.next(black=True, wait=0.5)
                if frame is not None and not isinstance(frame, bool) and frame.size > 0:
                    print(f"Camera started successfully with codec: {codec}")
                    return True
                time.sleep(0.1)
                
            print(f"Camera started but couldn't get valid frames with codec: {codec}")
            camera_thread.stop()
            time.sleep(0.5)
            
        except Exception as e:
            print(f"Failed to start camera with codec {codec}: {e}")
            try:
                camera_thread.stop()
            except:
                pass
            time.sleep(0.5)
            
    return False

if __name__ == '__main__':
    run()
