import gradio as gr
import numpy as np
import supervision as sv
from ultralytics import YOLOWorld
from PIL import Image

model = YOLOWorld("yolov8m-world")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Callback function to process each frame (for videos)
def video_callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(frame.copy(), detections=detections)
    return label_annotator.annotate(annotated_frame, detections=detections, labels=labels)

def process_video(video_path, classes_list):
    model.set_classes(classes_list)

  
    output_path = "result_video.mp4"
    sv.process_video(
        source_path=video_path,
        target_path=output_path,
        callback=video_callback
    )
    return output_path

def process_image(image, classes_list):
    image_np = np.array(image)
    model.set_classes(classes_list)

    results = model(image_np)[0]
    detections = sv.Detections.from_ultralytics(results)
    annotated_image = box_annotator.annotate(image_np.copy(), detections=detections)
    return label_annotator.annotate(annotated_image, detections=detections, 
                                     labels=[results.names[class_id] for class_id in detections.class_id])

def gradio_interface(input_file, classes_input):
    classes_list = [cls.strip() for cls in classes_input.split(',')]
    
    if input_file.endswith(('.jpg', '.jpeg', '.png')):
        image = Image.open(input_file)  
        result_image = process_image(image, classes_list)
        return result_image, None 
    else:
        result_video = process_video(input_file, classes_list)
        return None, result_video 

gr.Interface(
    fn=gradio_interface,  # Connect the function here
    inputs=[
        gr.File(label="Upload Image or Video"), 
        gr.Textbox(label="Classes to detect (comma-separated)", placeholder="e.g., person, car, dog")  
    ],
    outputs=[
        gr.Image(label="Processed Image", type="numpy"), 
        gr.Video(label="Processed Video")  # For video output
    ],
    title="DetectAndTrack",
    description="Upload an image or video and specify object classes for detection and tracking using YOLOWorld and ByteTrack."
).launch()
