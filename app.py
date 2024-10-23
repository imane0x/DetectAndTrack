import gradio as gr
import numpy as np
import supervision as sv
from ultralytics import YOLOWorld

# Initialize the YOLO model, tracker, and annotators
model = YOLOWorld("yolov8m-world")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Callback function to process each frame
def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    return label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)

# Function to process video with supervision
def process_video_gradio(video_path, classes_list):
    # Set the user-defined classes to detect
    model.set_classes(classes_list)

    # Define the output path
    output_path = "result.mp4"
    # Process the video using supervision's process_video
    sv.process_video(
        source_path=video_path,
        target_path=output_path,
        callback=callback
    )
    return output_path

# Gradio interface function
def gradio_interface(video, classes_input):
    # Split the user-provided classes by commas and remove whitespace
    classes_list = [cls.strip() for cls in classes_input.split(',')]

    video_path = video  # Use video directly as the file path
    result_path = process_video_gradio(video_path, classes_list)
    return result_path

# Gradio app setup
gr.Interface(
    fn=gradio_interface,  # Connect the function here
    inputs=[
        gr.Video(),  # Input video file
        gr.Textbox(label="Classes to detect (comma-separated)", placeholder="e.g., person, car, dog")  # Input classes list
    ],
    outputs=gr.Video(),   # Output the processed video
    title="YOLOWorld Object Tracking with Supervision",
    description="Upload a video and specify object classes for detection and tracking using YOLOWorld and Supervision."
).launch()
