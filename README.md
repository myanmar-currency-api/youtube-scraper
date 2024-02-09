# DVB TV News Currency Video Youtube-scraper
Unofficial Youtube Scraper of Currency Update News from DVB

## Technical Documentation

### Overview
This Python script is designed to extract currency exchange rate data from a video source, process it using various deep learning models, perform optical character recognition (OCR) to extract text information, and then save the results to a JSON file. Finally, it pushes the JSON data to a GitHub repository.

### Motivation
There exists inconsistency between the currency exchange rates displayed on Google and the actual rates of Myanmar Currency. To address this issue and provide accurate and up-to-date currency exchange rate data, this script utilizes deep learning models and optical character recognition (OCR) to extract currency information from video sources. By leveraging advanced technologies, the script aims to offer reliable exchange rate data for developers and users.

### Dependencies
The script requires several Python libraries, including:
- `os`: For interacting with the operating system.
- `transformers`: For loading deep learning models.
- `PIL`: For image processing tasks.
- `ffmpeg`: For extracting frames from a video file.
- `pytube`: For downloading YouTube videos.
- `json`: For handling JSON data.
- `time`: For working with timestamps.
- `torch`: For deep learning tasks.
- `easyocr`: For optical character recognition.
- `numpy`: For numerical operations.
- `csv`: For working with CSV files.
- `github`: For interacting with GitHub repositories.

### Workflow
1. **Video Download**: The script downloads a YouTube video from the specified URL using the `pytube` library.
   
2. **Image Extraction**: It extracts frames from the downloaded video using `ffmpeg`. Each frame is processed to draw a grid over the region of interest.
   
3. **Grid Drawing**: A grid is drawn over the first extracted image, which represents the table structure of the currency exchange rates.
   
4. **Deep Learning Model Initialization**: The script loads the required deep learning models for table structure recognition and object detection. In this case, the model used is `TableTransformerForObjectDetection` for table structure recognition.
   
5. **Image Transformation**: Each extracted image is transformed using `torchvision` to prepare it for input into the object detection model.
   
6. **Object Detection**: The transformed image is fed into the object detection model to identify table cells.
   
7. **OCR and Data Extraction**: Optical character recognition (OCR) is performed on each identified table cell to extract text information.

### License
This script is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

### Additional Notes
- The script assumes a specific structure of the table in the video frames and relies on hard-coded grid coordinates. Any changes in the table layout may require adjustments to the grid drawing process.
- The deep learning models used in the script are pretrained models for table structure recognition (`TableTransformerForObjectDetection`) and object detection. They are expected to perform well on similar tasks but may not generalize perfectly to all scenarios.
- Error handling for various stages of the process, such as video download failures or OCR errors, is not extensively covered in the script and may need to be implemented based on specific requirements.
