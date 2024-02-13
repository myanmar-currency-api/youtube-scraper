import os 
import re
import transformers
from PIL import Image, ImageDraw
import ffmpeg
from pytube import YouTube
import json
import time

# Load DL models 
from transformers import DonutProcessor, VisionEncoderDecoderModel
import torch
import easyocr
import random
import numpy as np
import csv
from torchvision import transforms


# LOAD up all the dl models

from transformers import TableTransformerForObjectDetection

reader = easyocr.Reader(['en']) # this needs to run only once to load the model into memory

device = "cuda" if torch.cuda.is_available() else "cpu"
model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
model.to(device)

print(device)


def download_youtube(url):
    yt = YouTube(url)

    yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first().download(filename="currency.mp4")

def extract_image_from_video():
    YOUR_FILE = 'currency.mp4'
    probe = ffmpeg.probe(YOUR_FILE)
    time = float(probe['streams'][0]['duration']) // 2
    width = probe['streams'][0]['width']

    parts = 1

    intervals = time // parts
    intervals = int(intervals)
    interval_list = [(i * intervals, (i + 1) * intervals) for i in range(parts)]
    i = 0

    interval_list

    for item in interval_list:
        (
            ffmpeg
            .input(YOUR_FILE, ss=item[1])
            .filter('crop','in_h-100','in_h-200','395','250')
            # .filter('scale', width, -1)
            .output('Image' + str(i) + '.jpg', vframes=1)
            .overwrite_output()
            .run()
        )
        i += 1

def draw_grid_on_image():
    image = Image.open("Image0.jpg").convert("RGB")
    # im = Image.new('RGBA', (400, 400), (0, 255, 0, 255)) 
    draw = ImageDraw.Draw(image) 
    line_color="#000"

    # This is pure hard coded grid until DVB make changes 
    draw.line((1500,70, 0,70), fill=line_color,width = 5 )
    draw.line((1500,130, 0,130), fill=line_color,width = 5)
    draw.line((1500,200, 0,200), fill=line_color,width = 5)
    draw.line((1500,250, 0,250), fill=line_color,width = 5)
    draw.line((1500,310, 0,310), fill=line_color,width = 5)
    draw.line((1500,370, 0,370), fill=line_color,width = 5)
    draw.line((1500,430, 0,430), fill=line_color,width = 5)

    draw.line((150,1000, 150,0), fill=line_color,width = 5)
    draw.line((350,1000, 350,0), fill=line_color,width = 5)

    image = image.crop((0, 10, 550, 440))
    return image


# Image Processors
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))

        return resized_image

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b


def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score),
                            'bbox': [float(elem) for elem in bbox]})

    return objects

# Image Processors Ends Here

def transform_image_using_torchvision(image):
    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    pixel_values = structure_transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)
    # Forward Passing 
    with torch.no_grad():
        outputs = model(pixel_values)
    return outputs

def extract_cell_informations(image,outputs):
    structure_id2label = model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    cells = outputs_to_objects(outputs, image.size, structure_id2label)
    return cells

def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates

def apply_ocr(cell_coordinates,image):
    # let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(cell_coordinates):
      row_text = []
      for cell in row["cells"]:
        # crop cell out of image
        cell_image = np.array(image.crop(cell["cell"]))
        # apply OCR
        result = reader.readtext(np.array(cell_image))
        if len(result) > 0:
          # print([x[1] for x in list(result)])
          text = " ".join([x[1] for x in result])
          row_text.append(text)

      if len(row_text) > max_num_columns:
          max_num_columns = len(row_text)

      data[idx] = list(map(lambda x: x.replace('v','').replace('V','').replace(' ',''), row_text))

    print("Max number of columns:", max_num_columns)

    # pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
          row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data

def get_myanmar_datetime():
    import datetime
 
    # Specifying Myanmar timedelta and timezone 
    # object
    mmTimeDelta = datetime.timedelta(hours=6.5)
    mmTZObject = datetime.timezone(mmTimeDelta,
                                    name="MMT")

    dt = datetime.datetime.now().astimezone(mmTZObject)
    return dt


def get_file_path():
    import datetime
 
    # Specifying Myanmar timedelta and timezone 
    # object
    mmTimeDelta = datetime.timedelta(hours=6.5)
    mmTZObject = datetime.timezone(mmTimeDelta,
                                    name="MMT")
    
    d1 = datetime.datetime.now().astimezone(mmTZObject)
     
    
    d2 = d1.strftime("%Y-%m-%d,%H")
     
    
    tod = "latest"
    return f'api/{d1.strftime("%Y")}/{d1.strftime("%m")}/{d1.strftime("%d")}/{tod}.json'

def write_results_to_json(data):
    
    # Add timestamp
    timestamp = get_myanmar_datetime().strftime("%Y-%m-%d %H:%M:%S")
    epoch_time = time.time()
    # Convert list to list of dictionaries with timestamp
    json_data = {"data":[{
        "currency": row[0],
        "buy": row[1],
        "sell": row[2]
    } for _,row in data.items()],"epoch": epoch_time,"timestamp": timestamp}

    # Save as JSON
    with open('data.json', 'w') as json_file:
        json.dump(json_data, json_file, indent=4)

def push_to_github():
    from github import Github

    # Define GitHub repository details
    repo_owner = 'myanmar-currency-api'
    repo_name = 'myanmar-currency-api.github.io'
    file_path = 'data.json'
    file_content = ''

    # GitHub personal access token with 'repo' scope
    access_token = os.getenv("GH_ACCESS_TOKEN") 

    # Create a GitHub instance using access token
    g = Github(access_token)


    try:
        with open(file_path, 'r') as file:
            file_content = file.read()
            print("File content:")
            print(file_content)
    except FileNotFoundError:
        print(f"File '{file_path}' not found.")
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

    # Get the repository
    repo = g.get_user(repo_owner).get_repo(repo_name)
    # Write to latest path
    latest_path='api/latest.json'
    try:
        contents = repo.get_contents(latest_path)
        repo.update_file(latest_path, "Update file", file_content, contents.sha)
        print(f"File '{latest_path}' updated successfully.")
    except Exception as e:
        repo.create_file(latest_path, "Create file", file_content)
        print(f"File '{latest_path}' created successfully.")
    # Create or update file
    try:
        contents = repo.get_contents(get_file_path())
        repo.update_file(get_file_path(), "Update file", file_content, contents.sha)
        print(f"File '{file_path}' updated successfully.")
    except Exception as e:
        repo.create_file(get_file_path(), "Create file", file_content)
        print(f"File '{file_path}' created successfully.")

def main():
    download_youtube(os.getenv("YOUTUBE_URL"))
    extract_image_from_video()
    image = draw_grid_on_image()
    outputs=transform_image_using_torchvision(image)
    cells = extract_cell_informations(image,outputs)
    cell_coordinates = get_cell_coordinates_by_row(cells)
    data = apply_ocr(cell_coordinates,image)
    write_results_to_json(data)
    push_to_github()
    for row, row_data in data.items():
        print(row_data)

main()
