# import spaces
import cv2
import numpy as np
import gradio as gr
import cwm.utils as utils
import os
# os.system("pip uninstall -y gradio")
# os.system("pip install gradio==4.31.0")

# Points color and arrow properties
arrow_color = (0, 255, 0)  # Green color for all arrows
dot_color = (0, 255, 0)  # Green color for the dots at start and end
dot_color_fixed = (255, 0, 0)  # Red color for zero-length vectors
thickness = 3  # Thickness of the arrow
tip_length = 0.3  # The length of the arrow tip relative to the arrow length
dot_radius = 7  # Radius for the dots
dot_thickness = -1  # Thickness for solid circle (-1 fills the circle)
from PIL import Image
import torch
import json
#load model
from cwm.model.model_factory import model_factory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Load CWM 3-frame model (automatically download pre-trained checkpoint)
model = model_factory.load_model('vitb_8x8patch_2frames_learnable_pos_embed').to(device)

model.requires_grad_(False)
model.eval()

model = model#.to(torch.float16)


import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
from PIL import Image
import numpy as np

from torchvision import transforms

import os

# @spaces.GPU(duration=110)
def get_c(x, points):
    x = utils.imagenet_normalize(x).to(device)
    with torch.no_grad():
        counterfactual = model.get_intervention_outcome(x, points)
    return counterfactual

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown('''# Scene editing interventions with Counterfactual World Models!
        ''')

    # Annotating arrows on an image
    with gr.Tab(label='Image'):
        with gr.Row():
            with gr.Column():
                # Input image
                original_image = gr.State(value=None)  # store original image without arrows
                original_image_high_res = gr.State(value=None)  # store original image without arrows
                input_image = gr.Image(type="numpy", label="Upload Image")

                # Annotate arrows
                selected_points = gr.State([])  # store points
                zero_length_toggle = gr.Checkbox(label="Select patches to be kept fixed", value=False)  # Toggle for zero-length vectors
                with gr.Row():
                    gr.Markdown('1. **Click on the image** to specify patch motion by selecting a start and end point. \n 2. After selecting the points to move, enable the **"Select patches to be kept fixed"** checkbox to choose a few points to keep fixed. \n 3. **Click "Run Model"** to visualize the result of the edit.')
                    undo_button = gr.Button('Undo last action')
                    clear_button = gr.Button('Clear All')

                # Run model button
                run_model_button = gr.Button('Run Model')

            # Show the image with the annotated arrows
            with gr.Tab(label='Intervention'):
                output_image = gr.Image(type='numpy')

        # Store the original image and resize to square size once uploaded
        def resize_to_square(img, size=512):
            print("Resizing image to square")
            img = Image.fromarray(img)
            transform = transforms.Compose([
                transforms.Resize((size, size)),
                # transforms.CenterCrop(size)
            ])
            img = transform(img)  # .transpose(1, 2, 0)

            return np.array(img)


        def load_img(evt: gr.SelectData):
            img_path = evt.value['image']['path']
            img = np.array(Image.open(img_path))
            #load_points
            with open('./assets/intervention_test_images/annot.json', 'r') as f:
                points_json = json.load(f)

            # print(f"Image uploaded with shape: {input.shape}")
            resized_img = resize_to_square(img)

            if os.path.basename(img_path) not in points_json:
                return resized_img, resized_img, img, []

            points_json = points_json[os.path.basename(img_path)]

            temp = resized_img.copy()

            # Redraw all remaining arrows and dots
            for i in range(0, len(points_json), 2):
                start_point = points_json[i]
                end_point = points_json[i + 1]
                if start_point == end_point:
                    # Zero-length vector: Draw a dot
                    color = dot_color_fixed
                else:
                    cv2.arrowedLine(temp, start_point, end_point, arrow_color, thickness, tipLength=tip_length,
                                    line_type=cv2.LINE_AA)
                    color = arrow_color
                # Draw arrow

                # Draw dots at start and end points
                cv2.circle(temp, start_point, dot_radius, color, dot_thickness, lineType=cv2.LINE_AA)
                cv2.circle(temp, end_point, dot_radius, color, dot_thickness, lineType=cv2.LINE_AA)

            # If there is an odd number of points (e.g., only a start point), draw a dot for it
            if len(points_json) == 1:
                start_point = points_json[0]
                cv2.circle(temp, start_point, dot_radius, dot_color, dot_thickness, lineType=cv2.LINE_AA)



            return temp, resized_img, img,  points_json


        def store_img(img):
            resized_img = resize_to_square(img)  # Resize the uploaded image to a square
            print(f"Image uploaded with shape: {resized_img.shape}")
            return resized_img, resized_img, img, []


        with gr.Row():
            with gr.Column():
                gallery = gr.Gallery(["./assets/intervention_test_images/ducks.jpg", "./assets/intervention_test_images/robot_arm.jpg", "./assets/intervention_test_images/bread.jpg",  "./assets/intervention_test_images/bird.jpg", "./assets/intervention_test_images/desk_1.jpg", "./assets/intervention_test_images/glasses.jpg", "./assets/intervention_test_images/watering_pot.jpg"], columns=5, allow_preview=False, label="Select an example image to test")

        gallery.select(load_img, outputs=[input_image, original_image, original_image_high_res, selected_points])

        input_image.upload(store_img, [input_image], [input_image, original_image, original_image_high_res, selected_points])

        # Get points and draw arrows or zero-length vectors based on the toggle
        def get_point(img, sel_pix, zero_length, evt: gr.SelectData):
            sel_pix.append(evt.index)  # Append the point's location (coordinates)

            # Zero-length vector case: Draw a single dot at the clicked point
            if zero_length:
                point = sel_pix[-1]  # Last point clicked
                cv2.circle(img, point, dot_radius, dot_color_fixed, dot_thickness, lineType=cv2.LINE_AA)  # Draw a dot at the point
                sel_pix.append(evt.index)
            else:
                # Regular case: two clicks for an arrow
                # Check if this is the first point (start point for the arrow)
                if len(sel_pix) % 2 == 1:
                    # Draw a dot at the start point to give feedback
                    start_point = sel_pix[-1]  # Last point is the start
                    cv2.circle(img, start_point, dot_radius, dot_color, dot_thickness, lineType=cv2.LINE_AA)

                # Check if two points have been selected (start and end points for an arrow)
                if len(sel_pix) % 2 == 0:
                    # Draw an arrow between the last two points
                    start_point = sel_pix[-2]  # Second last point is the start
                    end_point = sel_pix[-1]  # Last point is the end

                    # Draw arrow
                    cv2.arrowedLine(img, start_point, end_point, arrow_color, thickness, tipLength=tip_length, line_type=cv2.LINE_AA)

                    # Draw a dot at the end point
                    cv2.circle(img, end_point, dot_radius, dot_color, dot_thickness,  lineType=cv2.LINE_AA)

            return img if isinstance(img, np.ndarray) else np.array(img)

        input_image.select(get_point, [input_image, selected_points, zero_length_toggle], [input_image])

        # Undo the last selected action
        def undo_arrows(orig_img, sel_pix, zero_length):
            temp = orig_img.copy()

            if len(sel_pix) >= 2:
                sel_pix.pop()  # Remove the last end point
                sel_pix.pop()  # Remove the last start point

            # Redraw all remaining arrows and dots
            for i in range(0, len(sel_pix), 2):
                start_point = sel_pix[i]
                end_point = sel_pix[i + 1]
                if start_point == end_point:
                    # Zero-length vector: Draw a dot
                    color = dot_color_fixed
                else:
                    cv2.arrowedLine(temp, start_point, end_point, arrow_color, thickness, tipLength=tip_length, line_type=cv2.LINE_AA)
                    color = arrow_color
                # Draw arrow

                # Draw dots at start and end points
                cv2.circle(temp, start_point, dot_radius, color, dot_thickness, lineType=cv2.LINE_AA)
                cv2.circle(temp, end_point, dot_radius, color, dot_thickness, lineType=cv2.LINE_AA)

            # If there is an odd number of points (e.g., only a start point), draw a dot for it
            if len(sel_pix) == 1:
                start_point = sel_pix[0]
                cv2.circle(temp, start_point, dot_radius, dot_color, dot_thickness, lineType=cv2.LINE_AA)

            return temp if isinstance(temp, np.ndarray) else np.array(temp)

        undo_button.click(undo_arrows, [original_image, selected_points, zero_length_toggle], [input_image])


        # Clear all points and reset the image
        def clear_all_points(orig_img, sel_pix):
            sel_pix.clear()  # Clear all points
            return orig_img  # Reset image to original

        clear_button.click(clear_all_points, [original_image, selected_points], [input_image])

        # Dummy model function to simulate running a model
        def run_model_on_points(points, input_image, original_image):
            H = input_image.shape[0]
            W = input_image.shape[1]
            factor = 256/H
            # Example: pretend the model processes points and returns a simple transformation on the image
            print("Running model on points:", points)
            points = torch.from_numpy(np.array(points).reshape(-1, 4)) * factor

            points = points[:, [1, 0, 3, 2]]

            img = Image.fromarray(original_image)

            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                # transforms.CenterCrop(256)
            ])
            img = np.array(transform(img))

            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            img = img[None]

            # reshape image to [B, C, T, H, W], C = 3, T = 3 (3-frame model), H = W = 224
            x = img[:, :, None].expand(-1, -1, 2, -1, -1)#.to(torch.float16)

            counterfactual = get_c(x, points)


            counterfactual = counterfactual.squeeze()

            counterfactual = counterfactual.clamp(0, 1).permute(1,2,0).detach().cpu().numpy()

            return counterfactual

        # Run model when the button is clicked
        run_model_button.click(run_model_on_points, [selected_points, input_image, original_image_high_res], [output_image])



    # Launch the app
demo.queue().launch(inbrowser=True, share=True)
