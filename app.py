from predict import Predictor, model_cfg
from PIL import Image
import gradio as gr

# set a lot of global variables

predictor = None
vocabulary = ["bat man"]
input_image: Image.Image = None
outputs: dict = None
cur_model_name: str = None


def set_vocabulary(text):
    global vocabulary
    vocabulary = text.split(",")
    print("set vocabulary to", vocabulary)


def set_input(image):
    global input_image
    input_image = image
    print("set input image to", image)


def set_predictor(model_name: str):
    global cur_model_name
    if cur_model_name == model_name:
        return
    global predictor
    predictor = Predictor(**model_cfg[model_name])
    print("set predictor to", model_name)
    cur_model_name = model_name


set_predictor(list(model_cfg.keys())[0])


# for visualization
def visualize(vis_mode):
    if outputs is None:
        return None
    return predictor.visualize(**outputs, mode=vis_mode)


def segment_image(vis_mode, voc_mode, model_name):
    set_predictor(model_name)
    if input_image is None:
        return None
    global outputs
    result = predictor.predict(
        input_image, vocabulary=vocabulary, augment_vocabulary=voc_mode
    )
    outputs = result

    return visualize(vis_mode)


def segment_e2e(image, vis_mode):
    set_input(image)
    return segment_image(vis_mode)


# gradio

with gr.Blocks(
    css="""
               #submit {background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px;width: 20%;margin: 0 auto; display: block;}
 
                """
) as demo:
    gr.Markdown(
        f"<h1 style='text-align: center; margin-bottom: 1rem'>Side Adapter Network for Open-Vocabulary Semantic Segmentation</h1>"
    )
    gr.Markdown(
        """   
    This is the demo for our conference paper : "[Side Adapter Network for Open-Vocabulary Semantic Segmentation](https://arxiv.org/abs/2302.12242)".
    """
    )
    # gr.Image(type="pil", value="./resources/arch.png", shape=(460, 200), elem_id="arch")
    gr.Markdown(
        """
        ---
        """
    )
    with gr.Row():
        image = gr.Image(type="pil", elem_id="input_image")
        plt = gr.Image(type="pil", elem_id="output_image")

    with gr.Row():
        model_name = gr.Dropdown(
            list(model_cfg.keys()), label="Model", value="san_vit_b_16"
        )
        augment_vocabulary = gr.Dropdown(
            ["COCO-all", "COCO-stuff"],
            label="Vocabulary Expansion",
            value="COCO-all",
        )
        vis_mode = gr.Dropdown(
            ["overlay", "mask"], label="Visualization Mode", value="overlay"
        )
    object_names = gr.Textbox(value=",".join(vocabulary), label="Object Names", lines=5)

    button = gr.Button("Run", elem_id="submit")
    #

    object_names.change(set_vocabulary, [object_names], queue=False)
    image.change(set_input, [image], queue=False)
    vis_mode.change(visualize, [vis_mode], plt, queue=False)
    button.click(
        segment_image, [vis_mode, augment_vocabulary, model_name], plt, queue=False
    )
    demo.load(
        segment_image, [vis_mode, augment_vocabulary, model_name], plt, queue=False
    )

demo.queue().launch()
